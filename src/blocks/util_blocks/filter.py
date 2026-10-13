import logging
import math
import re
from typing import Callable, Literal, Optional

import polars as pl
from pydantic import BaseModel, Field, model_validator

from chaperone import BlockFunction
from chaperone.data.schema import Column, StructureColumn

logger = logging.getLogger(__name__)

# Common misspellings of ``filter_groups`` that Pydantic's default
# ``extra="ignore"`` would silently drop -> no group filtering -> more rows
# pass than intended (WS7-2). Rejected loudly by FilterBlock.
_GROUP_FILTER_TYPOS = ("group_filters", "filters_groups")


class MetricFilter(BaseModel):
    """A min/max range filter and/or quantile filter for a single metric.

    The dotted metric key (e.g. "rasa.ligand_atom_rasa.N1") is translated to a
    JSONPath query on the packed metrics column so filtering stays in Polars/Rust.

    Quantile filtering keeps the best *fraction* of rows by this metric.
    Set ``quantile_direction`` to indicate which end is "better":
    ``"lower"`` for metrics like ddG/RMSD, ``"higher"`` for pLDDT/contact_score.

    When ``parent_based=True``, quantile is computed per parent structure
    (grouped by ``structures[-2].path``), ensuring every backbone retains
    its best fraction of sequences.
    """

    min: Optional[float] = None
    max: Optional[float] = None
    quantile: Optional[float] = Field(
        default=None,
        description=(
            "Keep the best fraction of rows by this metric. "
            "E.g. 0.4 keeps the best 40%. Must be in (0.0, 1.0]."
        ),
    )
    quantile_direction: Literal["lower", "higher"] = Field(
        default="lower",
        description=(
            "Which direction is 'better': 'lower' (ddG, RMSD) or "
            "'higher' (pLDDT, contact_score)."
        ),
    )
    parent_based: bool = Field(
        default=False,
        description=(
            "When True, quantile is applied per parent group "
            "(grouped by structures[-2].path). When False, quantile is global."
        ),
    )

    @model_validator(mode="after")
    def _validate_quantile(self):
        if self.quantile is not None and not (0.0 < self.quantile <= 1.0):
            raise ValueError(f"quantile must be in (0.0, 1.0], got {self.quantile}")
        return self


def _dotted_key_to_jsonpath(key: str) -> str:
    """Convert "block.inner_key" to JSONPath with bracket notation.

    Splits on the first dot only: the left part is the block name, the right
    part is the metric key (which may itself contain literal dots).
    """
    if "." not in key:
        return f"$['{key}']"
    block_name, metric_key = key.split(".", 1)
    return f"$['{block_name}']['{metric_key}']"


# ── Boolean filter-expression DSL ────────────────────────────────────────
#
# A compact string language over metric keys that compiles to the SAME single
# Polars boolean expression the ``filters`` dict produces, but with arbitrary
# ``and`` / ``or`` / ``not`` nesting and parentheses. This lets an OR-of-ANDs
# pocket rule be written inline instead of being hand-distributed into the
# fixed ``filter_groups`` OR-pool. Example:
#
#   "af3.min_pae_interaction <= 10
#    and ( (h.nicotinamide >= 2 and a.num_aromatic_neighbors >= 1)
#       or (h.nicotinamide >= 1 and a.num_aromatic_neighbors >= 2) )
#    and ( r.ligand_atom_rasa.C21 >= 0.1 or r.ligand_atom_rasa.C22 >= 0.1 )"
#
# Grammar (precedence low -> high; standard: or < and < not):
#   or_expr    := and_expr ('or'  and_expr)*
#   and_expr   := not_expr ('and' not_expr)*
#   not_expr   := 'not' not_expr | atom
#   atom       := '(' or_expr ')' | comparison
#   comparison := KEY OP NUMBER
#   KEY        := bare metric key [A-Za-z_][A-Za-z0-9_./]*  (dots split on the
#                 first dot only, exactly like the ``filters`` dict keys, so
#                 both namespaces ("opt/cycle_0/af3") and flat inner keys
#                 ("ligand_atom_rasa.C21") work) -- or a `backtick-quoted` key
#                 for exotic characters.
#   OP         := '>=' | '<=' | '==' | '!=' | '>' | '<'
#   NUMBER     := optionally-signed int/float literal (negatives for ddG etc.)
#
# Null handling matches the ``filters`` dict: a missing/non-numeric metric
# yields null, Polars three-valued logic propagates it (so ``a OR b`` still
# passes when a present branch is True even if the other is null), and the
# final ``.filter(mask)`` drops any row whose overall mask is null.

_EXPR_TOKEN_RE = re.compile(
    r"""
      \s+
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
    | (?P<OP>>=|<=|==|!=|>|<)
    | (?P<NUMBER>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)
    | (?P<QKEY>`[^`]+`)
    | (?P<WORD>[A-Za-z_][A-Za-z0-9_./]*)
    """,
    re.VERBOSE,
)

_EXPR_KEYWORDS = {"and": "AND", "or": "OR", "not": "NOT"}

_EXPR_OP_FUNCS = {
    ">=": lambda v, x: v >= x,
    "<=": lambda v, x: v <= x,
    ">": lambda v, x: v > x,
    "<": lambda v, x: v < x,
    "==": lambda v, x: v == x,
    "!=": lambda v, x: v != x,
}


def _tokenize_filter_expr(expr: str) -> list[tuple[str, object]]:
    """Tokenize a filter-expression string into (type, value) tuples."""
    tokens: list[tuple[str, object]] = []
    i, n = 0, len(expr)
    while i < n:
        m = _EXPR_TOKEN_RE.match(expr, i)
        if not m:
            raise ValueError(
                f"Filter expr: unexpected character {expr[i]!r} at position {i} "
                f"in {expr!r}"
            )
        i = m.end()
        kind = m.lastgroup
        if kind is None:  # whitespace alternative (no capture group)
            continue
        val = m.group()
        if kind == "WORD":
            kw = _EXPR_KEYWORDS.get(val)
            tokens.append((kw, val) if kw else ("KEY", val))
        elif kind == "QKEY":
            tokens.append(("KEY", val[1:-1]))  # strip surrounding backticks
        elif kind == "NUMBER":
            tokens.append(("NUMBER", float(val)))
        else:  # LPAREN / RPAREN / OP
            tokens.append((kind, val))
    tokens.append(("EOF", None))
    return tokens


def _compile_comparison(key: str, op: str, num: float) -> pl.Expr:
    """Compile a single ``KEY OP NUMBER`` comparison to a Polars expression."""
    jp = _dotted_key_to_jsonpath(key)
    value = (
        pl.col(Column.METRICS).str.json_path_match(jp).cast(pl.Float64, strict=False)
    )
    return _EXPR_OP_FUNCS[op](value, num)


class _FilterExprParser:
    """Recursive-descent parser: token stream -> single Polars boolean expr."""

    def __init__(self, tokens: list[tuple[str, object]], source: str):
        self.toks = tokens
        self.src = source
        self.i = 0

    def _peek(self) -> tuple[str, object]:
        return self.toks[self.i]

    def _advance(self) -> tuple[str, object]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def _expect(self, type_: str) -> tuple[str, object]:
        tok = self._advance()
        if tok[0] != type_:
            raise ValueError(
                f"Filter expr: expected {type_} but got {tok[0]} ({tok[1]!r}) "
                f"in {self.src!r}"
            )
        return tok

    def parse(self) -> pl.Expr:
        expr = self._or()
        self._expect("EOF")
        return expr

    def _or(self) -> pl.Expr:
        expr = self._and()
        while self._peek()[0] == "OR":
            self._advance()
            expr = expr | self._and()
        return expr

    def _and(self) -> pl.Expr:
        expr = self._not()
        while self._peek()[0] == "AND":
            self._advance()
            expr = expr & self._not()
        return expr

    def _not(self) -> pl.Expr:
        if self._peek()[0] == "NOT":
            self._advance()
            return ~self._not()
        return self._atom()

    def _atom(self) -> pl.Expr:
        if self._peek()[0] == "LPAREN":
            self._advance()
            expr = self._or()
            self._expect("RPAREN")
            return expr
        key = self._expect("KEY")[1]
        op = self._expect("OP")[1]
        num = self._expect("NUMBER")[1]
        return _compile_comparison(key, op, num)


def compile_filter_expression(expr: str) -> pl.Expr:
    """Compile a boolean filter-expression string into one Polars expression.

    See the DSL grammar above. The result is AND-combinable with the masks
    built from ``filters`` / ``filter_groups``.
    """
    tokens = _tokenize_filter_expr(expr)
    if tokens[0][0] == "EOF":
        raise ValueError(f"Filter expr is empty: {expr!r}")
    return _FilterExprParser(tokens, expr).parse()


def _filter_expression_keys(expr: str) -> list[str]:
    """Return the metric keys referenced by a filter-expression (deduped)."""
    seen: set[str] = set()
    keys: list[str] = []
    for kind, val in _tokenize_filter_expr(expr):
        if kind == "KEY" and val not in seen:
            seen.add(val)  # type: ignore[arg-type]
            keys.append(val)  # type: ignore[arg-type]
    return keys


def _build_filter_expr(
    filters: dict[str, MetricFilter],
    operator: Literal["and", "or"] = "and",
) -> pl.Expr:
    """Build a single Polars boolean expression from a filters dict."""
    expr: pl.Expr | None = None
    col = pl.col(Column.METRICS)

    for key, mf in filters.items():
        jp = _dotted_key_to_jsonpath(key)
        value = col.str.json_path_match(jp).cast(pl.Float64)

        if mf.min is not None:
            clause = value >= mf.min
            expr = (
                clause
                if expr is None
                else (expr & clause if operator == "and" else expr | clause)
            )
        if mf.max is not None:
            clause = value <= mf.max
            expr = (
                clause
                if expr is None
                else (expr & clause if operator == "and" else expr | clause)
            )

    if expr is None:
        return pl.lit(True)
    return expr


def _build_group_filter_expr(
    filter_groups: dict[str, dict[str, MetricFilter]],
    inner_operator: Literal["and", "or"] = "and",
) -> pl.Expr:
    """Build OR-of-ANDs expression from named filter groups.

    Each group's filters are combined with ``inner_operator``, then groups are OR'd.
    """
    group_expr: pl.Expr | None = None
    for group_name, group_filters in filter_groups.items():
        inner = _build_filter_expr(group_filters, operator=inner_operator)
        group_expr = inner if group_expr is None else group_expr | inner
    if group_expr is None:
        return pl.lit(True)
    return group_expr


def _build_quantile_mask(
    df: pl.DataFrame,
    filters: dict[str, MetricFilter],
) -> pl.Series:
    """Build a boolean mask for quantile filters across the DataFrame.

    For each metric with ``quantile`` set, ranks rows (globally or per-parent)
    and keeps the best fraction.  All per-metric masks are AND'd together.
    """
    mask: pl.Series | None = None
    col = pl.col(Column.METRICS)

    for key, mf in filters.items():
        if mf.quantile is None:
            continue

        jp = _dotted_key_to_jsonpath(key)
        val_expr = col.str.json_path_match(jp).cast(pl.Float64)
        descending = mf.quantile_direction == "higher"

        # Guard the all-null ranked-column failure mode: rank() returns null for
        # every null row and count() excludes nulls, so an all-null column yields
        # n_total=0 -> mask all-null -> fill_null(False) silently drops every row
        # regardless of the quantile fraction (the "even 0.6 drops everything"
        # footgun). Distinguish the two causes:
        #   - df empty (e.g. an upstream expr pruned to 0 rows): nothing to rank,
        #     so keep-all rather than annihilate.
        #   - df non-empty but the key resolves all-null (typo / mis-resolved
        #     namespace / metric nested deeper than one level): unfilterable as
        #     written, so raise instead of silently dropping everything.
        if df.height == 0:
            continue
        n_non_null = df.select(val_expr.is_not_null().sum())[0, 0]
        if n_non_null == 0:
            raise ValueError(
                f"Quantile filter key '{key}' (JSONPath: {jp}) resolved null for "
                f"all {df.height} rows entering the quantile phase - it cannot be "
                f"ranked and would silently drop every row. Check for a typo, or "
                f"that the metric is not nested deeper than one level (dotted keys "
                f"split on the first dot only and metrics flatten one level)."
            )

        if mf.parent_based:
            parent_key_expr = (
                pl.col(Column.STRUCTURES)
                .list.get(-2, null_on_oob=True)
                .struct.field(StructureColumn.PATH)
                .fill_null("__no_parent__")
            )
            work = df.with_columns(
                parent_key_expr.alias("_qf_parent"),
                val_expr.alias("_qf_val"),
            )
            rank_expr = (
                pl.col("_qf_val")
                .rank("ordinal", descending=descending)
                .over("_qf_parent")
            )
            count_expr = pl.col("_qf_val").count().over("_qf_parent")
            n_keep_expr = pl.max_horizontal(
                pl.lit(1),
                (count_expr.cast(pl.Float64) * mf.quantile).ceil().cast(pl.Int64),
            )
            q_mask = work.select((rank_expr <= n_keep_expr).alias("_qf_mask"))[
                "_qf_mask"
            ]
        else:
            work = df.with_columns(val_expr.alias("_qf_val"))
            rank_expr = pl.col("_qf_val").rank("ordinal", descending=descending)
            n_total = work.select(pl.col("_qf_val").count())[0, 0]
            n_keep = max(1, math.ceil(n_total * mf.quantile))
            q_mask = work.select((rank_expr <= n_keep).alias("_qf_mask"))["_qf_mask"]

        q_mask = q_mask.fill_null(False)
        mask = q_mask if mask is None else mask & q_mask

    if mask is None:
        return pl.Series("_qf_mask", [True] * df.height)
    return mask


class FilterBlock(BlockFunction):
    """Filter pipeline rows by metric range filters or a predicate function.

    Supports three modes (the first two compile to Polars and may be combined):
      1. ``expr`` string  -- boolean DSL with ``and``/``or``/``not``/parens over
         metric keys, e.g. ``"a >= 2 and (b >= 1 or c >= 1)"`` (preferred for
         anything beyond a flat AND -- replaces hand-distributed
         ``filter_groups``). See ``compile_filter_expression`` for the grammar.
      2. ``filters`` dict (+ ``filter_groups``) -- min/max/quantile per key.
      3. ``function`` callable -- row-wise Python predicate (legacy).

    When both ``expr`` and ``filters``/``filter_groups`` are given they are
    AND-combined. Quantile filtering lives only on ``filters`` (it is a ranking
    phase, not a boolean threshold, so it has no DSL form).
    """

    _name = "filter"
    _display_name = "Filter"
    _description = "Filter pipeline rows by metrics predicate"
    _inline = True
    _chunked = False

    expr: Optional[str] = Field(
        default=None,
        description=(
            "Boolean filter expression with and/or/not and parentheses over "
            "metric keys, e.g. 'af3.min_pae_interaction <= 10 and (h.nicotinamide "
            ">= 2 or a.num_aromatic_neighbors >= 2)'. Compiles to a single Polars "
            "mask; AND-combined with 'filters'/'filter_groups' when both are set."
        ),
    )
    filters: Optional[dict[str, MetricFilter]] = Field(
        default=None,
        description="Metric key -> MetricFilter(min/max). Uses Polars-native fast path.",
    )
    filter_groups: Optional[dict[str, dict[str, MetricFilter]]] = Field(
        default=None,
        description=(
            "Named groups of filters combined with OR logic. "
            "Filters within each group use 'filter_groups_inner_operator'; "
            "groups are OR'd. "
            "Combined with base 'filters' via AND."
        ),
    )
    filter_groups_inner_operator: Literal["and", "or"] = Field(
        default="and",
        description=(
            "How filters within each filter_group are combined before groups "
            "are OR'd together."
        ),
    )
    function: Optional[Callable[[dict], bool]] = Field(
        default=None,
        description="Predicate function: metrics dict -> bool (True = keep)",
        exclude=True,
    )
    check_column_exists_only: bool = Field(
        default=False,
        description=(
            "When True, validate that filter columns exist and contain valid "
            "floats (not NaN) but do not prune rows. Useful for testing "
            "pipelines end-to-end with minimal data."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_group_filter_typos(cls, data):
        """Raise on misspelled ``filter_groups`` keys.

        The base ``BlockFunction`` model uses Pydantic's default
        ``extra="ignore"`` (changing that to ``extra="forbid"`` is a
        stable-API risk), so a typo'd ``group_filters=`` would otherwise be
        dropped silently. We only guard the known ``filter_groups`` typos here
        rather than forbidding all extras.
        """
        if isinstance(data, dict):
            for typo in _GROUP_FILTER_TYPOS:
                if typo in data:
                    raise ValueError(
                        f"FilterBlock received '{typo}' which is not a valid "
                        f"field and would be silently ignored. Did you mean "
                        f"'filter_groups'?"
                    )
        return data

    def _validate_filter_keys(self, handler, all_filters: dict[str, MetricFilter]):
        """Check that each filter key exists and contains valid floats."""
        col = pl.col(Column.METRICS)
        for key, mf in all_filters.items():
            jp = _dotted_key_to_jsonpath(key)
            values = handler._df.select(
                col.str.json_path_match(jp).cast(pl.Float64, strict=False)
            ).to_series()
            non_null = values.is_not_null().sum()
            if non_null == 0:
                raise ValueError(
                    f"FilterBlock '{self.name}': key '{key}' (JSONPath: {jp}) "
                    f"returned null for all {handler._df.height} rows - "
                    f"check metric key naming"
                )
            n_nan = values.is_nan().sum()
            if n_nan > 0:
                raise ValueError(
                    f"FilterBlock '{self.name}': key '{key}' has "
                    f"{n_nan}/{handler._df.height} NaN values"
                )
            logger.info(
                f"FilterBlock '{self.name}': key '{key}' validated "
                f"({non_null}/{handler._df.height} non-null values)"
            )

    def _warn_all_null_keys(self, handler):
        """Warn once per filter key that matches no non-null value (WS6-7).

        Collects every key across ``filters`` and ``filter_groups`` (deduped),
        and for each one that resolves to null across all rows, emits a single
        warning. Because the dotted-key JSONPath splits on the first dot only
        (``_dotted_key_to_jsonpath``) and the metrics column is flattened just
        one level deep, a metric nested 3+ deep filtered with a 2-deep key, or
        any typo'd key, resolves to all-null and would otherwise silently drop
        every row -- the recurring "filter passes 0 rows" footgun.
        """
        seen: set[str] = set()
        keys: list[str] = []
        for key in self.filters or {}:
            if key not in seen:
                seen.add(key)
                keys.append(key)
        for group_filters in (self.filter_groups or {}).values():
            for key in group_filters:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        if self.expr:
            for key in _filter_expression_keys(self.expr):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)

        n_rows = handler._df.height
        for key in keys:
            jp = _dotted_key_to_jsonpath(key)
            non_null = handler._df.select(
                pl.col(Column.METRICS).str.json_path_match(jp).is_not_null().sum()
            )[0, 0]
            if non_null == 0:
                logger.warning(
                    "FilterBlock '%s': key '%s' (JSONPath: %s) returned null for "
                    "all %d rows - this key will drop every row. Check for a "
                    "typo, or that the metric is not nested deeper than one level "
                    "(dotted keys split on the first dot only and metrics flatten "
                    "one level), in which case it is unfilterable as written.",
                    self.name,
                    key,
                    jp,
                    n_rows,
                )

    def _log_per_metric_drops(self, df: pl.DataFrame):
        """Log how many rows each top-level ``filters`` key alone would drop.

        Reuses the JSONPath built by ``_dotted_key_to_jsonpath`` and evaluates
        each metric's min/max clause independently against ``df`` (WS7-4). This
        is diagnostic only -- the actual pruning still applies the combined
        mask. Quantile contributions are reported separately in Phase 2.
        """
        n_rows = df.height
        col = pl.col(Column.METRICS)
        for key, mf in (self.filters or {}).items():
            if mf.min is None and mf.max is None:
                continue
            jp = _dotted_key_to_jsonpath(key)
            value = col.str.json_path_match(jp).cast(pl.Float64, strict=False)
            clause: pl.Expr | None = None
            if mf.min is not None:
                clause = value >= mf.min
            if mf.max is not None:
                c = value <= mf.max
                clause = c if clause is None else (clause & c)
            # Null values fail the clause; count them as drops.
            n_pass = df.select(clause.fill_null(False).sum())[0, 0]
            logger.info(
                "FilterBlock '%s': key '%s' (min=%s, max=%s) alone keeps "
                "%d/%d rows (drops %d)",
                self.name,
                key,
                mf.min,
                mf.max,
                n_pass,
                n_rows,
                n_rows - n_pass,
            )

    def __call__(self, handler, sandbox_dir=None, **kwargs):
        if self.check_column_exists_only:
            all_filters = dict(self.filters or {})
            if self.filter_groups:
                for group_filters in self.filter_groups.values():
                    all_filters.update(group_filters)
            if self.expr:
                for key in _filter_expression_keys(self.expr):
                    all_filters.setdefault(key, MetricFilter())
            self._validate_filter_keys(handler, all_filters)
            logger.info(
                f"FilterBlock '{self.name}': check_column_exists_only=True, "
                f"returning all {handler._df.height} rows unchanged"
            )
            return handler

        if (
            self.filters is not None
            or self.filter_groups is not None
            or self.expr is not None
        ):
            # Guard: quantile in filter_groups is not supported
            if self.filter_groups:
                for gname, gfilters in self.filter_groups.items():
                    for gkey, gmf in gfilters.items():
                        if gmf.quantile is not None:
                            raise ValueError(
                                f"Quantile filters in filter_groups (group '{gname}', "
                                f"key '{gkey}') are not supported. Use the top-level "
                                f"'filters' dict for quantile filtering."
                            )

            # WS6-7: warn once per filter key that matches no non-null value
            # across rows. The dotted-key JSONPath splits on the first dot only
            # and metrics flatten only one level, so a wrong-depth or typo'd
            # key yields all-null and silently drops every row.
            self._warn_all_null_keys(handler)

            # Phase 1: min/max threshold filtering
            base_expr = (
                _build_filter_expr(self.filters) if self.filters else pl.lit(True)
            )
            group_expr = (
                _build_group_filter_expr(
                    self.filter_groups,
                    inner_operator=self.filter_groups_inner_operator,
                )
                if self.filter_groups
                else pl.lit(True)
            )
            expr_mask = (
                compile_filter_expression(self.expr) if self.expr else pl.lit(True)
            )
            mask = base_expr & group_expr & expr_mask
            # WS7-4: log each top-level filter key's individual contribution
            # (rows it alone would drop) before applying the combined mask.
            if self.filters:
                self._log_per_metric_drops(handler._df)
            filtered_df = handler._df.filter(mask)

            # Phase 2: quantile filtering (requires cross-row context)
            if self.filters and any(
                mf.quantile is not None for mf in self.filters.values()
            ):
                n_before_q = filtered_df.height
                q_mask = _build_quantile_mask(filtered_df, self.filters)
                filtered_df = filtered_df.filter(q_mask)
                logger.info(
                    "FilterBlock '%s': %d/%d rows pass quantile filters",
                    self.name,
                    filtered_df.height,
                    n_before_q,
                )

            from chaperone import PipelineDataHandler

            filtered = PipelineDataHandler.__new__(PipelineDataHandler)
            filtered._df = filtered_df
        elif self.function is not None:
            filtered = handler.filter(self.function)
        else:
            raise ValueError(
                "FilterBlock requires one of 'expr', 'filters', or 'function'."
            )

        from chaperone import PipelineDataHandler

        n_before = handler._df.height
        n_after = filtered._df.height
        logger.info(
            f"FilterBlock '{self.name}': {n_after}/{n_before} rows pass filters"
        )

        if n_after == 0:
            logger.warning(f"FilterBlock '{self.name}': all rows filtered out")
            return PipelineDataHandler(filtered._df)

        # WS7-3: Append a no-op structure entry naming this filter step.
        #
        # This entry is LOAD-BEARING and intentionally kept (we do NOT set
        # _creates_new_step=False): downstream blocks reference the filter step
        # by name as an input, and the lineage tip must carry the filter's
        # name. The path is identical to the parent's (a filter doesn't move
        # atoms). The cost is that each filter inflates the per-row
        # `structures` list: a negative `which=` index used by e.g. RMSD must
        # be shifted back by one per filter that sits *after* the structure of
        # interest. See `filterblock_appends_structure_entries.md`.
        rows = []
        for row in filtered._df.iter_rows(named=True):
            structures = list(row[Column.STRUCTURES])
            last_path = structures[-1][StructureColumn.PATH] if structures else ""
            structures.append(
                {
                    StructureColumn.NAME: self.name,
                    StructureColumn.PATH: last_path,
                    StructureColumn.METADATA: structures[-1].get(
                        StructureColumn.METADATA, {}
                    )
                    if structures
                    else {},
                }
            )
            rows.append({**row, Column.STRUCTURES: structures})

        return PipelineDataHandler(rows)
