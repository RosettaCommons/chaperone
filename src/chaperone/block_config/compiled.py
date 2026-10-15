"""
CompiledBlockFunction: Block execution split into prepare→execute→compile steps.
"""

from typing import ClassVar, List, Optional, TYPE_CHECKING
from abc import abstractmethod
from pydantic import PrivateAttr, model_validator
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import logging
from dataclasses import dataclass
from chaperone.block_config.base import BlockFunction
from chaperone.data import PipelineDataHandler
from chaperone.execution_tracking import track_execution
from chaperone.common import get_console, mkdir_p

if TYPE_CHECKING:
    from chaperone.containers import Image

logger = logging.getLogger(__name__)
console = get_console()


############################################################
# ScriptSpec — metadata for the inner script a block wraps
############################################################


@dataclass
class ScriptSpec:
    """Describes the external script/tool a CompiledBlockFunction wraps.

    Args:
        command: Inner command to run inside the container, e.g. ["rfd3", "design"]
            or ["python", "/path/to/script.py", "--flag=value"].
        image: Container Image that provides the runtime environment.
        source_url: URL to the upstream repo (for auditing/docs).
    """

    command: list[str]
    image: "Image"
    source_url: str = ""

    def to_template(self) -> list[str]:
        """Build the container command template.

        Returns a command list like:
            ["apptainer", "exec", "--nv", ..., "container.sif", "rfd3", "design"]
        """
        return self.image.build_command(*self.command)


############################################################
# Manifest — moved to its own module so lightweight processes (runner_wait)
# can import it without dragging in PipelineDataHandler + the block registry.
# Re-exported here for backward compatibility.
############################################################

from chaperone.block_config.manifest import Manifest  # noqa: E402,F401


############################################################
# Compiled blocks
############################################################


class CompiledBlockFunction(BlockFunction):
    """Block variant where execution is split into prepare -> execute -> compile steps.

    Resource requirements are defined via _resources ClassVar on each block.
    Container image is derived from _script_spec.image (preferred) or set
    explicitly via _image ClassVar. Use _get_image() to resolve.
    Command template is provided by _script_spec.to_template() or overridden
    via the command_template property.
    """

    _inline: ClassVar[bool] = False  # External command execution
    _chunked: ClassVar[bool] = True  # Always chunked
    _image: ClassVar["Image | None"] = (
        None  # Container Image (explicit, or derived from _script_spec)
    )
    _script_spec: ClassVar[ScriptSpec | None] = None  # Script metadata

    command_template: Optional[List[str]] = None
    cloud: bool = False
    _modal_job_args: object = PrivateAttr(default=None)
    _container_job_args: object = PrivateAttr(default=None)
    _generated_command_template: Optional[List[str]] = PrivateAttr(default=None)
    _container_bind_settings: Optional[dict] = PrivateAttr(default=None)

    def _wants_container_gpu(self) -> bool:
        resources = self._container_job_args or self.get_resources()
        return bool(resources and resources.wants_gpu)

    @model_validator(mode="after")
    def _resolve_command_template(self):
        if self.command_template is None:
            from chaperone.containers import container_runtime_gpu

            with container_runtime_gpu(self._wants_container_gpu()):
                self.command_template = self.get_command_template()
            self._generated_command_template = list(self.command_template)
        return self

    def _set_container_bind_settings(self, context: dict) -> None:
        """Resolve engine paths before caching configuration or serializing."""
        from chaperone.containers import container_runtime_bind_settings

        self._container_bind_settings = dict(context)
        with container_runtime_bind_settings(self._container_bind_settings):
            self._refresh_generated_command_template()

    def _refresh_generated_command_template(self) -> None:
        # Preserve explicitly supplied templates and user edits.
        generated = getattr(self, "_generated_command_template", None)
        if generated is not None and self.command_template == generated:
            from chaperone.containers import container_runtime_gpu

            with container_runtime_gpu(self._wants_container_gpu()):
                self.command_template = self.get_command_template()
            self._generated_command_template = list(self.command_template)

    def _needs_container_staging(self, src: str | Path) -> bool:
        """Whether an input lies outside the exact mounts available to a tool.

        Resolve symlinks before checking visibility: a link in the workspace
        does not make an external target visible inside a contained process.
        """
        if self.cloud:
            return True
        if self._get_image() is None:
            return False
        from contextlib import nullcontext
        from chaperone import get_settings
        from chaperone.containers import (
            _runtime_rundir_bind, _runtime_workspace_bind,
            container_runtime_bind_settings,
        )

        source = Path(src).expanduser().resolve()
        saved = getattr(self, "_container_bind_settings", None)
        with container_runtime_bind_settings(saved) if saved is not None else nullcontext():
            settings = get_settings()
            roots = (_runtime_rundir_bind(settings), _runtime_workspace_bind(settings))
        return not any(
            source.is_relative_to(Path(bind[0]).resolve())
            for bind in roots if bind is not None
        )

    def stage_file(
        self,
        src: str | Path | None,
        sandbox_dir: str | Path | None,
        subdir: str = "staged",
    ) -> Path | None:
        """Materialize CatCIF or copy an input outside the tool's exact mounts."""
        # 1. None always passes through silently.
        if src is None:
            return None

        # 2. Catcif tag: materialize the entry before any further staging logic.
        #    The inner tool never sees the ``archive.catcif:entry_id`` syntax.
        from chaperone.data.catcif import is_catcif_path, catcif_entry_filename

        if is_catcif_path(str(src)):
            import catcif_tools

            src = str(src)
            filename = catcif_entry_filename(src)
            if sandbox_dir:
                dest_dir = Path(sandbox_dir) / subdir
                mkdir_p(dest_dir)
                dest = dest_dir / filename
            else:
                import tempfile as _tempfile

                dest = Path(
                    _tempfile.NamedTemporaryFile(
                        suffix=".cif", delete=False, prefix=f"{Path(filename).stem}_"
                    ).name
                )
            if not dest.exists() or dest.stat().st_size == 0:
                dest.write_text(catcif_tools.get_structure(src))
            return dest

        if sandbox_dir and self._needs_container_staging(src):
            from chaperone.orchestration.modal_backend import stage_file

            return stage_file(Path(src).expanduser(), Path(sandbox_dir) / subdir)
        return Path(src)

    def stage_or_copy(self, src: str | Path, dst: str | Path) -> None:
        """Symlink a visible input; copy external inputs or materialize CatCIF.

        Drop-in replacement for os.symlink(src, dst). Catcif-tag sources
        (``archive.catcif:entry_id``) are materialized by writing the entry's
        CIF text directly to ``dst`` — symlinking the tag would fail since it
        is not a real filesystem path.
        """
        from chaperone.data.catcif import is_catcif_path

        if is_catcif_path(str(src)):
            import catcif_tools

            dst = Path(dst)
            if dst.exists():
                return
            mkdir_p(dst.parent)
            dst.write_text(catcif_tools.get_structure(str(src)))
            return

        src, dst = Path(src).expanduser(), Path(dst)
        needs_copy = self._needs_container_staging(src)
        if dst.is_symlink() and needs_copy:
            dst.unlink()  # replace a previously staged link with a contained copy
        if dst.exists():
            return
        if needs_copy:
            import shutil

            mkdir_p(dst.parent)
            shutil.copy2(src, dst)
        else:
            os.symlink(str(src.resolve()), str(dst))

    @classmethod
    def _get_image(cls) -> "Image | None":
        """Resolve container image: explicit _image, then _script_spec.image."""
        if cls.__dict__.get("_image") is not None:
            return cls._image
        if cls._script_spec is not None:
            return cls._script_spec.image
        return cls._image  # inherited _image from parent

    def get_command_template(self) -> List[str]:
        """Build the base container command for this block.

        Override in subclasses for dynamic templates (custom env, binds, etc.).
        Default: delegates to _script_spec.to_template().
        """
        if self._script_spec is not None:
            return self._script_spec.to_template()
        raise NotImplementedError(
            f"{self.__class__.__name__} must set _script_spec or override get_command_template"
        )

    def _setup_io_dirs(self, sandbox_dir) -> tuple[str, str, str]:
        """Create the standard ``inputs/`` + ``outputs/`` sandbox layout.

        Returns ``(sandbox_dir, inputdir, outputdir)`` with ``sandbox_dir``
        normalized to an absolute path and both subdirs created. Shared by the
        prediction oracles, which all open with the same makedirs trio.
        """
        sandbox_dir = os.path.abspath(str(sandbox_dir))
        inputdir = os.path.join(sandbox_dir, "inputs")
        outputdir = os.path.join(sandbox_dir, "outputs")
        os.makedirs(inputdir, exist_ok=True)
        os.makedirs(outputdir, exist_ok=True)
        return sandbox_dir, inputdir, outputdir

    def _guard_outputs(self, paths, where: str) -> None:
        """Raise if a block produced no output files.

        Replaces the per-oracle ``assert len(paths) > 0`` / ``if not paths:
        raise`` boilerplate with one message. Uses a real ``raise`` (not
        ``assert``) so it survives ``python -O``.
        """
        if not paths:
            raise RuntimeError(
                f"{self.name} execution completed but produced no output files "
                f"in {where}. Check the sandbox logs for errors."
            )

    @abstractmethod
    def create_command(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int,
    ) -> List[str]:
        """Prepare command to run"""

    @abstractmethod
    def compile(self, sandbox_dir) -> "PipelineDataHandler":
        """Compilation step for outputs"""

    def compress(
        self, output: "PipelineDataHandler", sandbox_dir: str
    ) -> "PipelineDataHandler":
        """Post-compile hook: consolidate outputs into a store.

        ``"none"`` is a no-op; ``"catcif"`` writes a ``.catcif`` archive.

        Per-chunk archives live alongside the step dir (not nested inside each
        ``chunk_N/``) so cleanup can safely remove sandboxes without touching
        the archives. Layout:

            <step_dir>/<name>.<chunk_N>.catcif   # per chunk
            <step_dir>/<name>.catcif             # after consolidation
        """
        if self.store == "none":
            return output
        sandbox_path = Path(sandbox_dir)
        if sandbox_path.name.startswith("chunk_"):
            store_path = sandbox_path.parent / f"{self.name}.{sandbox_path.name}.catcif"
        else:
            store_path = sandbox_path / f"{self.name}.catcif"
        return output.store(
            self.name,
            store_path,
            cleanup=self.store_cleanup,
        )

    def _resolve_command(self, command: List[str]) -> List[str]:
        """Resolve container path placeholders in command list.

        Replaces ``${CONTAINER:name}`` placeholders with actual image
        paths/tags at execution time.  Handles both standalone tokens
        (e.g. ``["${CONTAINER:chai1}"]``) and placeholders embedded in
        larger strings (e.g. compound commands joined with ``&&``).

        Resolution happens on the submitting host for local and Slurm runs.
        """
        import re

        pattern = re.compile(r"\$\{CONTAINER:(\w+)\}")

        def _replacer(match):
            container_name = match.group(1)
            from chaperone.containers import Image

            img = object.__new__(Image)
            img.name = container_name
            img._existing_path = None
            return str(img.get_image_path())

        return [pattern.sub(_replacer, str(arg)) for arg in command]

    def create_manifest(self, handler, sandbox_dir, chunk_id, write=True):
        """Create manifest for block execution."""
        sandbox_dir = Path(sandbox_dir)
        from chaperone.containers import (
            container_runtime_bind_settings, container_runtime_gpu,
            refresh_container_gpu_options,
        )

        with container_runtime_bind_settings(getattr(self, "_container_bind_settings", None)), container_runtime_gpu(
            self._wants_container_gpu()
        ):
            # Engine-managed blocks already resolved their template before
            # hashing/serialization. Preserve those submission-time settings in
            # fresh workers; direct block calls still resolve paths lazily here.
            if getattr(self, "_container_bind_settings", None) is None:
                self._refresh_generated_command_template()
            elif self.command_template == self._generated_command_template:
                self.command_template = refresh_container_gpu_options(
                    self.command_template, self._wants_container_gpu()
                )
                self._generated_command_template = list(self.command_template)
            command = self.create_command(handler, sandbox_dir, chunk_id)
        command = self._resolve_command(command)
        manifest = Manifest(
            name=self.name,
            command=command,
            cwd=str(sandbox_dir),
        )
        if write:
            manifest.write(sandbox_dir)
        return manifest

    def __call__(
        self,
        handler: PipelineDataHandler | None,
        sandbox_dir: Path | str,
        chunk_id,
        recompile=False,
        force_exec=False,
    ):
        """Execute block: prepare → execute → compile."""
        sandbox_dir = Path(sandbox_dir)

        if handler is None:
            handler = self.init_handler()

        if not recompile:
            # Generate command & manifest
            if self.cloud:
                from chaperone.orchestration.modal_backend import prepare_modal

                manifest, cloud_context = prepare_modal(self, handler, sandbox_dir, chunk_id)
            else:
                manifest = self.create_manifest(handler, sandbox_dir, chunk_id, write=True)
            step_dir = (
                sandbox_dir.parent
                if sandbox_dir.name.startswith("chunk_")
                else sandbox_dir
            )

            with track_execution(
                step_dir=step_dir,
                sandbox_dir=sandbox_dir,
                block_name=self.name,
                source="compiled_block",
                chunk_id=chunk_id,
                command=manifest.command_str or manifest.command,
                allocation=cloud_context.allocation if self.cloud else None,
            ):
                # A stale artifact must never turn a failed command into a
                # successful compile or execution-accounting record.
                if self.cloud:
                    from chaperone.orchestration.modal_backend import execute_modal

                    execute_modal(manifest, cloud_context, self._modal_job_args or self.get_resources())
                else:
                    Manifest.exec(sandbox_dir).check_returncode()

        # Compile results, optionally store, and backlink
        output: PipelineDataHandler = self.compile(str(sandbox_dir))
        output = self.compress(output, str(sandbox_dir))
        # Source blocks use an internal handler only to schedule/stage their
        # work.  Those synthetic rows are not part of the block's data output
        # and must not be backlink-merged into compiled designs.
        if not self._requires_input:
            return output

        handler = PipelineDataHandler.merge(handler, output)

        return handler

    def stage_then_exec_finisher(
        self,
        *,
        handler: PipelineDataHandler,
        block_path: Path | str,
        sandbox_dir: Path | str,
        chunk_id,
        output_path: Path | str,
        log_path: Path | str,
        step_dir: Path | str,
        input_row_ids: list[int] | None = None,
    ) -> None:
        """Phase 1 (stage) then ``os.execvp`` into the finisher. Does not return.

        Replaces the current heavy interpreter with a thin Python process
        (``runner.py wait``) that runs Phase 2 (subprocess) + Phase 3
        (compile/merge/write parquet) and writes submitit's result pickle
        itself. The 2-3 GB of imported modules + Phase 1 heap goes back to
        the OS for the duration of the subprocess wait.

        Used whenever the routing predicate in ``chunking._should_use_exec_finisher``
        decided this is a single-block compiled chunk running under
        submitit. No opt-in flag — the predicate enforces the only
        situations where execvp is safe.

        The block is NOT serialized into the state — ``block_path`` points
        at the cloudpickle already written by ``ChunkContext.cache_blocks``
        (``.pipeline/block.<name>.pkl``), so Phase 3 loads from there
        instead of us double-pickling a heavy object. With the block out of
        the state, every field is a plain str/int and the state is a single
        JSON file shared by both phases.

        Trade-offs documented here so future readers don't have to dig:

        * **Preemption** — ``os.execvp`` resets signal handlers, so
          submitit's SIGUSR2 catcher is gone after this point. A SLURM
          preemption signal hits Python's default SIGUSR2 action
          (terminate). ``SlurmFuture.wait`` sees a terminal failure state
          and the orchestration retry layer in ``execute_chunks``
          re-dispatches the chunk on a fresh JobID. Same functional
          outcome as legacy submitit requeue, just with a new JobID.
        * **Chunk log** — the per-chunk log file at
          ``{rundir}/logs/<block>/chunk_<id>/log.txt`` is captured by
          ``runner.py`` installing its own sys.stdout/sys.stderr tee at
          startup in both phases (mirrors legacy ``chunking._tee_output``).
        * **Half-success window** — if Phase 3 crashes between writing
          ``output.parquet`` and writing the result pickle, the retry sees
          ``output.parquet`` exists and skips the chunk. The window is
          microseconds (sequential writes); tracked as a follow-up.
        """
        from submitit.core import job_environment

        sandbox_dir = Path(sandbox_dir)

        # Phase 1 — stage inputs and write the manifest. This is the heavy
        # step that loads biotite/numpy/etc.
        self.create_manifest(handler, sandbox_dir, chunk_id, write=True)

        # Look up submitit's result pickle path while we're still inside the
        # parent's SLURM env (the finisher inherits these env vars but we
        # resolve the path here for clarity, and to bake it into the state
        # so a downstream env-var refactor can't shift it).
        result_pickle = job_environment.JobEnvironment().paths.result_pickle

        # Stash the input handler to disk and pass only its path through the
        # state. This keeps the supervisor process from holding the handler
        # in RAM through the long subprocess wait — Phase 3 only needs it
        # for the final merge, which it reloads lazily after Phase 2.
        input_handler_path = sandbox_dir / "_finisher_input.parquet"
        handler.write_parquet(input_handler_path)

        # One JSON state file shared by both phases. No cloudpickle: the
        # block is loaded from block_path (already on disk via cache_blocks).
        state = {
            "sandbox_dir": str(sandbox_dir),
            "step_dir": str(step_dir),
            "log_path": str(log_path),
            "chunk_id": chunk_id,
            "block_name": self.name,
            "block_path": str(block_path),
            "input_handler_path": str(input_handler_path),
            "output_path": str(output_path),
            "result_pickle": str(result_pickle),
            "terminal_path": str(Path(output_path).parent / "terminal.json"),
            "input_count": len(handler),
            "input_row_ids": input_row_ids or list(range(len(handler))),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        state_path = sandbox_dir / "_runner_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        # Launch runner.py by FILE PATH, not ``-m chaperone.block_config.runner``.
        # ``-m`` makes Python run the package __init__ files first, which would
        # eagerly load polars + the block registry — exactly the bloat we are
        # trying to avoid in the supervisor process during the long wait.
        # (Phase 3 is re-entered via ``-m`` from inside runner.py, where the
        # heavy imports are wanted anyway.)
        runner_path = str(Path(__file__).resolve().parent / "runner.py")
        logger.info(
            "stage_then_exec_finisher: execvp -> runner wait (state=%s)",
            state_path,
        )
        os.execvp(
            sys.executable,
            [
                sys.executable,
                runner_path,
                "wait",
                str(state_path),
            ],
        )
