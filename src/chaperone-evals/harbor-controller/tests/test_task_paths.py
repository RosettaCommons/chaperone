"""Task, scorer and checkout resolution for the split public/private layout."""

from __future__ import annotations

import pytest

import task_paths
from task_paths import (
    PRIVATE_TASKS_ENV,
    check_inputs_match,
    default_chaperone_repo,
    task_scorer,
)


@pytest.fixture
def private_tree(tmp_path, monkeypatch):
    """A private evals_task/ tree for a task that also exists publicly."""
    public = tmp_path / "public"
    private = tmp_path / "private"
    for root in (public, private):
        (root / "demo" / "inputs").mkdir(parents=True)
        (root / "demo" / "inputs" / "target.pdb").write_text("ATOM\n")
    (private / "demo" / "score").mkdir()
    (private / "demo" / "score" / "score.py").write_text("print('held out')\n")
    monkeypatch.setattr(task_paths, "PUBLIC_TASKS_ROOT", public)
    monkeypatch.setenv(PRIVATE_TASKS_ENV, str(private))
    return public, private


def test_scorer_resolves_from_the_private_tree(private_tree):
    _, private = private_tree

    assert task_scorer("demo") == private.resolve() / "demo" / "score" / "score.py"


def test_missing_scorers_explain_how_to_set_the_private_root(monkeypatch, tmp_path):
    (tmp_path / "demo" / "inputs").mkdir(parents=True)
    monkeypatch.setattr(task_paths, "PUBLIC_TASKS_ROOT", tmp_path)
    monkeypatch.delenv(PRIVATE_TASKS_ENV, raising=False)

    with pytest.raises(SystemExit, match=PRIVATE_TASKS_ENV):
        task_scorer("demo")


def test_scorer_never_defaults_to_an_in_repository_answer(monkeypatch, tmp_path):
    (tmp_path / "demo" / "score").mkdir(parents=True)
    (tmp_path / "demo" / "score" / "score.py").write_text("")
    monkeypatch.setattr(task_paths, "PUBLIC_TASKS_ROOT", tmp_path)
    monkeypatch.delenv(PRIVATE_TASKS_ENV, raising=False)

    with pytest.raises(SystemExit, match=PRIVATE_TASKS_ENV):
        task_scorer("demo")


def test_shipped_tasks_have_no_held_out_answers():
    shipped = sorted(
        path for path in task_paths.PUBLIC_TASKS_ROOT.iterdir() if path.is_dir()
    )

    assert shipped
    for task in shipped:
        assert not (task / "score").exists(), task.name
        assert (task / "inputs").is_dir(), task.name
        assert (task / "prompt.md").is_file(), task.name
        assert not (task / "example").exists(), task.name
    assert not (task_paths.PUBLIC_TASKS_ROOT / "audit_hints.json").exists()


def test_explicit_root_wins_over_the_environment(private_tree, tmp_path):
    other = tmp_path / "other"
    (other / "demo" / "score").mkdir(parents=True)
    (other / "demo" / "score" / "score.py").write_text("")

    assert task_scorer("demo", other) == other.resolve() / "demo" / "score" / "score.py"


@pytest.mark.parametrize("name", ["../demo", "demo/../demo", "", ".", ".."])
def test_task_names_cannot_leave_the_tasks_root(private_tree, name):
    with pytest.raises(SystemExit, match="No scorer"):
        task_scorer(name)


def test_unknown_task_lists_the_available_ones(private_tree):
    with pytest.raises(SystemExit, match="Available: demo"):
        task_scorer("missing")


def test_matching_inputs_pass(private_tree):
    check_inputs_match("demo")


def test_changed_input_is_reported(private_tree):
    _, private = private_tree
    (private / "demo" / "inputs" / "target.pdb").write_text("ATOM changed\n")

    with pytest.raises(SystemExit, match="differ.*target.pdb"):
        check_inputs_match("demo")


def test_extra_private_input_is_reported(private_tree):
    _, private = private_tree
    (private / "demo" / "inputs" / "reference.pdb").write_text("ATOM\n")

    with pytest.raises(SystemExit, match="reference.pdb"):
        check_inputs_match("demo")


def test_default_checkout_is_the_repository_containing_the_controller(monkeypatch):
    monkeypatch.delenv("CHAPERONE_REPO", raising=False)

    repo = default_chaperone_repo()

    assert (repo / "src" / "chaperone-evals" / "harbor-controller").is_dir()
    assert (repo / "pyproject.toml").is_file()


def test_chaperone_repo_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAPERONE_REPO", str(tmp_path))

    assert default_chaperone_repo() == tmp_path


def test_outer_job_does_not_receive_the_private_tree_location():
    from run_eval import job_environment

    environment = {
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "key",
        PRIVATE_TASKS_ENV: "/somewhere/private/evals_task",
    }

    assert job_environment(environment) == {"PATH": "/usr/bin", "OPENAI_API_KEY": "key"}
    assert PRIVATE_TASKS_ENV in environment  # the caller's copy is untouched


def test_private_tree_inside_a_bound_path_is_refused(tmp_path):
    from task_paths import assert_private_tree_unreachable

    bound = tmp_path / "shared"
    private = bound / "evals" / "evals_task"
    private.mkdir(parents=True)

    with pytest.raises(SystemExit, match="mounted into the agent container"):
        assert_private_tree_unreachable(private, [bound])
    with pytest.raises(SystemExit, match="mounted into the agent container"):
        assert_private_tree_unreachable(private, [private])


def test_private_tree_outside_every_bound_path_is_accepted(tmp_path):
    from task_paths import assert_private_tree_unreachable

    (tmp_path / "shared").mkdir()
    private = tmp_path / "private" / "evals_task"
    private.mkdir(parents=True)

    assert_private_tree_unreachable(private, [tmp_path / "shared", "/does/not/exist"])


def test_a_symlinked_alias_of_a_bound_path_is_still_refused(tmp_path):
    from task_paths import assert_private_tree_unreachable

    real = tmp_path / "mnt" / "net"
    private = real / "evals_task"
    private.mkdir(parents=True)
    alias = tmp_path / "net"
    alias.symlink_to(real)

    with pytest.raises(SystemExit):
        assert_private_tree_unreachable(private, [alias])


def test_evaluation_dependency_matches_release_distribution():
    import tomllib
    root = task_paths.EVALS_ROOT.parent.parent
    main = tomllib.loads((root / 'pyproject.toml').read_text())
    evaluation = tomllib.loads((task_paths.EVALS_ROOT / 'pyproject.toml').read_text())
    name = main['project']['name']
    assert name in evaluation['project']['dependencies']
    assert 'sources' not in evaluation.get('tool', {}).get('uv', {})
    assert 'chaperone' not in evaluation['project']['dependencies']


def test_private_file_bind_is_rejected_even_without_parent_mount(tmp_path):
    private = tmp_path / 'private'
    private.mkdir()
    answer = private / 'answer.txt'
    answer.write_text('held out')
    with pytest.raises(SystemExit):
        task_paths.assert_private_tree_unreachable(private, [answer])
