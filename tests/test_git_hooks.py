from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _environment(repo: Path) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_DIR=str(repo / ".git"),
        GIT_WORK_TREE=str(repo),
        GIT_INDEX_FILE=str(repo / ".git/index"),
        PYTHONPATH=str(repo / "src"),
        HOOK_TEST_PYTHON=sys.executable,
        HOOK_TEST_LOG=str(repo.parent / "gate.log"),
    )
    return env


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=_environment(repo),
        capture_output=True,
        text=True,
        check=check,
        timeout=15,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--template=")
    _git(repo, "config", "user.name", "Hook tests")
    _git(repo, "config", "user.email", "hooks@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    shutil.copytree(ROOT / ".githooks", repo / ".githooks")
    for dependency in (".venv", "desktop/electrobun/node_modules"):
        directory = repo / dependency
        directory.mkdir(parents=True)
        (directory / "sentinel").write_text("preserve", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src/hook_sample.py").write_text("VALUE = 'staged'\n", encoding="utf-8")
    (repo / "Makefile").write_text(
        "check:\n"
        '\t@printf "%s\\n" "$$PWD" > "$$HOOK_TEST_LOG"\n'
        '\t@test -z "$${GIT_DIR:-}" && test -z "$${GIT_INDEX_FILE:-}"\n'
        "\t@test -L .venv && test -L desktop/electrobun/node_modules\n"
        "\t@test ! -e untracked.py && test ! -e intent.py\n"
        '\t@"$$HOOK_TEST_PYTHON" -c "import hook_sample; assert hook_sample.VALUE == \'staged\'"\n',
        encoding="utf-8",
    )
    _git(repo, "add", "Makefile", "src", ".githooks")
    return repo


def _install(repo: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", ".githooks/install"],
        cwd=repo,
        env=_environment(repo),
        capture_output=True,
        text=True,
        check=check,
        timeout=5,
    )


def test_commit_checks_staged_python_not_unstaged_or_intent_to_add(repository: Path):
    repo = repository
    _install(repo)
    _install(repo)  # Installation is idempotent.
    sample = repo / "src/hook_sample.py"
    sample.write_text("VALUE = 'unstaged'\n", encoding="utf-8")
    (repo / "untracked.py").write_text("untracked work", encoding="utf-8")
    (repo / "intent.py").write_text("intent-to-add work", encoding="utf-8")
    _git(repo, "add", "--intent-to-add", "intent.py")

    _git(repo, "commit", "--quiet", "-m", "test(hook): accept checked staged content")

    assert _git(repo, "show", "HEAD:src/hook_sample.py").stdout == "VALUE = 'staged'\n"
    assert sample.read_text() == "VALUE = 'unstaged'\n"
    assert (repo / "untracked.py").read_text() == "untracked work"
    assert (repo / "intent.py").read_text() == "intent-to-add work"
    assert _git(repo, "ls-tree", "--name-only", "HEAD").stdout.splitlines() == [
        ".githooks",
        "Makefile",
        "src",
    ]
    snapshot = Path((repo.parent / "gate.log").read_text().strip())
    assert snapshot != repo and not snapshot.parent.exists()
    assert (repo / ".venv/sentinel").read_text() == "preserve"
    assert (repo / "desktop/electrobun/node_modules/sentinel").read_text() == "preserve"


def test_bad_staged_code_blocks_commit_even_when_working_file_is_fixed(
    repository: Path,
):
    repo = repository
    _install(repo)
    sample = repo / "src/hook_sample.py"
    sample.write_text("VALUE = 'bad'\n", encoding="utf-8")
    _git(repo, "add", "src/hook_sample.py")
    staged_tree = _git(repo, "write-tree").stdout
    sample.write_text("VALUE = 'staged'\n", encoding="utf-8")

    result = _git(
        repo,
        "commit",
        "--quiet",
        "-m",
        "test(hook): reject unchecked staged content",
        check=False,
    )

    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert _git(repo, "rev-parse", "--verify", "HEAD", check=False).returncode != 0
    assert _git(repo, "write-tree").stdout == staged_tree
    assert sample.read_text() == "VALUE = 'staged'\n"
    snapshot = Path((repo.parent / "gate.log").read_text().strip())
    assert not snapshot.parent.exists()


@pytest.mark.parametrize("dependency", [".venv", "desktop/electrobun/node_modules"])
def test_missing_dependencies_block_commits(repository: Path, dependency: str):
    _install(repository)
    shutil.rmtree(repository / dependency)

    result = _git(
        repository,
        "commit",
        "--quiet",
        "-m",
        "test(hook): require installed check dependencies",
        check=False,
    )

    assert result.returncode != 0
    assert f"missing {dependency}" in result.stderr


def test_check_gate_must_be_in_the_staged_tree(repository: Path):
    _install(repository)
    _git(repository, "rm", "--cached", "Makefile")

    result = _git(
        repository,
        "commit",
        "--quiet",
        "-m",
        "test(hook): require staged check tooling",
        check=False,
    )

    assert result.returncode != 0
    assert "stage the Makefile" in result.stderr
    assert (repository / "Makefile").is_file()


@pytest.mark.parametrize("existing", ["configured", "default"])
def test_installation_preserves_existing_hooks(repository: Path, existing: str):
    if existing == "configured":
        _git(repository, "config", "core.hooksPath", "custom-hooks")
    else:
        hook = repository / ".git/hooks/pre-push"
        hook.parent.mkdir()
        hook.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        hook.chmod(0o755)

    result = _install(repository, check=False)

    assert result.returncode != 0
    assert "Refusing" in result.stderr
    configured = _git(repository, "config", "--get", "core.hooksPath", check=False)
    if existing == "configured":
        assert configured.stdout.strip() == "custom-hooks"
    else:
        assert configured.returncode == 1
        assert (repository / ".git/hooks/pre-push").read_text() == "#!/bin/sh\nexit 0\n"
