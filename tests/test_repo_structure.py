from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_has_small_core_script_surface() -> None:
    script_files = {
        path.name
        for path in (REPO_ROOT / "scripts").glob("*.py")
    }
    assert script_files == {
        "run_app.py",
        "run_data_pipeline.py",
        "run_evaluation.py",
        "run_publication_export.py",
        "run_synthetic_generation.py",
        "run_training.py",
        "summarize_training_runs.py",
    }


def test_no_stale_interface_package_remains() -> None:
    assert not (REPO_ROOT / "src" / "interface").exists()


def test_no_empty_non_init_files_in_core_roots() -> None:
    allowed_empty_names = {"__init__.py"}
    roots = [
        REPO_ROOT / "app",
        REPO_ROOT / "config",
        REPO_ROOT / "docs",
        REPO_ROOT / "scripts",
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
    ]
    empty_files = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.stat().st_size == 0 and path.name not in allowed_empty_names:
                empty_files.append(path.relative_to(REPO_ROOT).as_posix())
    assert empty_files == []


def test_cli_help_works_without_importing_training_runtime_eagerly() -> None:
    python_bin = shutil.which("python3") or sys.executable
    train_help = subprocess.run(
        [python_bin, "scripts/run_training.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    eval_help = subprocess.run(
        [python_bin, "scripts/run_evaluation.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    publication_help = subprocess.run(
        [python_bin, "scripts/run_publication_export.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert train_help.returncode == 0, train_help.stderr
    assert eval_help.returncode == 0, eval_help.stderr
    assert publication_help.returncode == 0, publication_help.stderr
    assert "--category" in train_help.stdout
    assert "--categories" in train_help.stdout
    assert "--checkpoint" in eval_help.stdout
    assert "--shared-eval-run" in publication_help.stdout
