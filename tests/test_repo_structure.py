from pathlib import Path


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
        "run_training.py",
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
