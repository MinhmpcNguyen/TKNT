from __future__ import annotations

import shutil
from pathlib import Path


def configure_fresh_cache_env(env: dict[str, str], run_dir: Path) -> None:
    cache_dir = run_dir / ".cache"
    clear_run_cache(run_dir)

    matplotlib_dir = cache_dir / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLCONFIGDIR"] = str(matplotlib_dir)


def clear_run_cache(run_dir: Path) -> None:
    _remove_path_if_exists(run_dir / ".cache")


def _remove_path_if_exists(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        return
