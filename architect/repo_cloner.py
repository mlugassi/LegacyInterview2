import os
import shutil
import stat
import time
import git

from architect.state import ArchitectState

WORKSPACES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workspaces")


def _force_remove(func, path, _exc_info):
    """Error handler for shutil.rmtree: clear read-only flag, wait, then retry."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    time.sleep(0.1)
    try:
        func(path)
    except PermissionError:
        # Last resort: wait longer and try once more
        time.sleep(1.0)
        os.chmod(path, stat.S_IWRITE)
        func(path)


def _rmtree_with_retry(path: str, retries: int = 3, delay: float = 1.5) -> None:
    """Remove a directory tree, retrying on Windows file-lock errors."""
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onexc=_force_remove)
            return
        except PermissionError as exc:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Cannot delete '{path}' after {retries} attempts — "
                    "another process (OneDrive, antivirus, or a previous server) "
                    "is holding a file open. Close it and retry."
                ) from exc
            print(f"[cloner] File locked, waiting {delay}s before retry ({attempt + 1}/{retries})…")
            time.sleep(delay)


def clone_repo(state: ArchitectState) -> ArchitectState:
    url = state["github_url"].rstrip("/")
    repo_name = url.split("/")[-1].removesuffix(".git")
    dest = os.path.join(WORKSPACES_DIR, repo_name)

    os.makedirs(WORKSPACES_DIR, exist_ok=True)

    if os.path.exists(dest):
        print(f"[cloner] Removing existing clone at {dest}")
        _rmtree_with_retry(dest)

    print(f"[cloner] Cloning {url} → {dest}")
    git.Repo.clone_from(url, dest)
    print(f"[cloner] Done.")

    state["clone_path"] = dest
    return state
