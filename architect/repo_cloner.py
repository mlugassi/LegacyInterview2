import os
import shutil
import stat
import git

from architect.state import ArchitectState

WORKSPACES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workspaces")


def _force_remove(func, path, _exc_info):
    """Error handler for shutil.rmtree: clear read-only flag then retry (needed on Windows for .git)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repo(state: ArchitectState) -> ArchitectState:
    url = state["github_url"].rstrip("/")
    repo_name = url.split("/")[-1].removesuffix(".git")
    dest = os.path.join(WORKSPACES_DIR, repo_name)

    os.makedirs(WORKSPACES_DIR, exist_ok=True)

    if os.path.exists(dest):
        print(f"[cloner] Removing existing clone at {dest}")
        shutil.rmtree(dest, onexc=_force_remove)

    print(f"[cloner] Cloning {url} → {dest}")
    git.Repo.clone_from(url, dest)
    print(f"[cloner] Done.")

    state["clone_path"] = dest
    return state
