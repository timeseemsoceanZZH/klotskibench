"""
Ensure `sys.path` is ordered so the repository and external legacy `klotski_*` packages
resolve when CLI scripts in `scripts/` are run as files from the repo root.

- Prepends: repo root; optional nested `klotskibench/` and `klotskibench/src/`;
  optional `legacy/`; a sibling of the repo when `../klotski_bfs` (or related)
  exists; and paths from the environment (see end of this docstring).

- The repository provides `legacy/klotski_*.py` (frozen local engine) when present;
  for overrides or extra dist roots, set:

    export KLOTSKIBENCH_EXTRA_SYS_PATH=/path/to/roots/that/hold/klotski_bfs

  Use `os.pathsep` to add multiple roots (e.g. `dir1:dir2`). Last entry has highest
  import priority.

Idempotent: safe to call from multiple modules in the same process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_root: Path | None = None
_inited: bool = False


def _unique_prepend(p: Path) -> None:
    s = p.resolve()
    s_str = str(s)
    if s_str in sys.path:
        return
    sys.path.insert(0, s_str)


def ensure_repo_sys_path() -> Path:
    """Add search paths; return the repository root (parent of `scripts/`)."""
    global _repo_root, _inited
    if _inited and _repo_root is not None:
        return _repo_root

    here = Path(__file__).resolve().parent  # .../scripts
    repo = here.parent
    _repo_root = repo

    # 1) Repository root — `import src` (package is .../src/ under the repo)
    if repo.is_dir():
        _unique_prepend(repo)

    # 2) Nested `klotskibench/` and `klotskibench/src/` (optional monorepo layout)
    nested = repo / "klotskibench"
    if nested.is_dir():
        _unique_prepend(nested)
        nsrc = nested / "src"
        if nsrc.is_dir():
            _unique_prepend(nsrc)

    # 3) `legacy/` at repo root (optional: drop-in of legacy dist roots)
    leg = repo / "legacy"
    if leg.is_dir():
        _unique_prepend(leg)

    # 4) Sibling to repo, e.g. a checkout next to this repo: parent/klotski_bfs/
    parent = repo.parent
    has_sibling_leg = False
    for name in ("klotski_bfs", "klotski_state", "klotski_moves"):
        p = parent / name
        if p.is_file() and p.suffix == ".py":
            has_sibling_leg = True
            break
        if p.is_dir():
            has_sibling_leg = True
            break
    if has_sibling_leg:
        _unique_prepend(parent)

    # 5) User override (highest import priority: listed last, prepended last)
    extra = os.environ.get("KLOTSKIBENCH_EXTRA_SYS_PATH", "").strip()
    if extra:
        for part in reversed(extra.split(os.pathsep)):
            part = part.strip()
            if not part:
                continue
            ep = Path(part).expanduser()
            if ep.is_dir():
                _unique_prepend(ep)

    _inited = True
    return repo
