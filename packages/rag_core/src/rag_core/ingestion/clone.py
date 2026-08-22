"""Shallow git clone into scratch space (decision #15).

The cloned tree is discarded after ingestion regardless of outcome -- nothing
downstream queries the filesystem directly, and v1 sources are always public
GitHub URLs, so the origin is always re-fetchable on demand. This module only
produces the scratch directory; the pipeline owns cleaning it up.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class CloneError(RuntimeError):
    pass


def clone_repo(url: str) -> Path:
    dest = Path(tempfile.mkdtemp(prefix="rag-ingest-"))
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneError(f"git clone failed for {url}: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneError(f"git clone timed out for {url}") from exc
    return dest
