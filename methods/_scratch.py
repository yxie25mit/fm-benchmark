"""Node-local scratch staging shared by the per-method train_one.py workers.

Training I/O against the shared NFS results/ volume — reading data every batch and
writing a checkpoint every epoch — is the throughput bottleneck at concurrency.
Workers stage transient training state on node-local disk and write only the final
artifacts (metrics.json, predictions, done.flag) back to results/.

Python 3.7 compatible: the molclr env is old.
"""
import atexit
import os
import shutil
import signal
import sys
from pathlib import Path


def _remove_stale_dirs(parent, prefix):
    """Reclaim scratch left by SIGKILLed workers, whose atexit never ran. The scratch
    parent is node-local, so the pid embedded in the name is comparable to live pids."""
    try:
        candidates = list(parent.glob(prefix + "_*"))
    except OSError:
        return
    for path in candidates:
        try:
            pid = int(path.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def _exit_on_sigterm(signum, frame):
    sys.exit(128 + signum)


def make_scratch_root(env_var, prefix):
    """Create (and schedule cleanup of) a per-process node-local scratch directory.

    env_var overrides the parent directory (default /var/tmp) for hosts where a
    faster or larger local disk is mounted elsewhere.
    """
    parent = Path(os.environ.get(env_var, "/var/tmp"))
    _remove_stale_dirs(parent, prefix)
    root = (parent / "{}_{}".format(prefix, os.getpid())).absolute()
    root.mkdir(parents=True, exist_ok=True)
    atexit.register(shutil.rmtree, str(root), ignore_errors=True)
    # A subprocess timeout SIGKILLs and cannot be trapped (the stale sweep above
    # covers that), but a plain SIGTERM should still unwind through atexit.
    try:
        signal.signal(signal.SIGTERM, _exit_on_sigterm)
    except ValueError:
        pass
    return root
