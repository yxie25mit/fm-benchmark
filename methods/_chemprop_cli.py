"""Run a `chemprop` CLI subcommand inside the current process.

The val-prediction pass used to be a second `chemprop` process, paying the ~20 s
`import chemprop` cost a second time. `run_predict` reuses the CLI's own parser and
`PredictSubcommand.func`, so argument processing, featurization and model loading are
the same code the subprocess ran. Only `HpoptSubcommand` (ray + hyperopt, never used)
is left unregistered.

`start_background_import` warms the import while the training subprocess runs, so the
in-process pass costs no interpreter startup at all.
"""
import logging
import sys
import threading

_import_thread = None


def start_background_import():
    """Import chemprop.cli.predict on a daemon thread (overlaps the train subprocess)."""
    global _import_thread

    def _warm():
        try:
            import chemprop.cli.predict  # noqa: F401
        except Exception:
            pass

    if _import_thread is None:
        _import_thread = threading.Thread(target=_warm, daemon=True)
        _import_thread.start()
    return _import_thread


def run_predict(argv):
    """Execute `chemprop predict <argv>` in-process. Returns True on success."""
    if _import_thread is not None:
        _import_thread.join()
    try:
        from configargparse import ArgumentParser

        from chemprop.cli.conf import LOG_LEVELS
        from chemprop.cli.predict import PredictSubcommand
        from chemprop.cli.utils import pop_attr

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)
        parent = ArgumentParser(add_help=False)
        parent.add_argument("--logfile", "--log", nargs="?", const="default")
        parent.add_argument("-v", action="store_true")
        parent.add_argument("-q", action="count", default=0)
        PredictSubcommand.add(subparsers, [parent])

        args = parser.parse_args(["predict", *argv])
        _, v_flag, q_count, _, func = (
            pop_attr(args, attr) for attr in ["logfile", "v", "q", "mode", "func"]
        )
        verbosity = q_count * -1 if q_count else (1 if v_flag else 0)
        logging.basicConfig(
            handlers=[logging.StreamHandler(sys.stderr)],
            format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
            level=LOG_LEVELS.get(verbosity, logging.ERROR),
            datefmt="%Y-%m-%dT%H:%M:%S",
            force=True,
        )
        func(args)
        return True
    # SystemExit too: argparse errors used to kill only the subprocess, and the worker
    # still wrote metrics.json with val_metric=None. Keep that behaviour.
    except (Exception, SystemExit) as exc:
        print(f"[chemprop-inproc] predict failed: {type(exc).__name__}: {exc}")
        return False
