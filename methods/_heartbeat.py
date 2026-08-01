"""Shared liveness heartbeat for method workers.

Writes <out_dir>/heartbeat.json (atomic) every `interval` seconds from a daemon
thread, so a monitor on the shared results volume can tell a running cell from a
dead one without ssh. The timestamp advances only while the process is alive, so
a died-at-launch worker leaves no fresh heartbeat. molformer instead writes an
epoch-accurate heartbeat inline (it can wedge mid-run); these methods fail by
process death, which the wall-clock thread catches.
"""
import json
import os
import threading
import time


def start_heartbeat(out_dir, interval=60):
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def _beat():
        while True:
            try:
                tmp = os.path.join(out_dir, "heartbeat.json.tmp")
                with open(tmp, "w") as fh:
                    json.dump({"ts": int(time.time()), "pid": os.getpid()}, fh)
                os.replace(tmp, os.path.join(out_dir, "heartbeat.json"))
            except Exception:
                pass
            time.sleep(interval)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return thread
