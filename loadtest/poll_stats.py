"""
Poll the deployed Modal Function's autoscaler stats and log them to CSV.

Records, every few seconds:
  - container count (runners)
  - in-flight traffic seen by Modal (running inputs + backlog)

Run this in parallel with load_test.py for the whole duration:
  python loadtest/poll_stats.py --duration 600

Then plot.py turns stats.csv into the "container count vs. traffic" graph.

Note: the Function must be DEPLOYED (`modal deploy app.py`), not just `modal serve`.
"""

import argparse
import csv
import os
import time

import modal

APP_NAME = "qwen3-cpu-inference"
CLASS_NAME = "VllmServer"
METHOD_NAME = "serve"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def read_stats(fn):
    """Return (runners, backlog, running) defensively across modal versions."""
    s = fn.get_current_stats()
    runners = getattr(s, "num_total_runners", None)
    if runners is None:
        runners = getattr(s, "runners", 0)
    backlog = getattr(s, "backlog", 0)
    running = getattr(s, "num_total_inputs", None)
    if running is None:
        running = getattr(s, "running_inputs", 0)
    return runners, backlog, running


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    fn = cls().serve

    out = os.path.join(RESULTS_DIR, "stats.csv")
    t0 = time.time()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        # ts_unix lets plot.py align this file with Locust's absolute timestamps.
        w.writerow(["t", "ts_unix", "runners", "backlog", "running", "traffic"])
        print(f"Polling {APP_NAME}/{CLASS_NAME}.{METHOD_NAME} every {args.interval}s -> {out}")
        while time.time() - t0 < args.duration:
            t = time.time() - t0
            try:
                runners, backlog, running = read_stats(fn)
            except Exception as e:  # noqa: BLE001
                print(f"[{t:6.1f}s] stats error: {e}")
                time.sleep(args.interval)
                continue
            traffic = running + backlog  # total work in the system
            w.writerow([f"{t:.1f}", f"{time.time():.1f}", runners, backlog, running, traffic])
            f.flush()
            print(f"[{t:6.1f}s] runners={runners} running={running} "
                  f"backlog={backlog} traffic={traffic}")
            time.sleep(args.interval)
    print(f"Done -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=int, default=600,
                   help="how long to poll, seconds (cover the whole load test)")
    p.add_argument("--interval", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
