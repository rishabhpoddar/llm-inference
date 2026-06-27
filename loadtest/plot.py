"""
Generate the two required graphs from a load-test run.

  Graph 1 (containers_vs_traffic.png):
      container count vs. in-flight traffic over time, from results/stats.csv
      (produced by poll_stats.py).

  Graph 2 (latency.png):
      TTFT (p50/p99) and E2E (p99) over time, from Locust's full-history CSV
      (results/locust_stats_history.csv). Latencies in that file are in ms.

Usage:
  python loadtest/plot.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# --------------------------------------------------------------------------- #
# Graph 1: container count vs. traffic
# --------------------------------------------------------------------------- #
def _load_locust_user_count():
    """Return (unix_ts_list, user_count_list) — raw Unix timestamps, not normalized."""
    path = os.path.join(RESULTS_DIR, "locust_stats_history.csv")
    if not os.path.exists(path):
        return None, None
    ts_list, users_list = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("Name") != "Aggregated":
                continue
            try:
                ts_list.append(float(row["Timestamp"]))
                users_list.append(int(row["User Count"]))
            except (KeyError, ValueError):
                continue
    return ts_list, users_list


def plot_containers_vs_traffic():
    stats_path = os.path.join(RESULTS_DIR, "stats.csv")
    if not os.path.exists(stats_path):
        print(f"skip graph 1: {stats_path} not found (run poll_stats.py)")
        return

    stats_unix, runners = [], []
    with open(stats_path) as f:
        for row in csv.DictReader(f):
            ts = row.get("ts_unix") or None
            stats_unix.append(float(ts) if ts else None)
            runners.append(float(row["runners"]))

    user_unix, user_count = _load_locust_user_count()

    # Align both series on Unix time -> common t=0.
    all_unix = [ts for ts in stats_unix if ts is not None] + (user_unix or [])
    t0 = min(all_unix) if all_unix else 0

    if any(ts is not None for ts in stats_unix):
        t = [((ts or 0) - t0) for ts in stats_unix]
    else:
        # Fallback: stats.csv has no ts_unix column (old run), use relative t.
        with open(stats_path) as f:
            t = [float(row["t"]) for row in csv.DictReader(f)]

    user_t = [ts - t0 for ts in user_unix] if user_unix else None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.step(t, runners, where="post", color="tab:blue", linewidth=2,
             label="containers (runners)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("container count", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    if user_t:
        ax2.step(user_t, user_count, where="post", color="tab:orange",
                 linewidth=1.5, alpha=0.8, label="concurrent users (Locust)")
    ax2.set_ylabel("concurrent users", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(bottom=0)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right")
    plt.title("Container count vs. traffic over time")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "containers_vs_traffic.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Graph 2: latency (TTFT + p99)
# --------------------------------------------------------------------------- #
def _load_locust_history():
    """Return {name: {"t":[], "p50":[], "p99":[]}} from Locust history CSV (ms)."""
    path = os.path.join(RESULTS_DIR, "locust_stats_history.csv")
    if not os.path.exists(path):
        print(f"skip graph 2: {path} not found (run locust with --csv)")
        return None
    series = {}
    t0 = None
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "")
            if name not in ("TTFT", "E2E"):
                continue
            try:
                ts = float(row["Timestamp"])
                p50 = float(row["50%"])
                p99 = float(row["99%"])
            except (KeyError, ValueError):
                continue
            t0 = ts if t0 is None else min(t0, ts)
            s = series.setdefault(name, {"ts": [], "p50": [], "p99": []})
            s["ts"].append(ts)
            s["p50"].append(p50)
            s["p99"].append(p99)
    if not series:
        print("skip graph 2: no TTFT/E2E rows in Locust history")
        return None
    # normalize timestamps to relative seconds, ms -> s
    for s in series.values():
        s["t"] = [ts - t0 for ts in s["ts"]]
        s["p50"] = [v / 1000.0 for v in s["p50"]]
        s["p99"] = [v / 1000.0 for v in s["p99"]]
    return series


def plot_latency():
    series = _load_locust_history()
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if "TTFT" in series:
        s = series["TTFT"]
        ax.plot(s["t"], s["p50"], color="tab:green", label="TTFT p50")
        ax.plot(s["t"], s["p99"], color="tab:red", label="TTFT p99")
    if "E2E" in series:
        s = series["E2E"]
        ax.plot(s["t"], s["p99"], color="tab:purple", linestyle="--",
                label="E2E p99")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("latency (s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    plt.title("Latency under load: TTFT (p50/p99) and E2E p99")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "latency.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_containers_vs_traffic()
    plot_latency()
