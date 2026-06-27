"""
Generate the two required graphs from a load-test run.

  Graph 1 (containers_vs_traffic.png):
      container count (from results/stats.csv, produced by poll_stats.py) vs.
      concurrent users (from Locust's full-history CSV) over time.

  Graph 2 (latency.png):
      TTFT (p50/p99) and E2E (p99) over time, with concurrent users overlaid,
      from Locust's full-history CSV (results/locust_stats_history.csv).
      Latencies in that file are in ms.

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
    """Return (unix_ts_list, user_count_list, first_user_ts).

    first_user_ts is the Unix timestamp when Locust spawned its first user —
    used as the common t=0 so the containers line doesn't appear to lead traffic.
    """
    path = os.path.join(RESULTS_DIR, "locust_stats_history.csv")
    if not os.path.exists(path):
        return None, None, None
    ts_list, users_list = [], []
    first_user_ts = None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("Name") != "Aggregated":
                continue
            try:
                ts = float(row["Timestamp"])
                users = int(row["User Count"])
            except (KeyError, ValueError):
                continue
            if first_user_ts is None and users > 0:
                first_user_ts = ts
            ts_list.append(ts)
            users_list.append(users)
    return ts_list, users_list, first_user_ts


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

    user_unix, user_count, first_user_ts = _load_locust_user_count()

    # Align both series so t=0 is when Locust spawned its first user.
    # This prevents poll_stats (which starts before Locust) making the
    # containers line appear to lead traffic.
    t0 = first_user_ts if first_user_ts else (
        min(ts for ts in stats_unix if ts is not None) if stats_unix else 0
    )

    if any(ts is not None for ts in stats_unix):
        t = [(ts - t0) for ts in stats_unix]
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
    """Return ({name: {"t":[], "p50":[], "p99":[]}}, users_t, users_vals).

    Latency series and the concurrent-user series share t=0 = first_user_ts
    (same reference as graph 1). User counts come from the Aggregated rows.
    """
    path = os.path.join(RESULTS_DIR, "locust_stats_history.csv")
    if not os.path.exists(path):
        print(f"skip graph 2: {path} not found (run locust with --csv)")
        return None, None, None

    # Use first non-zero user timestamp as t=0 (consistent with graph 1).
    t0 = None
    users_ts, users_vals = [], []
    series = {}

    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "")
            try:
                ts = float(row["Timestamp"])
            except (KeyError, ValueError):
                continue

            if name == "Aggregated":
                try:
                    users = int(row["User Count"])
                except (KeyError, ValueError):
                    continue
                if t0 is None and users > 0:  # first moment a user existed
                    t0 = ts
                users_ts.append(ts)
                users_vals.append(users)

            # Latency from TTFT / E2E rows.
            if name in ("TTFT", "E2E"):
                try:
                    p50 = float(row["50%"])
                    p99 = float(row["99%"])
                except (KeyError, ValueError):
                    continue
                s = series.setdefault(name, {"ts": [], "p50": [], "p99": []})
                s["ts"].append(ts)
                s["p50"].append(p50)
                s["p99"].append(p99)

    if not series:
        print("skip graph 2: no TTFT/E2E rows in Locust history")
        return None, None, None

    if t0 is None:
        t0 = min(s["ts"][0] for s in series.values())

    # Normalize: timestamps -> relative seconds, latency ms -> s.
    for s in series.values():
        s["t"] = [ts - t0 for ts in s["ts"]]
        s["p50"] = [v / 1000.0 for v in s["p50"]]
        s["p99"] = [v / 1000.0 for v in s["p99"]]

    users_t = [ts - t0 for ts in users_ts]
    return series, users_t, users_vals


def plot_latency():
    series, users_t, users_vals = _load_locust_history()
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

    if users_t:
        ax2 = ax.twinx()
        ax2.step(users_t, users_vals, where="post", color="tab:orange",
                 linewidth=1.5, alpha=0.8, label="concurrent users (Locust)")
        ax2.set_ylabel("concurrent users", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(bottom=0)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right")
    else:
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
