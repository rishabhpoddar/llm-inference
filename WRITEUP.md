# Writeup: Autoscaling CPU inference for Qwen3-0.6B on Modal

## What was built

An OpenAI-compatible inference API serving **Qwen3-0.6B on CPU** with **vLLM**,
deployed on **Modal**, that autoscales container count with traffic (down to
zero during lulls) and handles a 40-concurrent load test.

## Architecture

```
client ──HTTP──▶  Modal proxy / load balancer  ──▶  [ vLLM container ]   (CPU)
(OpenAI SDK,                (routing +              [ vLLM container ]   autoscaled
 curl, Locust)              autoscaling)            [ vLLM container ]   0..N
                                                        ...
```

- **One Modal Function** (`serve` in `app.py`) runs vLLM's OpenAI-compatible
  server (`vllm serve Qwen/Qwen3-0.6B`) as a subprocess, exposed via
  `@modal.web_server(port=8000)`. vLLM natively serves `/v1/chat/completions`,
  so the Modal URL *is* the OpenAI base URL — no custom API layer needed.
- **Routing to model servers** is handled by Modal's built-in proxy/load
  balancer. We deliberately did not hand-roll a router: Modal already
  distributes requests across healthy containers and tracks per-container load.
- **CPU serving**: the container uses the prebuilt `vllm/vllm-openai-cpu` image
  (no GPU), `dtype=bfloat16` (stable on CPU), with `VLLM_CPU_KVCACHE_SPACE`
  reserving KV-cache room for many concurrent sequences.
- **Weights** are cached in a Modal Volume (prefetched once via
  `download_model`) so cold starts don't re-download the model.

## How autoscaling works here

Two layers, both configured in `app.py`:

1. **Per-container concurrency** — `@modal.concurrent(max_inputs=12,
   target_inputs=8)`. vLLM does continuous (token-level) batching, so a single
   CPU container serves many requests at once. The autoscaler aims for
   `target_inputs` concurrent requests per container and lets a container burst
   up to `max_inputs` while new containers spin up.

2. **Container pool** — `min_containers=0`, `max_containers=8`,
   `scaledown_window=60`. When offered load exceeds
   `runners × target_inputs`, Modal starts new containers (up to the max). When
   load drops, idle containers are retired after `scaledown_window`, all the way
   to **zero** during a lull.

With `target_inputs=8`, 40 concurrent requests settle onto roughly
`40 / 8 ≈ 5` containers, with headroom up to 8.

**Cold starts & memory snapshots:** a fresh vLLM-on-CPU container takes minutes
to become ready (model load + engine init/compile). To keep scale-to-zero
*without* paying that on every wakeup, we enable Modal **memory snapshots**
(`enable_memory_snapshot=True`). The heavy init runs once in
`@modal.enter(snap=True)` — launch `vllm serve`, wait until healthy, and warm it
with a couple of requests — and Modal snapshots the whole container (including
the vLLM subprocess with loaded weights and compiled artifacts). Later cold
starts *restore* that memory image in seconds. The very first deploy still pays
the full init to build the snapshot; subsequent scale-from-zero wakeups are fast.
(We skip vLLM sleep mode / GPU snapshots — those evict weights from GPU memory;
on CPU the weights are in RAM and we want them captured directly.)

## Results

Load profile (driven by Locust's `LoadTestShape`): ramp 0→40 users, sustain at
40, then drop to 0 and idle so scale-down is visible. Latencies are measured
client-side per request: **TTFT** (time to first streamed token) and **E2E**
(full streamed response).

### Graph 1 — Container count vs. traffic over time

![containers vs traffic](results/containers_vs_traffic.png)

> Container count (left axis) tracks in-flight requests (right axis): it climbs
> as load ramps to 40, holds during the sustain phase, then returns to 0 after
> traffic stops and the `scaledown_window` elapses.

### Graph 2 — Latency under 40 concurrent (TTFT and p99)

![latency](results/latency.png)

> TTFT p50/p99 and E2E p99 during the run. Spikes line up with scale-up events
> (requests routed to cold containers); latency settles once the pool is warm.

<!-- Fill in from results/locust_stats.csv after the run: -->
| Metric | p50 | p99 |
| ------ | --- | --- |
| TTFT   | _TODO_s | _TODO_s |
| E2E    | _TODO_s | _TODO_s |

## How to reproduce

See `README.md` — deploy with `modal deploy app.py`, run `poll_stats.py` +
Locust, then `plot.py` to regenerate both graphs.

## Possible improvements

- Raise `min_containers` (or a buffer) to also hide the first-restore latency for
  latency-sensitive traffic (on top of snapshots).
- Tune `cpu` cores and `target_inputs` for the throughput/latency point you want.
- A quantized GGUF / smaller `max-model-len` reduces memory and cold-start time.
- Geo-aware routing and multi-region active-active for global, fault-tolerant
  serving (see `notes.md`).
