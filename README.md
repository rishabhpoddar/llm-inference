# Qwen3-0.6B inference API on Modal (CPU, autoscaling)

An OpenAI-compatible `/v1/chat/completions` API backed by **Qwen3-0.6B**, served
on **CPU** with **vLLM**, running on **Modal**. Modal autoscales the number of
containers with traffic (down to a warm minimum of 1 during lulls) and
load-balances requests across them.

```
app.py                  Modal app: image, weight prefetch, autoscaling vLLM server
loadtest/
  locustfile.py         Locust load test (streaming, measures TTFT + E2E)
  poll_stats.py         Polls Modal autoscaler stats -> results/stats.csv
  plot.py               Builds the two graphs from the CSVs
requirements.txt        Local tooling (modal, locust, matplotlib)
results/                Generated CSVs + PNGs
WRITEUP.md              Architecture + autoscaling notes + the two graphs
notes.md                Inference Engineering, Chapter 7 notes
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Authenticate Modal with *your* account.** A `~/.modal.toml` from a different
account must be replaced:

```bash
rm -f ~/.modal.toml      # remove the foreign token
modal token new          # log in / create a token for your own account
```

## 2. Prefetch model weights (one-time)

Caches Qwen3-0.6B into a Modal Volume so cold starts don't re-download it:

```bash
modal run app.py::download_model
```

## 3. Deploy the endpoint

```bash
modal deploy app.py
```

This prints the public URL, e.g.
`https://<workspace>--qwen3-cpu-inference-vllmserver-serve.modal.run`. That URL
is the OpenAI base; the endpoint is `<url>/v1/chat/completions`.

Smoke test:

```bash
curl https://<workspace>--qwen3-cpu-inference-vllmserver-serve.modal.run/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"hello"}]}'
```

Or with any OpenAI client: `base_url=<url>/v1`, `api_key="not-needed"`.

## 4. Run the load test (40 concurrent) + capture autoscaling

In one terminal, poll the autoscaler for the whole run (~8 min profile):

```bash
python loadtest/poll_stats.py --duration 540
```

In another terminal, run Locust headless (the `LoadTestShape` drives the
ramp -> sustain @ 40 -> drop-to-0 profile, so no `-u/-r` needed):

```bash
locust -f loadtest/locustfile.py \
  --host https://<workspace>--qwen3-cpu-inference-vllmserver-serve.modal.run \
  --headless --csv results/locust --csv-full-history
```

## 5. Generate the graphs

```bash
python loadtest/plot.py
# -> results/containers_vs_traffic.png
# -> results/latency.png
```

Locust also prints final aggregate percentiles (incl. p99 for `TTFT` and `E2E`)
and writes them to `results/locust_stats.csv`.

## Tuning

Autoscaling/throughput knobs live in `app.py`:

- `@modal.concurrent(max_inputs, target_inputs)` — concurrent requests per
  container; the autoscaler targets `target_inputs` and bursts to `max_inputs`.
- `min_containers` / `max_containers` / `scaledown_window` on `@app.cls`.
- `cpu` / `memory` — more cores speed up decode for this small model.
