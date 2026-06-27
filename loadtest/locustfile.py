"""
Locust load test for the Modal vLLM endpoint.

Each user fires streaming chat completions against `/v1/chat/completions` and
measures two latencies, recorded as custom Locust metrics so they show up in
Locust's stats / CSV with full percentiles (incl. p99):

  - "TTFT"  : time to first streamed token
  - "E2E"   : end-to-end time until the stream completes

A `LoadTestShape` drives a traffic profile of ramp-up -> sustain @ 40 -> drop to
0, so the autoscaler's scale-up and scale-down are both exercised.

Run (headless), writing time-series + summary CSVs into ../results:

  locust -f loadtest/locustfile.py \
      --host https://<workspace>--qwen3-cpu-inference-vllmserver-serve.modal.run \
      --headless --csv results/locust --csv-full-history

The shape class sets users/spawn-rate itself, so you don't pass -u/-r.
"""

import random
import time

from locust import HttpUser, LoadTestShape, events, task

PROMPTS = [
    "Explain what a transformer is in two sentences.",
    "Write a haiku about autoscaling servers.",
    "What is the capital of France, and why is it famous?",
    "Give me three tips for writing clean Python.",
    "Summarize the idea of continuous batching for LLM inference.",
    "List five common HTTP status codes and what they mean.",
]

MODEL = "Qwen3-0.6B"
MAX_TOKENS = 128


def _record(name, seconds, exc=None):
    """Fire a custom Locust request event (response_time is in milliseconds)."""
    events.request.fire(
        request_type="STREAM",
        name=name,
        response_time=seconds * 1000.0,
        response_length=0,
        exception=exc,
        context={},
    )


class QwenUser(HttpUser):
    # No think time: each user streams back-to-back to generate sustained load.
    def on_start(self):
        self.payload_base = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "temperature": 0.7,
        }

    @task
    def chat(self):
        payload = dict(self.payload_base)
        payload["messages"] = [{"role": "user", "content": random.choice(PROMPTS)}]

        start = time.time()
        ttft = None
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True,
            name="chat (headers)",
        ) as resp:
            try:
                if resp.status_code != 200:
                    resp.failure(f"status {resp.status_code}")
                    return
                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:"):].strip()
                    if data == b"[DONE]":
                        break
                    if ttft is None:
                        ttft = time.time() - start
                        _record("TTFT", ttft)
                _record("E2E", time.time() - start)
                resp.success()
            except Exception as e:  # noqa: BLE001
                _record("E2E", time.time() - start, exc=e)
                resp.failure(str(e))


class StagesShape(LoadTestShape):
    """Ramp to 40 users, sustain, then drop load to 0 to observe scale-down.

    Each stage's `end` is a cumulative elapsed-time threshold (seconds).
    """

    stages = [
        {"end": 60, "users": 40, "spawn_rate": 1},    # ~ramp 0->40 over 40s
        {"end": 240, "users": 40, "spawn_rate": 10},  # sustain @ 40
        {"end": 480, "users": 0, "spawn_rate": 40},   # drop to 0, idle tail
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["end"]:
                return (stage["users"], stage["spawn_rate"])
        return None  # end the test
