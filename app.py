"""
OpenAI-compatible inference API for Qwen3-0.6B on CPU, served on Modal.

Architecture (mirrors Modal's official vLLM example, swapped GPU -> CPU):

  - A Modal class runs vLLM's OpenAI-compatible server (`vllm serve`) as a
    subprocess, exposed over HTTP via `@modal.web_server`. vLLM natively serves
    `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, so the
    Modal web URL *is* the OpenAI base URL (append `/v1`).
  - Modal's built-in proxy load-balances requests across containers -- that is
    the "router to the model servers". We do not hand-roll a router.
  - `@modal.concurrent` lets one container serve many requests at once via vLLM
    continuous batching; the autoscaler adds containers when `target_inputs`
    per container is exceeded, up to `max_containers`.
  - `min_containers=1` keeps one container warm so the first request never pays
    a cold start; `scaledown_window` retires excess containers during lulls.

Cold starts:
  A fresh vLLM-on-CPU container takes ~1-2 min to become ready (image pull +
  model load from the Volume + engine init). `--enforce-eager` skips vLLM's
  graph-capture/compilation warmup, which brought this down from ~5-6 min.
  Modal memory snapshots were tried to cut this further, but did not restore the
  vLLM subprocess reliably on CPU (the documented examples rely on the
  GPU-only `--enable-sleep-mode`), so we keep one container warm instead.

Usage:
  modal run app.py::download_model     # one-time: prefetch weights into the volume
  modal deploy app.py                  # deploy the autoscaling endpoint
"""

import subprocess
import time
import urllib.error
import urllib.request

import modal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
APP_NAME = "qwen3-cpu-inference"
MODEL_NAME = "Qwen/Qwen3-0.6B"
SERVED_MODEL_NAME = "Qwen3-0.6B"  # the `model` name clients pass in requests
VLLM_PORT = 8000
MINUTES = 60

# vLLM CPU build is published as a dedicated prebuilt image (no source compile).
# Pinned to a concrete version for reproducibility; bump deliberately.
# See: https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html
VLLM_CPU_IMAGE = "vllm/vllm-openai-cpu:v0.23.0-x86_64"

# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #
# The CPU image already has vLLM installed and its entrypoint is the API server;
# we clear the entrypoint so we can launch `vllm serve` ourselves under web_server.
vllm_image = (
    modal.Image.from_registry(VLLM_CPU_IMAGE)
    .entrypoint([])
    .env(
        {
            # KV cache space reserved for vLLM on CPU, in GiB. Qwen3-0.6B is
            # tiny, so a couple GiB holds plenty of concurrent sequences.
            "VLLM_CPU_KVCACHE_SPACE": "2",
            # Let vLLM bind OpenMP threads to the cores Modal gives us.
            "VLLM_CPU_OMP_THREADS_BIND": "auto",
            # Faster HF downloads.
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

app = modal.App(APP_NAME)

# Persist HuggingFace weights across cold starts so we don't re-download.
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache_vol}


# --------------------------------------------------------------------------- #
# One-time weight prefetch:  `modal run app.py::download_model`
# --------------------------------------------------------------------------- #
@app.function(image=vllm_image, volumes=VOLUMES, timeout=20 * MINUTES)
def download_model():
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME} into the hf-cache volume...")
    snapshot_download(MODEL_NAME)
    hf_cache_vol.commit()
    print("Done. Weights cached.")


# --------------------------------------------------------------------------- #
# Local helpers (run inside the container) to drive/poll the vLLM subprocess
# --------------------------------------------------------------------------- #
def _wait_ready(proc, timeout=15 * MINUTES):
    """Block until vLLM's /health returns 200, or the process dies / times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5
            ) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(3)
    raise TimeoutError("vLLM did not become healthy in time")



# --------------------------------------------------------------------------- #
# The autoscaling, OpenAI-compatible inference server
# --------------------------------------------------------------------------- #
@app.cls(
    image=vllm_image,
    volumes=VOLUMES,
    # --- CPU resources (no GPU) -------------------------------------------- #
    cpu=8,  # cores; more cores -> faster decode for this small model
    memory=16384,  # MiB
    # --- Autoscaling ------------------------------------------------------- #
    min_containers=1,  # always keep one warm; eliminates cold starts for new requests
    max_containers=8,  # cap; 40 concurrent / target_inputs(8) ~= 5 containers
    buffer_containers=0,  # only spin up containers when actually needed
    scaledown_window=60,  # seconds idle before excess containers are retired
    timeout=10 * MINUTES,
)
@modal.concurrent(
    # vLLM continuous-batches many requests per container. The autoscaler aims
    # for `target_inputs` concurrent requests per container and adds containers
    # past that; a container may burst up to `max_inputs` while new ones start.
    max_inputs=12,
    target_inputs=8,
)
class VllmServer:
    @modal.enter()
    def start(self):
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "bfloat16",  # more stable than float16 on CPU
            "--max-model-len",
            "4096",
            "--enforce-eager",  # skip graph compilation; faster startup, ~10% less throughput
        ]
        print("Launching:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)
        _wait_ready(self.proc)
        print("vLLM ready.")

    @modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
    def serve(self):
        # The actual server is the subprocess started in `start()`; this method
        # just declares the port Modal should route HTTP traffic to.
        pass

    @modal.exit()
    def stop(self):
        if getattr(self, "proc", None) is not None:
            self.proc.terminate()
