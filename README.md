# GLM-5.3 on two GB200 nodes with SGLang

This repository records a working two-node GLM-5.3 deployment built around SGLang 0.5.19. It contains the launch and control scripts, API middleware, regression tests, validation tools, and an operational write-up. It intentionally contains no model weights, Hugging Face cache, Python environment, runtime logs, raw request captures, generated API responses, or credentials.

The validated deployment uses eight GB200 GPUs across two nodes with tensor parallelism 8, native FP8 weights, FP8 E4M3 KV cache, NEXTN speculative decoding, CUDA graphs, overlap scheduling, prefix caching, chunked prefill, NCCL NVLS, symmetric memory, TRT-LLM attention, and CUTLASS MoE runners. See [DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md) for measured behavior, limitations, and the history behind these choices.

## Repository contents

- `serve_glm53_sglang_0519.sh`: production launch command and runtime environment.
- `serve_glm53_sglang.sh`: stable entrypoint for the selected SGLang version.
- `glm53ctl`: two-node start, stop, restart, status, and log control.
- `hide_thinking.py`: optional response-only reasoning redaction.
- `text_only.py`: input guard for clients that try to send images to this text-only checkpoint.
- `web_search.py`: optional Tavily execution for Anthropic-style server-side web-search requests.
- `attribution.py`: removes Claude Code's billing-attribution system block to keep prompt prefixes stable.
- `test_*.py`: offline middleware regression tests.
- `validate_*.py`, `verify_*.py`, and `check_capabilities.py`: live API probes.
- `cache_stats.py`: scheduler-log prefix-cache statistics.
- `patches/sglang-0.5.19-http-server-middlewares.patch`: the small SGLang integration patch used by the running environment.
- `AGENTS.md`: detailed handoff for agents maintaining the original deployment.

SGLang itself is not vendored and is not a submodule. The deployed package is the PyPI wheel `sglang==0.5.19`; its installed metadata does not expose a source Git commit. The repository records the local integration as a patch instead of copying the SGLang source tree.

## Configure and run

The checked-in scripts contain the paths, hosts, interfaces, and CUDA/NCCL settings from the validated cluster. Review them before using this deployment elsewhere.

1. Install SGLang 0.5.19 and its GPU dependencies on both nodes, and place the same GLM-5.3 checkpoint on both nodes. Model acquisition and synchronization are intentionally outside this repository.
2. Apply `patches/sglang-0.5.19-http-server-middlewares.patch` to SGLang's `sglang/srt/entrypoints/http_server.py` in both environments.
3. Copy `output_policy.env.example` to `output_policy.env` on both nodes. Put `TAVILY_API_KEY` only in that ignored file if web search is enabled.
4. Update the node addresses, paths, network interfaces, and environment locations in `serve_glm53_sglang_0519.sh` and `glm53ctl` if they differ from the original cluster.
5. Start and inspect the service from the head node:

   ```bash
   ./glm53ctl start
   ./glm53ctl status
   ```

The API is served at port 8000 under `/v1`. The original deployment exposes OpenAI-compatible Chat Completions and Responses endpoints plus an Anthropic-compatible Messages endpoint.

## Validate

Offline middleware tests do not require a running model:

```bash
python3 -m unittest test_hide_thinking.py test_text_only.py test_web_search.py
```

With the service running, use the relevant live checks:

```bash
python3 validate_sglang_0519.py
python3 check_capabilities.py
python3 verify_output_policy.py visible
python3 verify_text_only.py
```

These live scripts target the original cluster address by default. Update their `BASE`/`base` value before running against another deployment.

Middleware modules are imported directly when the server starts. There is no hot reload; changes to them require `./glm53ctl restart`, which reloads the model and clears the prefix cache.

## Secrets and generated data

`output_policy.env`, proxy configuration, logs, audit payloads, wire captures, caches, model formats, and common checkpoint directories are ignored. Keep API keys and tunnel credentials out of tracked files. Raw captures can contain system prompts, request headers, device/session identifiers, and conversation data even when they contain no model weights.
