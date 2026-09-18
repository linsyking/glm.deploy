# GLM-5.3 serving: agent handoff

Context recorded 2026-09-17. This directory controls a working two-node production deployment. Read `DEPLOYMENT_STATUS.md` for detailed history and evidence. Treat the facts below as a dated snapshot; inspect live configuration before making claims about current state.

## Working preferences

- Continue authorized work autonomously; avoid unnecessary confirmation. Do not restart or change serving settings merely to answer a configuration question.
- Keep task artifacts, caches, temporary files, and sockets under `/scratch_local/user_data/yiming`. Avoid `/home` and `/tmp`; do not repurpose `HOME` or `CODEX_HOME`.
- Preserve the working deployment, model weights, and audit evidence. Scope cleanup precisely and check dependencies on both nodes before deleting anything.
- Keep shared deployment scripts and policy files consistent on both nodes when changing them. The controller does not synchronize files.
- Record material changes and validation in `DEPLOYMENT_STATUS.md`; update this handoff when its facts change.

## Deployment map

| Item | Value |
|---|---|
| Head | `10.78.202.30`, `s04-p1-dgx-02-c02`, node rank 0 |
| Worker | `10.78.202.29`, `s04-p1-dgx-02-c01`, node rank 1 |
| Hardware | Four GB200 GPUs per node; TP=8 across both nodes |
| Serving directory | `/scratch_local/user_data/yiming/serving` on both nodes |
| Active environment | `/scratch_local/user_data/yiming/sglang_env_0519` on both nodes |
| Model | `/scratch_local/user_data/yiming/models/GLM-5.3` on both nodes |
| API base / model alias | `http://10.78.202.30:8000/v1` / `glm-5.3` |
| Software | SGLang 0.5.19, Torch 2.13.0, Triton 3.7.1, FlashInfer 0.6.18 |
| Toolchain | CUDA 13.3 under the user scratch directory; GCC/G++ 13 |

The user explicitly selected SGLang 0.5.19. Do not assume it remains the latest release without checking. The old SGLang 0.5.9 `sglang_env`, vLLM `serving_env`, and obsolete launch/debug code were removed from both nodes at the user's request. Rollback requires reinstalling them. Each node has an `audit/legacy-cleanup.json` manifest.

## Operations

Run the controller **on the head** from this directory:

```bash
./glm53ctl status
./glm53ctl start
./glm53ctl stop
./glm53ctl restart
./glm53ctl logs
```

`serve_glm53_sglang.sh` delegates to `serve_glm53_sglang_0519.sh`. Both ranks must run together. The controller launches detached tmux sessions, prevents duplicate/partial starts, and archives logs on subsequent starts. Sessions survive SSH disconnects, but there is no automatic reboot/crash recovery.

- Tmux socket: `/scratch_local/user_data/yiming/tmp/glm53-tmux`.
- Sessions: `glm53-head` and `glm53-worker` on their respective nodes.
- Logs: local `sglang_head.log`; worker `sglang_worker.log`; archives in `logs/`.
- Healthy state: healthy API and four scheduler ranks on each node.
- Read runtime state with `curl --noproxy '*' --max-time 10 http://10.78.202.30:8000/get_server_info`; health at `/health`; Prometheus metrics at `/metrics` (enabled 2026-09-17 via `--enable-metrics`).
- Port 8000 currently uses an unauthenticated LAN binding. Do not broaden exposure as a routine maintenance step. `/metrics` shares this binding (also forwarded through the frp tunnel).

## Performance configuration

- Native FP8 model weights and FP8 E4M3 KV cache.
- Speculative decoding enabled with `--speculative-algorithm NEXTN`; runtime resolves to EAGLE using GLM's own NEXTN/MTP draft weights. Observed defaults: 3 speculative steps, top-k 1, 4 draft tokens; adaptive speculation off.
- CUDA graphs, overlap scheduling, and radix/prefix caching enabled. `disable_chunked_prefix_cache=true` means not every prefix-cache variant is enabled.
- Observed defaults: chunked prefill 16,384, maximum running requests 48, static memory fraction 0.85.
- DSA attention with TRT-LLM prefill/decode; **CUTLASS for both target and draft MoE**. TRT-LLM MoE compilation previously crashed GCC; do not confuse that failure with the working TRT-LLM attention backend.
- NCCL NVLS and symmetric memory enabled. Startup logs previously confirmed MNNVL communication support. Inspect startup logs before inferring automatic AllReduce fusion from a top-level boolean alone.
- FlashInfer autotuning disabled in the launcher.
- A runtime snapshot reported average speculative acceptance length about 2.69 tokens per verification iteration. This is not a measured 2.69x speedup.

Prior small benchmark: 1,024 input tokens, forced 256 output tokens, low effort, temperature 0, seed 42, one warmup. SGLang 0.5.19 measured 84.8 ms mean TTFT and 184.2 decode tokens/s at concurrency 1 (4 requests); concurrency 8 measured 1.93 s mean TTFT and 356 aggregate output tokens/s (16 requests). Cache/JIT state was not fully controlled. These are whole-stack samples, not guarantees or an isolated speculation-on/off comparison. Evidence: `audit/sglang-*-benchmark-c*.jsonl`.

## API and model behavior

- Tested APIs: OpenAI-compatible `/v1/chat/completions`, `/v1/responses`, and Anthropic-compatible `/v1/messages`, including streaming and tool calls.
- Reasoning parser is `glm45`; tool parser is `glm47`. Earlier incorrect parsing mixed reasoning into answer text and exposed a literal `</think>`.
- Codex CLI 0.154.0 passed a shell-read tool round trip; Claude Code 2.1.274 passed a Read-tool round trip. These are smoke tests, not comprehensive coding evaluations. The user prefers Claude Code for real-agent testing.
- **Text-only model.** The native API rejects images. Currently `SGLANG_TEXT_ONLY=1` enables local `text_only.py` middleware on both nodes: image/document blocks become explicit text-only placeholders, and Read tool image-capability claims are rewritten. This prevents repeated image-history errors but does not let the model see images. Do not imply that removed images were analyzed. With the middleware disabled, remove images/reset affected histories or use a vision-capable adapter/model.
- Text-only middleware previously froze the API because its ASGI receive wrapper replayed the body forever. The fix delivers it once and then awaits the original receive function; preserve this behavior for streaming disconnect listeners. Regression: `python3 -m unittest test_text_only.py`; live image/stream/health probes: `python3 verify_text_only.py`.
- Server/model context limit: **1,048,576 tokens**; full-context capacity and accuracy have not been benchmarked. Claude Code may use fallback 200k metadata for the unknown model name.
- Effective reasoning levels: `low` -> low; `high` -> high; `max`, `xhigh`, `medium`, or omitted -> max. The checkpoint template recognizes only low/high specially; medium is not a distinct level. Anthropic effort mapping explicitly converts xhigh to max.
- Chat uses `reasoning_effort`; Responses uses `reasoning.effort`. The model card recommends `chat_template_kwargs.clear_thinking=true`. Do not assume `enable_thinking=false` disables reasoning for this checkpoint.
- Strict structured output passed small schema probes for Chat `response_format.json_schema` and Responses `text.format`; do not claim all JSON Schema features are covered.
- Mid-conversation system messages passed probes on all three APIs. Inline system roles in Messages are a SGLang extension, not standard Anthropic client behavior.
- Web search (2026-09-17): Claude Code's WebSearch works through the local `web_search.py` middleware (enabled by `SGLANG_WEB_SEARCH=1` plus `TAVILY_API_KEY` in `output_policy.env`). The harness executes WebSearch via a `/v1/messages` sub-request declaring Anthropic's date-versioned server tool (`web_search_20250305`); SGLang's Anthropic adapter skips such tools (empty results). The middleware accepts any `web_search_<YYYYMMDD>` type, runs the query through Tavily, and injects results into the prompt. SGLang's own native search (Responses API only, Exa-backed, `EXA_API_KEY`) remains unconfigured. On Tavily failure requests pass through unchanged.
- FP8 KV cache logs report missing scaling factors, defaulting to 1.0. Functional tests do not establish long-context accuracy or numerical equivalence across kernels.
- Cache/token observability (2026-09-17): `/metrics` exposes `sglang:cache_hit_rate`, cumulative `sglang:cached_tokens_total` / `sglang:prompt_tokens_total` / `sglang:generation_tokens_total`, latency histograms, and more; Chat `usage.prompt_tokens_details.cached_tokens` reports per-request prefix reuse. Radix matching is page-granular (`page_size` 64): prompts shorter than one page and sub-page tails report zero cached tokens, so `prompt_tokens_details: null` on short prompts is expected. `cache_stats.py` parses prefill batches from the current `sglang_head.log` for aggregate hit-rate history. Baseline before enabling flags: 90.5% over one hour of agent-session traffic.

## Thinking visibility: preserve the user's latest choice

**`output_policy.env` sets `SGLANG_HIDE_THINKING=0` on both nodes. Thinking is visible.** The user initially requested hiding, then explicitly requested setting the flag to 0 and restarting. Do not re-enable hiding without a new request.

An optional local output-only filter remains installed:

- `hide_thinking.py` provides ASGI middleware. When enabled, it removes Chat `reasoning_content`, Responses reasoning items/events (including retrieved responses), and Anthropic thinking/redacted-thinking blocks from JSON/SSE; stream indexes are remapped.
- It preserves answer text, tool arguments, usage, requests, generation settings, and server-side stored Responses. It does not disable reasoning computation, save reasoning tokens, or move reasoning into final-answer content. Raw `/generate` is outside its scope.
- Keep `separate_reasoning=true`; mixing reasoning into answer content defeats structured-field filtering.
- Upstream 0.5.19 accepts `thinking.display=omitted` but does not implement hiding.
- Registration of all four local middlewares is patched into `sglang_env_0519/lib/python3.12/site-packages/sglang/srt/entrypoints/http_server.py` on both nodes, immediately after `app.router.route_class = ORJSONRoute`: condition on `get_bool_env_var("SGLANG_HIDE_THINKING")`, import `HideThinkingMiddleware` from `hide_thinking`, and call `app.add_middleware(HideThinkingMiddleware)`; likewise `SGLANG_TEXT_ONLY`/`text_only`, `SGLANG_WEB_SEARCH`/`web_search`, and `SGLANG_STRIP_ATTRIBUTION`/`attribution`.
- Attribution strip: Claude Code's `x-anthropic-billing-header` system block is removed server-side, keeping the prompt prefix cache-stable regardless of client version (measured 2026-09-18: the block is session-constant in Claude Code 2.1.274 and the cumulative prefix-cache hit rate was 98.7%, so the strip is insurance, not a current fix).
- Package reinstalls can overwrite registration. The launcher sources the policy file and exports the serving directory through `PYTHONPATH`.
- To change visibility when requested: update/sync `output_policy.env` on both nodes, restart through the controller, and validate the requested policy.
- Hiding preserves the current generation, but clients cannot replay hidden reasoning in manually managed history. Later-turn behavior can therefore differ; do not promise complete conversation equivalence. Server-side Responses storage retains reasoning.

## Validation and evidence

Run relevant checks for the change, from this directory. Live probes generate requests against the production service; use them deliberately, and avoid unnecessary repeat benchmarking.

```bash
python3 validate_sglang_0519.py
python3 check_capabilities.py
python3 -m unittest test_hide_thinking.py test_text_only.py test_web_search.py test_attribution.py
python3 verify_output_policy.py visible
# Use `hidden` instead only when the hiding policy is enabled.
python3 verify_text_only.py
# Web search (needs SGLANG_WEB_SEARCH=1): claude --allowedTools WebSearch -p "Search the web for the latest sglang release and tell me."
```

`validate_glm53.py` is an additional general endpoint smoke suite. The 0.5.19 suite checks generated reasoning usage without requiring visible reasoning, so it supports either policy.

Evidence lives under `audit/`: protocol validation, reasoning-effort checks, capabilities, output-policy visible/hidden captures, Claude Code/Codex round trips, benchmarks, checkpoint validation, and a Prometheus metrics snapshot (`audit/metrics-sample-20260917.txt`). All 141 indexed checkpoint shards were structurally checked on both nodes; sizes matched (755,632,050,320 bytes total). This was not full cryptographic payload verification. Existing `audit/server_info.json` and other captures are historical, not authoritative live state.
