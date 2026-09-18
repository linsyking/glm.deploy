# GLM-5.3 dual-node deployment

Audited and corrected on 2026-09-16; legacy environments and obsolete deployment scripts removed on 2026-09-17 from both nodes. Removal manifests: `audit/legacy-cleanup.json` on each node. The active environment remains `sglang_env_0519`.

The reusable code and workflow documentation are published at `git@github.com:linsyking/glm.deploy.git`. The repository deliberately excludes the live `output_policy.env`, proxy configuration and binary, logs, raw audit outputs, wire captures, caches, model files, Hugging Face data, Python environments, and model synchronization script. SGLang is recorded as version 0.5.19 plus a minimal middleware registration patch; it is neither vendored nor included as a submodule.

## Configuration

- API: `http://10.78.202.30:8000/v1`, model name `glm-5.3`.
- Head: `10.78.202.30` (`s04-p1-dgx-02-c02`), rank 0, four GB200 GPUs.
- Worker: `10.78.202.29` (`s04-p1-dgx-02-c01`), rank 1, four GB200 GPUs.
- SGLang 0.5.19, TP=8, native FP8 weights, FP8 E4M3 KV cache; promoted following user approval.
- Torch 2.13.0, Triton 3.7.1, FlashInfer 0.6.18, NCCL 2.29.7.
- Environment: `/scratch_local/user_data/yiming/sglang_env_0519` on both nodes. Target and draft MoE runners explicitly use `cutlass` to avoid the GCC crash compiling FlashInfer TRT-LLM MoE.
- Existing CUDA 13.3 toolkit and GCC/G++ 13 retained.
- NEXTN speculative decoding resolves to EAGLE in this SGLang version, with 3 steps / 4 draft tokens by default. The old plan's 5-token speculation was not deployed.
- MNNVL/NVLS and symmetric memory retained. Prior NCCL logs confirm an eight-GPU MNNVL clique and NVLS multicast support.
- Runtime defaults: 48 running requests, 16,384-token chunked prefill, model context limit 1,048,576. Full-context capacity has not been benchmarked.
- Reasoning parser `glm45`, tool parser `glm47`.

## Audit findings and corrections

The old vLLM/Ray attempt failed: GCC crashed while compiling a kernel, followed by engine initialization and Ray cleanup errors. The later SGLang deployment successfully loaded and generated text. Its launch script was retained and corrected to add a stable model alias and reasoning/tool parsing. Before correction, an arithmetic answer contained a literal `</think>` and reasoning was mixed into content.

All 141 indexed checkpoint shards exist on each node. Their sizes match across nodes, totaling 755,632,050,320 bytes (703.73 GiB). Safetensors header offsets agree with file sizes. This is structural validation, not full cryptographic verification of the checkpoint payloads. Evidence: `audit/checkpoint_validation.json` on the head.

NCCL collective-level debug logging was disabled. Scratch-local temporary/cache paths are explicit; the tmux socket is also on scratch. Existing weights and environments were reused. No Ray cluster is needed for this deployment.

## Operating the server

The current processes run in detached tmux sessions and survive SSH disconnects. They are not configured to restart automatically after a reboot or crash.

Use the controller on the head node:

```bash
cd /scratch_local/user_data/yiming/serving
./glm53ctl status
./glm53ctl stop
./glm53ctl start
./glm53ctl restart
./glm53ctl logs
```

`start` launches both ranks, waits for the health endpoint, and refuses to create duplicate or partial sessions. `stop` shuts down both ranks and preserves the logs. Each subsequent start archives the prior logs under `serving/logs/`.

Head session:

```bash
tmux -S /scratch_local/user_data/yiming/tmp/glm53-tmux list-sessions
tail -f /scratch_local/user_data/yiming/serving/sglang_head.log
```

Worker session:

```bash
ssh 10.78.202.29 'tmux -S /scratch_local/user_data/yiming/tmp/glm53-tmux list-sessions'
ssh 10.78.202.29 'tail -f /scratch_local/user_data/yiming/serving/sglang_worker.log'
```

To start after both ranks have stopped, run on the head:

```bash
tmux -S /scratch_local/user_data/yiming/tmp/glm53-tmux new-session -d -s glm53-head 'exec bash /scratch_local/user_data/yiming/serving/serve_glm53_sglang.sh head 8000 > /scratch_local/user_data/yiming/serving/sglang_head.log 2>&1'
ssh 10.78.202.29 "tmux -S /scratch_local/user_data/yiming/tmp/glm53-tmux new-session -d -s glm53-worker 'exec bash /scratch_local/user_data/yiming/serving/serve_glm53_sglang.sh worker 8000 > /scratch_local/user_data/yiming/serving/sglang_worker.log 2>&1'"
```

Wait for health and generation warmup to complete. Both ranks must run together. Preserve logs before subsequent launches if needed.

```bash
curl --noproxy '*' -f http://10.78.202.30:8000/health
curl --noproxy '*' http://10.78.202.30:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"What is 17 times 23?"}],"reasoning_effort":"low","chat_template_kwargs":{"clear_thinking":true},"max_tokens":512}'
```

The checkpoint has three effective reasoning levels: `low`, `high`, and `max`. Omit `reasoning_effort` to select the default `max`. SGLang 0.5.19 accepts all four labels, but the checkpoint maps `medium` to `max`; it is not a distinct budget. Chat uses `reasoning_effort`; Responses uses `reasoning: {"effort": "high"}`. The model card recommends `clear_thinking=true` for chat. Do not assume `enable_thinking=false` disables reasoning for this checkpoint.

The tested endpoints are `/v1/chat/completions`, `/v1/responses`, and Anthropic-compatible `/v1/messages`. Version 0.5.19 passed reasoning separation, usage, function calling, and Responses streaming checks that failed under 0.5.9. Codex CLI 0.154.0 completed a shell-read tool round trip; Claude Code 2.1.274 completed a Read-tool round trip. These are smoke tests, not full coding evaluations. Claude Code applies fallback metadata (200k context) to the unknown model name despite the server's 1,048,576-token limit.

Run `python3 validate_sglang_0519.py` for the current protocol suite. Evidence is in `audit/sglang-0.5.19-validation.json`, `audit/sglang-0.5.19-reasoning-efforts.json`, and the Codex/Claude Code JSONL files in `audit/`.

Sample benchmark: 1,024 input tokens, 256 output tokens, low effort, temperature zero, seed 42, one warmup. Four requests at concurrency 1; sixteen at concurrency 8. These short runs compare whole software stacks, not isolated MoE kernels; cache/JIT state was not fully controlled.

| Stack | Concurrency | Mean TTFT | Aggregate output tok/s | Decode tok/s (1 / mean TPOT) |
|---|---:|---:|---:|---:|
| 0.5.19 CUTLASS | 1 | 84.8 ms | 173.7 | 184.2 |
| 0.5.9 TRT-LLM | 1 | 637.3 ms | 77.6 | 95.8 |
| 0.5.19 CUTLASS | 8 | 1.93 s | 356.0 | 70.4 |
| 0.5.9 TRT-LLM | 8 | 9.81 s | 70.8 | 13.7 |

Raw benchmark results are `audit/sglang-*-benchmark-c*.jsonl`. Kernel changes can introduce numerical differences; these smoke tests do not establish accuracy equivalence. The obsolete 0.5.9 and vLLM environments and launch scripts were deleted at the user's request; rollback would require reinstalling them.

## Capability probes (2026-09-17)

`python3 check_capabilities.py` saves live results in `audit/capabilities.json`.

- Structured output: Chat `response_format.json_schema` and Responses `text.format` passed a strict object schema with required integer/string fields, enum, and no additional properties. Both returned `{"answer":391,"unit":"items"}`. This is a small schema test, not coverage of every JSON Schema feature.
- Mid-conversation system message: Chat, Responses, and this server's Anthropic Messages adapter accepted an inline system update after a previous assistant turn and returned the new instructed marker. Inline system roles in Messages are a SGLang extension; standard Anthropic clients may only permit the top-level system field.
- Native web search: unavailable in the current GLM deployment. A Responses `web_search` declaration returned HTTP 200 but no search execution/result. Use a client-executed search function/MCP tool and feed its results back; the declaration alone does not provide search.
- Upstream 0.5.19 accepts Anthropic `thinking.display=omitted` but does not implement it. A local output-only filter is now installed; see below.

## Observability: metrics and per-request cache reporting (2026-09-17)

Added `--enable-metrics` and `--enable-cache-report` to `serve_glm53_sglang_0519.sh` on both nodes and restarted through the controller at the user's request. `get_server_info` confirms both flags true.

- `/metrics` on port 8000 now serves SGLang Prometheus metrics, including `sglang:cache_hit_rate` (scheduler-reported gauge), cumulative `sglang:cached_tokens_total`, `sglang:prompt_tokens_total`, `sglang:generation_tokens_total`, `sglang:num_requests_total`, `sglang:token_usage`, `sglang:spec_accept_length`, TTFT/inter-token/e2e latency histograms, and `sglang:uncached_prompt_tokens_histogram`. A snapshot is in `audit/metrics-sample-20260917.txt`. The endpoint inherits the unauthenticated port-8000 LAN binding, which is also forwarded through the frp tunnel.
- OpenAI Chat responses now report `usage.prompt_tokens_details.cached_tokens` when cached prefix pages were reused; `null` means nothing to report. Radix matching is page-granular (`page_size` 64): prompts shorter than one page, and the sub-page tail of any prompt, report zero cached tokens. Verified live: repeating a 464-token prompt returned `cached_tokens: 448` (seven full pages).
- Baseline measured from scheduler logs before the flags existed (`python3 cache_stats.py`, which parses TP0 prefill batches from `sglang_head.log`): 90.5% aggregate prefix-cache hit rate over the 2026-09-17 08:13–09:27 window — 603 prefilled sequences, 23,836,544 cached vs 2,516,608 computed prefill tokens. This reflects the agent-session workload's long shared prefixes, not a universal server-health norm. The script only sees the current log; restarts archive it.

Post-restart validation: `validate_sglang_0519.py` passed (Responses streaming first text delta 0.102 s), `verify_output_policy.py visible` passed all twelve probes, and `./glm53ctl status` reports a healthy API with 4/4 scheduler ranks on each node.

## Text-only middleware and API stall fix (2026-09-17)

`output_policy.env` currently enables `SGLANG_TEXT_ONLY=1` on both nodes. The local `text_only.py` middleware rewrites image-capability claims in Read tool descriptions and replaces image/document content with an explicit text-only placeholder for Chat, Responses, and Messages. It does not provide vision capability. Thinking remains visible (`SGLANG_HIDE_THINKING=0`).

The initial middleware replayed the same buffered request body on every ASGI `receive()` call without yielding. Streaming disconnect listeners can therefore loop indefinitely and starve the entire HTTP event loop. Observed symptoms were timeouts on both LAN and localhost health requests, a CPU-active HTTP process, and idle GPUs with all eight scheduler ranks alive.

Fixed replay to deliver the body once, then delegate to the original transport receive function for waiting/disconnect notifications. Synced the fix to both nodes. The regression test fails against the original implementation (`audit/text-only-old-regression.txt`) and passes against the fix; all 13 text-only/thinking-filter unit tests pass on the head, and all nine text-only tests pass on the worker. Live streaming/image probes are in `verify_text_only.py` and save evidence to `audit/text-only-live.json`.

Follow-up validation (same day) found and fixed a bug in `verify_text_only.py` itself: its assertion rejected any stream containing the substring `"error"`, but every Responses API event legitimately carries a top-level `"error":null` field, so the responses probe could never pass and the evidence file was never written. The check now looks for actual failure markers (`response.failed`, `event: error`, `"error":{`); all three API probes pass and `audit/text-only-live.json` exists. After the fix the full set passed: 13 middleware unit tests, `validate_sglang_0519.py` (Responses streaming first text delta 0.086 s), `verify_output_policy.py visible` (12/12), and a Claude Code 2.1.274 end-to-end run reading a real PNG (`audit/claude-code-image-read.jsonl`): the model stated it cannot view images, never attempted a Read call on the image, worked around it with file metadata, and finished with `subtype: success` and no API errors.

## Tavily-backed web search for Claude Code (2026-09-17)

Root cause of empty WebSearch results in Claude Code: the harness executes its WebSearch client tool by sending a follow-up `/v1/messages` request that declares Anthropic's date-versioned server tool `{"type": "web_search_20250305"}` with the query in the user message ("Perform a web search for the query: …"). SGLang's Anthropic adapter unconditionally skips Anthropic server-side tools ("no native support in the OpenAI-compatible backend"), so the search silently returned nothing — 206 skip lines accumulated in one day's head log before the fix. SGLang 0.5.19 does implement native web search on the Responses API only, Exa-backed and gated on `EXA_API_KEY`; that path never serves the Anthropic Messages protocol Claude Code uses.

`web_search.py` (third local middleware, registered in the same http_server.py patch block, gated by `SGLANG_WEB_SEARCH=1` plus `TAVILY_API_KEY` in `output_policy.env`, now mode 600 on both nodes) intercepts `/v1/messages` requests whose `tools` contain any `web_search_<YYYYMMDD>` server tool: it extracts the query, calls Tavily directly (stdlib `urllib` via `asyncio.to_thread`; direct egress confirmed working without the site proxy; 15 s timeout; 10-minute result cache), injects the results into the user message, removes the server tool and `tool_choice`, and lets the model report findings. Any request without a web_search server tool — including normal conversations that merely expose the WebSearch client tool — passes through byte-identical. On Tavily failure the request is forwarded unchanged (previous behavior).

All Anthropic web_search tool versions route to this same basic search: `web_search_20250305` (basic), `web_search_20260209` (dynamic filtering), and `web_search_20260318` (response inclusion control) differ only in server-side execution features that require Anthropic's code-execution infrastructure; the search semantics are identical, so refusing newer versions would break clients for no benefit. `allowed_domains`/`blocked_domains` are honored via Tavily's `include_domains`/`exclude_domains` (both together: allowed wins, where Anthropic 400s). `max_uses` is inapplicable (one search per sub-request; the client iterates). `response_inclusion: "excluded"` is trivially satisfied because raw search blocks are never echoed. Dynamic filtering (20260209+) is not provided. Every interception logs one line to the server log — `grep '\[web-search\]' sglang_head.log` shows the tool version, query, filters, and result count for each client (Claude Code 2.1.274 sends `web_search_20250305`; a ZCode capture is pending user-side rerun). OpenAI-protocol clients declaring the unversioned Responses-API `web_search` tool are not covered by this middleware.

Validated end-to-end: Claude Code 2.1.274 with `--allowedTools WebSearch` answered both a SGLang and a PyTorch current-version query with correct facts and markdown source links, including an iterative refine (stale first results, follow-up search). Zero server-tool skip lines after the restart (previously 206/day). Evidence: `audit/web-search-subrequest.json` (captured wire request), `audit/web-search-claude-code-e2e.txt` (PyTorch answer). Full regression after the restart: 21 unit tests across the three middlewares, `validate_sglang_0519.py`, `verify_output_policy.py visible` (12/12), `verify_text_only.py` (3/3), healthy cluster 4/4 ranks per node. The Tavily key is a dev key shared in conversation; rotate it from the Tavily dashboard if that transcript leaves this machine.



## User-input recording and middleware hot reload (2026-09-17)

**Removed 2026-09-18 at the user's request.** `user_history.py`, `hot_reload.py`, their tests, and the `history/` data (test traffic only) were deleted from both nodes; `SGLANG_RECORD_USER_INPUT` is gone from `output_policy.env`; the http_server.py registration patch returned to direct middleware imports for the remaining four middlewares. Consequence: middleware edits now require `./glm53ctl restart` again — there is no hot reload. Historical record of the removed feature:

Post-removal verification at 16:14 PDT found one stale 8,276-byte history file left on the head; it was not open and was deleted with its directory. The history directory and recording/hot-reload code are now absent on both nodes. The apparent hang around the restart was engine work rather than an ASGI receive loop: cold startup took 127.57 seconds, then several reconnecting agent sessions submitted uncached long histories at once. The queue reached three requests and 713,944 pending prefill tokens; temporary ~3.2 GB allocations also logged recoverable OOM warnings with only ~2.8 GB free per GPU. Requests continued and the backlog drained. Afterward, health was responsive, a small Chat request completed in 89 ms, and all eight scheduler ranks remained live. Removing hot reload means every future middleware edit incurs another full model restart and cold-cache refill.

`user_history.py` (fourth local middleware, enabled by `SGLANG_RECORD_USER_INPUT=1` in `output_policy.env`) appends human-authored user turns to `history/user-inputs-YYYY-MM-DD.jsonl` with timestamp, API, session, and model. It records only what the user wrote: assistant turns, tool results, harness-injected wrappers (`<system-reminder>`, `<command-*>`, `<local-command-caveat>`, `<total_tokens>`), and WebSearch execution sub-requests are excluded, and turns that are empty after stripping are skipped. Clients resend whole conversations every request, so entries are deduplicated per (session, text hash) — session from Anthropic `metadata.user_id`, falling back to a conversation fingerprint for Chat/Responses. The dedup set is rebuilt from the history files at startup, so restarts do not duplicate entries. Writes never fail requests; text is capped at 20,000 characters. Live proof: a Claude Code run with a web search produced exactly one record (the human prompt); the system prompt, tool definitions, and search sub-request were all excluded.

All four middlewares are now registered through `hot_reload.py` (`HotMiddleware`) in the http_server.py patch: each is imported lazily and re-imported whenever its file's mtime changes. Editing `text_only.py`'s placeholder and reverting it both took effect on the next request with no restart — verified live against the running server. In-flight requests finish on the old code; a module that fails to import (syntax error mid-edit) keeps the previous version and logs `[hot-reload]` to the server log. Scope: this covers the serving-directory middleware layer only. Changes to SGLang's own Python (engine, schedulers, adapters, the registration patch in `http_server.py`) still require the full restart — schedulers hold weights, CUDA graphs, and KV state in separate processes with no reload path. Tests: `python3 -m unittest test_user_history.py test_hot_reload.py`.



## Attribution-header strip and prefix-cache verification (2026-09-18)

The SGLang Anthropic-API docs recommend `CLAUDE_CODE_ATTRIBUTION_HEADER=0` because a varying `x-anthropic-billing-header:` system block would be the first differing token between turns and force full re-prefill. Measured on this deployment (Claude Code 2.1.274 wire captures): the block is byte-identical across all same-session main requests (11/11 across two runs), the build tag varies only per CLI invocation (first request of a session misses, then stable), and the per-turn `<total_tokens>` reminders sit at the end of history where they cannot invalidate the prefix. Cumulative prefix-cache hit rate at the time: 98.7% (359.3M cached of 364.1M prompt tokens). The WebSearch sub-request path uses a different build tag, but sub-requests are one-shot and do not reuse cache.

As insurance against future client versions varying the block, `attribution.py` (`SGLANG_STRIP_ATTRIBUTION=1` in `output_policy.env`; one of the four middlewares remaining after the 2026-09-18 removal of user-history recording and hot reload) removes the attribution system block server-side. This covers every client — including tunnel clients that cannot set the env var — needs no client configuration, and trims the block's ~20 tokens from each request. Verified live: with an attribution block plus a real system prompt sent to `/v1/messages`, the model's first system line is the real prompt; prefix caching still engages (576/617 tokens cached on an identical repeat). Regression after the restart: 40 middleware unit tests across all six modules then present, `validate_sglang_0519.py`, `verify_output_policy.py visible` (12/12), `verify_text_only.py` (3/3), cluster healthy.

## Hide thinking output


`output_policy.env` currently sets `SGLANG_HIDE_THINKING=0`: reasoning is visible, as requested by the user. Set it to `1` to hide reasoning, sync the file to the worker, then run `./glm53ctl restart`. Clients need no special parameters. This does not disable generation of reasoning or change sampling/prompt settings. Token usage and reasoning latency remain. With hiding enabled, streaming callers wait for the first final-answer text while reasoning is generated.

The local `hide_thinking.py` ASGI middleware removes Chat `reasoning_content`, Responses reasoning output items/events (including retrieved Responses), and Anthropic thinking blocks/events from outgoing JSON/SSE only. It preserves text, tool calls/arguments, usage, and stream indexes. Requests and SGLang's stored Responses history are untouched. Clients must keep the default `separate_reasoning=true`; disabling parsing or deliberately asking for reasoning in answer text is outside this structured-field filter's scope. Raw engine endpoints such as `/generate` are outside this client-API filter.

The registration patch is in the environment's `sglang/srt/entrypoints/http_server.py` on both nodes, immediately after `app.router.route_class = ORJSONRoute`. Package reinstalls may overwrite it; restore the conditional import of `HideThinkingMiddleware` and `app.add_middleware(HideThinkingMiddleware)` guarded by `get_bool_env_var("SGLANG_HIDE_THINKING")` before serving again. The launcher exports the policy flag and serving directory in PYTHONPATH.

Tests: `python3 -m unittest test_hide_thinking.py`; live JSON/SSE probes: `python3 verify_output_policy.py hidden`. Captures are in `audit/output-policy-visible.json` and `audit/output-policy-hidden.json`. Since hidden reasoning is not returned to clients, they cannot replay those hidden blocks in manually managed conversation history; the server-side Responses store retains them. This filter preserves the current generation, but cannot promise identical future generations if client-supplied history changes.

## Validation and limits

Run `python3 validate_glm53.py` from this directory to exercise the real endpoint. It checks the model alias, arithmetic, reasoning separation, concurrent generation, structured tool calls, and streaming; full responses and timings are saved to `audit/validation.json`.

The existing FP8 KV configuration logs that no KV scaling factors were supplied and defaults them to 1.0. Functional tests do not establish long-context or benchmark accuracy. The API uses the existing unauthenticated LAN binding on port 8000.

### Verified result (2026-09-16)

All API validation checks passed after the corrected restart: alias, arithmetic with separated reasoning, four concurrent requests, structured `get_weather` tool call with Paris argument, and streaming. The worker can reach the head API. All eight GPU scheduler ranks are present.

Measured sample timings: arithmetic 0.52 s, tool call 0.61 s, streaming time to first generated token 0.39 s and total 1.61 s. These are short functional samples, not a throughput benchmark. Runtime configuration is recorded in `audit/server_info.json`; responses are in `audit/validation.json`.
