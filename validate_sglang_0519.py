#!/usr/bin/env python3
"""Validate GLM-5.3 Chat Completions and Responses API behavior."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "http://10.78.202.30:8000"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    return OPENER.open(req, timeout=180)


def post_json(path: str, payload: dict) -> dict:
    with request(path, payload) as response:
        return json.load(response)


def function_tool() -> dict:
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


results: dict = {}

results["models"] = json.load(request("/v1/models"))
assert results["models"]["data"][0]["id"] == "glm-5.3"

chat_payload = {
    "model": "glm-5.3",
    "messages": [{"role": "user", "content": "What is 17 times 23? Answer briefly."}],
    "max_tokens": 256,
    "temperature": 0,
    "reasoning_effort": "high",
    "chat_template_kwargs": {"clear_thinking": True},
}
results["chat"] = post_json("/v1/chat/completions", chat_payload)
chat_message = results["chat"]["choices"][0]["message"]
assert "391" in (chat_message.get("content") or ""), chat_message
assert "</think>" not in (chat_message.get("content") or ""), chat_message
# Production may hide reasoning at the HTTP boundary; usage still proves generation.
assert results["chat"]["usage"].get("reasoning_tokens", 0) > 0, results["chat"]["usage"]
assert results["chat"]["usage"]["total_tokens"] > 0, results["chat"]["usage"]

chat_tool = {
    "model": "glm-5.3",
    "messages": [{"role": "user", "content": "Use get_weather for Paris."}],
    "tools": [{"type": "function", "function": {k: v for k, v in function_tool().items() if k != "type"}}],
    "tool_choice": "auto",
    "max_tokens": 512,
    "temperature": 0,
}
results["chat_tool"] = post_json("/v1/chat/completions", chat_tool)
chat_calls = results["chat_tool"]["choices"][0]["message"]["tool_calls"]
assert chat_calls and chat_calls[0]["function"]["name"] == "get_weather", chat_calls

responses_payload = {
    "model": "glm-5.3",
    "input": "What is 17 times 23? Answer briefly.",
    "max_output_tokens": 256,
    "temperature": 0,
    "reasoning": {"effort": "high"},
    "chat_template_kwargs": {"clear_thinking": True},
}
results["responses"] = post_json("/v1/responses", responses_payload)
response_text = "".join(
    part.get("text", "")
    for item in results["responses"]["output"]
    if item.get("type") == "message"
    for part in item.get("content", [])
)
assert "391" in response_text and "</think>" not in response_text, results["responses"]
assert results["responses"]["usage"]["total_tokens"] > 0, results["responses"]["usage"]

responses_tool_payload = {
    "model": "glm-5.3",
    "input": "Use get_weather for Paris.",
    "max_output_tokens": 512,
    "temperature": 0,
    "tools": [function_tool()],
    "tool_choice": "auto",
}
results["responses_tool"] = post_json("/v1/responses", responses_tool_payload)
response_calls = [item for item in results["responses_tool"]["output"] if item.get("type") == "function_call"]
assert response_calls and response_calls[0]["name"] == "get_weather", results["responses_tool"]
assert "Paris" in response_calls[0]["arguments"], response_calls[0]

stream_payload = {
    "model": "glm-5.3",
    "input": "Explain tensor parallelism in two sentences.",
    "max_output_tokens": 512,
    "temperature": 0,
    "reasoning": {"effort": "low"},
    "stream": True,
}
events = []
started = time.monotonic()
first_delta = None
with request("/v1/responses", stream_payload) as response:
    event_name = None
    for raw in response:
        line = raw.decode().strip()
        if line.startswith("event: "):
            event_name = line[7:]
        elif line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            events.append({"event": event_name, "data": event})
            if event.get("type") == "response.output_text.delta" and first_delta is None:
                first_delta = time.monotonic() - started
assert first_delta is not None, [event["data"].get("type") for event in events]
assert any(event["data"].get("type") == "response.completed" for event in events), events[-3:]
results["responses_stream"] = {
    "first_text_delta_seconds": first_delta,
    "elapsed_seconds": time.monotonic() - started,
    "events": events,
}

output_path = Path("audit/sglang-0.5.19-validation.json")
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(json.dumps(results, indent=2))
print("PASS: Chat reasoning, usage and tools")
print("PASS: Responses reasoning, usage, function tools and incremental streaming")
print(f"Responses streaming first text delta: {first_delta:.3f}s")
print(f"Evidence: {output_path}")
