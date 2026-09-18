"""Tavily-backed execution for Anthropic-style web_search server tools.

Claude Code executes its WebSearch tool by sending a follow-up /v1/messages
request that declares Anthropic's SERVER-side tool (type "web_search_<date>").
SGLang's Anthropic adapter skips such tools, so the search silently returns
nothing. This middleware intercepts those requests, runs the query through
Tavily, and injects the results into the prompt so the model reports them.
Every other request passes through untouched; responses are never modified.

Enabled via SGLANG_WEB_SEARCH=1 with TAVILY_API_KEY set (output_policy.env).
"""
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# Anthropic date-versions its server tools (web_search_20250305 = the tool
# version released 2025-03-05). Accept the whole family plus a bare name.
SERVER_TOOL_RE = re.compile(r"^web_search(_\d{8})?$")
QUERY_RE = re.compile(r"for the query:\s*(.+)", re.S)

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 15
MAX_RESULTS = 8
SNIPPET_CHARS = 500
CACHE_TTL = 600
CACHE_LIMIT = 100
_cache = {}


def find_server_tools(body):
    """Indices of web_search server tools in body['tools']."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    return [i for i, t in enumerate(tools)
            if isinstance(t, dict) and SERVER_TOOL_RE.match(str(t.get("type", "")))]


def extract_query(body):
    """The query text from the last user message, and that message."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return "", None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(str(p.get("text", "")) for p in content
                            if isinstance(p, dict) and p.get("type") == "text")
        else:
            continue
        match = QUERY_RE.search(text)
        return (match.group(1) if match else text).strip(), message
    return "", None


def run_tavily(query, include_domains=None, exclude_domains=None):
    """Blocking Tavily search; returns a list of {title, url, content}."""
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return []
    payload = {"query": query, "max_results": MAX_RESULTS,
               "include_answer": False}
    if include_domains:
        payload["include_domains"] = list(include_domains)
    if exclude_domains:
        payload["exclude_domains"] = list(exclude_domains)
    request = urllib.request.Request(
        TAVILY_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=TAVILY_TIMEOUT) as response:
        data = json.load(response)
    return data.get("results", [])[:MAX_RESULTS]


async def tavily_search(query, include_domains=None, exclude_domains=None):
    """Cached async wrapper around run_tavily."""
    key = (query, tuple(include_domains or ()), tuple(exclude_domains or ()))
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    results = await asyncio.to_thread(run_tavily, *key)
    if len(_cache) >= CACHE_LIMIT:
        _cache.clear()
    _cache[key] = (now, results)
    return results


def build_search_message(query, results):
    lines = [f'Web search results for "{query}" (Tavily):']
    if not results:
        lines.append("The search returned no results.")
    for n, r in enumerate(results, 1):
        title = " ".join(str(r.get("title", "")).split())[:200]
        url = str(r.get("url", ""))
        snippet = " ".join(str(r.get("content", "")).split())[:SNIPPET_CHARS]
        lines.append(f"[{n}] {title}\n    {url}\n    {snippet}")
    lines.append(
        "Report the search findings to the caller: for each result give its "
        "title, URL, and a one-sentence summary, then end with a \"Sources:\" "
        "list of the URLs as markdown links. Do not attempt any further web "
        "searches."
    )
    return "\n\n".join(lines)


async def maybe_rewrite(body):
    """Rewrite a /v1/messages body if it is a web_search sub-request.

    Returns True when the body was rewritten in place.
    """
    indexes = find_server_tools(body)
    if not indexes:
        return False
    query, message = extract_query(body)
    if not query or message is None:
        return False
    # All web_search tool versions share the same search semantics; the
    # version only changes server-side execution features (dynamic filtering,
    # response inclusion) that this basic route does not provide. Domain
    # filters are honored via Tavily; Anthropic rejects both filters together,
    # so allowed_domains wins if a client ever sends both.
    tool = body["tools"][indexes[0]]
    include = [str(d) for d in tool.get("allowed_domains") or [] if d]
    exclude = [str(d) for d in tool.get("blocked_domains") or [] if d]
    if include and exclude:
        exclude = []
    results = await tavily_search(query, include, exclude)
    print(f"[web-search] {tool.get('type')} query={query!r} results={len(results)}"
          f" allowed_domains={include} blocked_domains={exclude}",
          file=sys.stderr, flush=True)
    tools = body["tools"]
    for i in reversed(indexes):
        del tools[i]
    if not tools:
        body.pop("tool_choice", None)
    message["content"] = build_search_message(query, results)
    return True


class WebSearchMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST" \
                or scope.get("path") != "/v1/messages":
            return await self.app(scope, receive, send)

        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        changed = False
        try:
            data = json.loads(body)
            changed = await maybe_rewrite(data)
        except (ValueError, UnicodeDecodeError, OSError, urllib.error.URLError):
            pass  # not JSON or Tavily unreachable: forward unchanged
        if changed:
            body = json.dumps(data, ensure_ascii=False).encode()

        body_delivered = False

        async def replay():
            nonlocal body_delivered
            if not body_delivered:
                body_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Deliver once, then hand back the original transport so streaming
            # disconnect listeners keep waiting instead of spinning.
            return await receive()

        headers = list(scope.get("headers") or [])
        if changed:
            headers = [
                (b"content-length", str(len(body)).encode())
                if name.lower() == b"content-length" else (name, value)
                for name, value in headers
            ]
        return await self.app(dict(scope, headers=headers), replay, send)
