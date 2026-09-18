"""Strip Claude Code's attribution system block from /v1/messages requests.

The `x-anthropic-billing-header:` system block is Anthropic billing metadata
with no meaning on a self-hosted server. Most Claude Code versions keep it
byte-identical within a session (measured: 11/11 same-session requests, and
a 98.7% cumulative prefix-cache hit rate), but the SGLang docs warn that some
builds vary it per request, which would invalidate the prefix cache from the
first token on. Removing it keeps the prompt prefix stable for every client
version and trims ~20 tokens per request. Other requests pass through
byte-identical; responses are never modified.

Enabled via SGLANG_STRIP_ATTRIBUTION=1 (see output_policy.env).
"""
import json

PREFIX = "x-anthropic-billing-header:"


def _is_attribution(block):
    return (isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block["text"].lstrip().startswith(PREFIX))


def strip_attribution(body):
    """Remove the attribution system block in place; True if changed."""
    system = body.get("system")
    if isinstance(system, list):
        kept = [b for b in system if not _is_attribution(b)]
        if len(kept) != len(system):
            if kept:
                body["system"] = kept
            else:
                body.pop("system", None)
            return True
    elif isinstance(system, str) and system.lstrip().startswith(PREFIX):
        body.pop("system", None)
        return True
    return False


class StripAttributionMiddleware:
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
            changed = strip_attribution(data)
        except (ValueError, UnicodeDecodeError):
            pass  # not JSON: forward untouched
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
