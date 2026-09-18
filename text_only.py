"""Transport-level text-only enforcement for SGLang's three client APIs.

Rewrites incoming request JSON so a text-only model is never told it can view
images and never receives image/document content blocks. Requests that match
no rule are forwarded byte-identical. Responses are never modified.

Enabled at registration time via SGLANG_TEXT_ONLY (see output_policy.env and
the http_server.py patch next to HideThinkingMiddleware).
"""
import json
import re

# Rewritten APIs; /v1/responses subpaths (retrieval) are left alone.
PATHS = ("/v1/messages", "/v1/chat/completions", "/v1/responses")

# Bullet in Claude Code's Read tool description that claims image/PDF viewing.
IMAGE_CLAIM_RE = re.compile(
    r"Reads images \(PNG, JPG.*?as cells with outputs\.", re.S
)
IMAGE_CLAIM_REPLACEMENT = (
    "Text-only environment: you cannot view or analyze images — do not call "
    "Read on image files (PNG, JPG, etc.); instead tell the user you cannot "
    "see images. PDFs cannot be viewed either. Jupyter notebooks (.ipynb) are "
    "read as cells with outputs."
)
# Older/variant descriptions without the notebook tail.
IMAGE_CLAIM_FALLBACK_RE = re.compile(
    r"Reads images \(PNG, JPG.*?presents them visually\.", re.S
)
IMAGE_CLAIM_FALLBACK_REPLACEMENT = (
    "Cannot view images (text-only environment); do not attempt to read image "
    "files."
)


def rewrite_description(text):
    if "Reads images" not in text:
        return text
    text = IMAGE_CLAIM_RE.sub(IMAGE_CLAIM_REPLACEMENT, text)
    text = IMAGE_CLAIM_FALLBACK_RE.sub(IMAGE_CLAIM_FALLBACK_REPLACEMENT, text)
    return text


# Content-part type -> equivalent text-part type for each API family.
TEXT_TYPE_FOR = {
    "image": "text",              # Anthropic message/tool_result content
    "document": "text",            # Anthropic PDF blocks
    "image_url": "text",           # OpenAI Chat content parts
    "input_image": "input_text",   # OpenAI Responses input parts
    "input_file": "input_text",
    "input_audio": "input_text",
}
PLACEHOLDER_TEXT = "[Image or file content removed: this model is text-only.]"


def strip_image_parts(node, changed):
    if isinstance(node, list):
        return [strip_image_parts(item, changed) for item in node]
    if isinstance(node, dict):
        part_type = node.get("type")
        if isinstance(part_type, str) and part_type in TEXT_TYPE_FOR:
            changed.append(part_type)
            return {"type": TEXT_TYPE_FOR[part_type], "text": PLACEHOLDER_TEXT}
        return {k: strip_image_parts(v, changed) for k, v in node.items()}
    return node


def rewrite_request(body):
    """Mutate a parsed request body in place; return True if anything changed."""
    if not isinstance(body, dict):
        return False
    changed = []

    # Tool descriptions: Anthropic/Responses use {name, description, ...},
    # OpenAI Chat nests under {type: function, function: {description, ...}}.
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            holder = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            description = holder.get("description")
            if isinstance(description, str):
                rewritten = rewrite_description(description)
                if rewritten != description:
                    holder["description"] = rewritten
                    changed.append("tool")

    stripped = strip_image_parts(body.get("messages"), changed)
    if stripped is not None:
        body["messages"] = stripped
    if isinstance(body.get("input"), list):
        body["input"] = strip_image_parts(body["input"], changed)

    return bool(changed)


class TextOnlyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST" \
                or scope.get("path") not in PATHS:
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
            changed = rewrite_request(data)
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
            # Preserve the original transport's wait/disconnect behavior.
            # Replaying forever makes streaming disconnect listeners spin
            # without yielding and blocks the entire HTTP event loop.
            return await receive()

        headers = list(scope.get("headers") or [])
        if changed:
            headers = [
                (b"content-length", str(len(body)).encode())
                if name.lower() == b"content-length" else (name, value)
                for name, value in headers
            ]
        return await self.app(dict(scope, headers=headers), replay, send)
