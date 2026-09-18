import asyncio
import copy
import json
import unittest
from text_only import (
    PLACEHOLDER_TEXT, TextOnlyMiddleware, rewrite_description, rewrite_request,
)

REAL_READ_DESCRIPTION = (
    "Reads a file from the local filesystem.\n\n- `file_path` must be an absolute path.\n"
    "- Reads up to 2000 lines by default.\n"
    "- When you already know which part of the file you need, only read that part. "
    "This can be important for larger files.\n- Results are returned using cat -n format, "
    "with line numbers starting at 1\n"
    "- Reads images (PNG, JPG, …) and presents them visually. Reads PDFs via the "
    "`pages` parameter (e.g. \"1-5\", max 20 pages/request; required for PDFs over 10 pages). "
    "Reads Jupyter notebooks (.ipynb) as cells with outputs.\n"
    "- Reading a directory, a missing file, or an empty file returns an error or system reminder rather than content.\n"
    "- Do NOT re-read a file you just edited to verify — Edit/Write would have errored if the change failed, "
    "and the harness tracks file state for you."
)

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def anthropic_body():
    return {
        "model": "glm-5.3",
        "system": [{"type": "text", "text": "You are Claude Code."}],
        "tools": [{"name": "Read", "description": REAL_READ_DESCRIPTION,
                   "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Let me check."}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
                ]},
            ]},
        ],
        "max_tokens": 128,
    }


async def run_middleware(body_bytes, path="/v1/messages"):
    seen = {}

    async def receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    async def send(message):
        pass

    async def app(scope, recv, send_):
        request = {"scope": scope, "recv": await recv()}
        seen.update(request)
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        await send_({"type": "http.response.body", "body": b""})

    scope = {"type": "http", "method": "POST", "path": path, "headers": [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body_bytes)).encode()),
    ]}
    await TextOnlyMiddleware(app)(scope, receive, send)
    return seen


class TextOnlyTests(unittest.TestCase):
    def test_body_replayed_once_then_waits_for_real_disconnect(self):
        async def check():
            disconnect = asyncio.Event()
            calls = 0

            async def receive():
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {"type": "http.request", "body": b'{}', "more_body": False}
                await disconnect.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                pass

            async def app(scope, recv, send_):
                self.assertEqual((await recv())["body"], b'{}')
                pending = asyncio.create_task(recv())
                await asyncio.sleep(0)
                self.assertFalse(pending.done(), "receive must wait, not replay forever")
                disconnect.set()
                self.assertEqual(await pending, {"type": "http.disconnect"})

            await TextOnlyMiddleware(app)(
                {"type": "http", "method": "POST", "path": "/v1/messages"}, receive, send)
            self.assertEqual(calls, 2)

        asyncio.run(check())

    def test_read_description_rewritten(self):
        out = rewrite_description(REAL_READ_DESCRIPTION)
        self.assertNotIn("presents them visually", out)
        self.assertNotIn("Reads images", out)
        self.assertIn("cannot view or analyze images", out)
        # Untouched parts of the bullet list survive.
        self.assertIn("`file_path` must be an absolute path", out)
        self.assertIn("Reads up to 2000 lines by default", out)

    def test_variant_description_without_notebook_tail(self):
        out = rewrite_description("Reads images (PNG, JPG, ...) and presents them visually. Other stuff.")
        self.assertIn("Cannot view images", out)
        self.assertIn("Other stuff.", out)

    def test_anthropic_request_end_to_end(self):
        body = anthropic_body()
        payload = json.dumps(body).encode()
        seen = asyncio.run(run_middleware(payload))
        self.assertEqual(seen["scope"]["method"], "POST")
        rewritten = json.loads(seen["recv"]["body"])
        desc = rewritten["tools"][0]["description"]
        self.assertIn("cannot view or analyze images", desc)
        # User image block -> text placeholder.
        self.assertEqual(rewritten["messages"][0]["content"][1],
                         {"type": "text", "text": PLACEHOLDER_TEXT})
        # Image inside tool_result content also stripped.
        self.assertEqual(rewritten["messages"][2]["content"][0]["content"][0],
                         {"type": "text", "text": PLACEHOLDER_TEXT})
        # Content-length header updated to the new body size.
        headers = dict(seen["scope"]["headers"])
        self.assertEqual(headers[b"content-length"], str(len(seen["recv"]["body"])).encode())

    def test_openai_chat_and_responses_formats(self):
        chat = {"model": "glm-5.3", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ]}], "tools": [{"type": "function", "function": {"name": "Read", "description": REAL_READ_DESCRIPTION,
                                                         "parameters": {}}}]}
        seen = asyncio.run(run_middleware(json.dumps(chat).encode(), "/v1/chat/completions"))
        rewritten = json.loads(seen["recv"]["body"])
        self.assertEqual(rewritten["messages"][0]["content"][1]["type"], "text")
        self.assertIn("cannot view or analyze images", rewritten["tools"][0]["function"]["description"])

        responses = {"model": "glm-5.3", "input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": "hi"},
                {"type": "input_image", "image_url": "data:image/png;base64,x"},
            ]}]}
        seen = asyncio.run(run_middleware(json.dumps(responses).encode(), "/v1/responses"))
        rewritten = json.loads(seen["recv"]["body"])
        self.assertEqual(rewritten["input"][0]["content"][1],
                         {"type": "input_text", "text": PLACEHOLDER_TEXT})

    def test_no_match_passes_through_byte_identical(self):
        body = {"model": "glm-5.3", "messages": [{"role": "user", "content": "plain text"}]}
        payload = json.dumps(body).encode()
        seen = asyncio.run(run_middleware(payload))
        self.assertEqual(seen["recv"]["body"], payload)
        headers = dict(seen["scope"]["headers"])
        self.assertEqual(headers[b"content-length"], str(len(payload)).encode())

    def test_non_matching_paths_and_methods_keep_original_receive(self):
        original = {"type": "http.request", "body": b"x", "more_body": False}

        async def receive():
            return original

        async def send(message):
            pass

        async def app(scope, recv, send_):
            self.assertIs(recv, receive)
            self.assertEqual((await recv())["body"], b"x")

        scope = {"type": "http", "method": "POST", "path": "/v1/models", "headers": []}
        asyncio.run(TextOnlyMiddleware(app)(scope, receive, send))
        scope = {"type": "http", "method": "GET", "path": "/v1/messages", "headers": []}
        asyncio.run(TextOnlyMiddleware(app)(scope, receive, send))

    def test_malformed_json_forwarded(self):
        payload = b"not json at all"
        seen = asyncio.run(run_middleware(payload))
        self.assertEqual(seen["recv"]["body"], payload)

    def test_rewrite_request_in_place_semantics(self):
        body = anthropic_body()
        before = copy.deepcopy(body)
        self.assertTrue(rewrite_request(body))
        self.assertIn("cannot view", body["tools"][0]["description"])
        self.assertEqual(before["tools"][0]["description"], REAL_READ_DESCRIPTION)
        # A second pass over the already-rewritten body changes only the
        # remaining image blocks.
        self.assertFalse(rewrite_request({}))


if __name__ == "__main__":
    unittest.main()
