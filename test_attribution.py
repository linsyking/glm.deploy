import asyncio
import json
import unittest

from attribution import StripAttributionMiddleware, strip_attribution

ATTRIBUTION = "x-anthropic-billing-header: cc_version=2.1.274.990; cc_entrypoint=sdk-cli;"


def messages_body(system):
    return {
        "model": "glm-5.3",
        "system": system,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
    }


async def run_middleware(body_bytes, path="/v1/messages"):
    seen = {}

    async def receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    async def send(message):
        pass

    async def app(scope, recv, send_):
        seen.update(scope=scope, recv=await recv())
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        await send_({"type": "http.response.body", "body": b""})

    scope = {"type": "http", "method": "POST", "path": path, "headers": [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body_bytes)).encode()),
    ]}
    await StripAttributionMiddleware(app)(scope, receive, send)
    return seen


class AttributionTests(unittest.TestCase):
    def test_strips_attribution_block_keeping_real_system(self):
        body = messages_body([
            {"type": "text", "text": ATTRIBUTION, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "You are Claude Code, an agent."},
        ])
        self.assertTrue(strip_attribution(body))
        self.assertEqual(body["system"],
                         [{"type": "text", "text": "You are Claude Code, an agent."}])

    def test_strips_string_form_and_only_attribution(self):
        body = messages_body(ATTRIBUTION)
        self.assertTrue(strip_attribution(body))
        self.assertNotIn("system", body)
        body = messages_body("You are helpful.")
        self.assertFalse(strip_attribution(body))
        self.assertEqual(body["system"], "You are helpful.")

    def test_no_attribution_passes_through_byte_identical(self):
        payload = json.dumps(messages_body([
            {"type": "text", "text": "You are Claude Code."},
        ])).encode()
        seen = asyncio.run(run_middleware(payload))
        self.assertEqual(seen["recv"]["body"], payload)
        headers = dict(seen["scope"]["headers"])
        self.assertEqual(headers[b"content-length"], str(len(payload)).encode())

    def test_middleware_end_to_end_rewrites_and_fixes_length(self):
        payload = json.dumps(messages_body([
            {"type": "text", "text": ATTRIBUTION},
            {"type": "text", "text": "Real system prompt."},
        ])).encode()
        seen = asyncio.run(run_middleware(payload))
        rewritten = json.loads(seen["recv"]["body"])
        self.assertEqual(rewritten["system"],
                         [{"type": "text", "text": "Real system prompt."}])
        headers = dict(seen["scope"]["headers"])
        self.assertEqual(headers[b"content-length"],
                         str(len(seen["recv"]["body"])).encode())

    def test_non_messages_paths_untouched(self):
        async def check():
            original = {"type": "http.request", "body": b"x", "more_body": False}

            async def receive():
                return original

            async def send(message):
                pass

            async def app(scope, recv, send_):
                self.assertIs((await recv())["body"], b"x")
                await send_({"type": "http.response.start", "status": 200, "headers": []})
                await send_({"type": "http.response.body", "body": b""})

            await StripAttributionMiddleware(app)(
                {"type": "http", "method": "POST", "path": "/v1/chat/completions"},
                receive, send)
            await StripAttributionMiddleware(app)(
                {"type": "http", "method": "GET", "path": "/v1/messages"},
                receive, send)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
