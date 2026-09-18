import asyncio
import json
import unittest
from unittest import mock

import web_search


def subrequest(query="SGLang latest release version", tool_type="web_search_20250305"):
    return {
        "model": "glm-5.3",
        "max_tokens": 128000,
        "stream": True,
        "system": [{"type": "text", "text": "You are an assistant for performing a web search tool use"}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": f"Perform a web search for the query: {query}"}]}],
        "tools": [{"type": tool_type, "name": "web_search", "max_uses": 8}],
        "tool_choice": {"type": "auto"},
    }


def fake_results():
    return [{"title": "Releases · sgl-project/sglang",
             "url": "https://github.com/sgl-project/sglang/releases",
             "content": "v0.5.19 is the latest release."}]


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
    await web_search.WebSearchMiddleware(app)(scope, receive, send)
    return seen


class WebSearchTests(unittest.TestCase):
    def test_subrequest_rewritten_with_tavily_results(self):
        async def fake_tavily(query, include, exclude):
            self.assertEqual(query, "SGLang latest release version")
            self.assertEqual((include, exclude), ([], []))
            return fake_results()

        with mock.patch.object(web_search, "tavily_search", fake_tavily):
            seen = asyncio.run(run_middleware(json.dumps(subrequest()).encode()))
        rewritten = json.loads(seen["recv"]["body"])
        self.assertEqual(rewritten["tools"], [])
        self.assertNotIn("tool_choice", rewritten)
        text = rewritten["messages"][0]["content"]
        self.assertIn("v0.5.19", text)
        self.assertIn("https://github.com/sgl-project/sglang/releases", text)
        self.assertIn("SGLang latest release version", text)
        headers = dict(seen["scope"]["headers"])
        self.assertEqual(headers[b"content-length"], str(len(seen["recv"]["body"])).encode())

    def test_domain_filters_forwarded_to_tavily(self):
        seen_args = []

        async def fake_tavily(query, include, exclude):
            seen_args.append((include, exclude))
            return fake_results()

        body = subrequest()
        body["tools"][0]["allowed_domains"] = ["docs.sglang.ai", "github.com"]
        with mock.patch.object(web_search, "tavily_search", fake_tavily):
            asyncio.run(run_middleware(json.dumps(body).encode()))
        self.assertEqual(seen_args, [(["docs.sglang.ai", "github.com"], [])])

        body = subrequest()
        body["tools"][0]["blocked_domains"] = ["pinterest.com"]
        with mock.patch.object(web_search, "tavily_search", fake_tavily):
            asyncio.run(run_middleware(json.dumps(body).encode()))
        self.assertEqual(seen_args[-1], ([], ["pinterest.com"]))

    def test_both_filters_allowed_wins(self):
        async def fake_tavily(query, include, exclude):
            self.assertEqual(include, ["a.com"])
            self.assertEqual(exclude, [])
            return fake_results()

        body = subrequest()
        body["tools"][0]["allowed_domains"] = ["a.com"]
        body["tools"][0]["blocked_domains"] = ["b.com"]
        with mock.patch.object(web_search, "tavily_search", fake_tavily):
            asyncio.run(run_middleware(json.dumps(body).encode()))

    def test_interception_logged_with_tool_type(self):
        import contextlib, io
        async def fake_tavily(query, include, exclude):
            return fake_results()

        stderr = io.StringIO()
        with mock.patch.object(web_search, "tavily_search", fake_tavily), \
                contextlib.redirect_stderr(stderr):
            asyncio.run(run_middleware(json.dumps(subrequest(
                tool_type="web_search_20260318")).encode()))
        self.assertIn("web_search_20260318", stderr.getvalue())
        self.assertIn("SGLang latest release version", stderr.getvalue())

    def test_other_date_versions_and_bare_type(self):
        for tool_type in ("web_search_20241125", "web_search_20260209",
                          "web_search_20260318", "web_search"):
            body = subrequest(tool_type=tool_type)
            self.assertEqual(web_search.find_server_tools(body), [0], tool_type)

    def test_main_conversation_untouched_and_no_search(self):
        calls = []

        async def fake_tavily(query, include, exclude):
            calls.append(query)
            return fake_results()

        body = {"model": "glm-5.3", "tools": [
            {"name": "WebSearch", "description": "Search the web.",
             "input_schema": {"type": "object"}},
            {"name": "Read", "description": "Reads a file.",
             "input_schema": {"type": "object"}},
        ], "messages": [{"role": "user", "content": "hi"}]}
        payload = json.dumps(body).encode()
        with mock.patch.object(web_search, "tavily_search", fake_tavily):
            seen = asyncio.run(run_middleware(payload))
        self.assertEqual(seen["recv"]["body"], payload)  # byte-identical
        self.assertEqual(calls, [])

    def test_tavily_failure_forwards_original(self):
        async def failing(query, include, exclude):
            raise OSError("network down")

        payload = json.dumps(subrequest()).encode()
        with mock.patch.object(web_search, "tavily_search", failing):
            seen = asyncio.run(run_middleware(payload))
        self.assertEqual(seen["recv"]["body"], payload)

    def test_empty_results_reported(self):
        async def empty(query, include, exclude):
            return []

        with mock.patch.object(web_search, "tavily_search", empty):
            seen = asyncio.run(run_middleware(json.dumps(subrequest()).encode()))
        text = json.loads(seen["recv"]["body"])["messages"][0]["content"]
        self.assertIn("returned no results", text)

    def test_query_extraction_variants(self):
        body = subrequest(query="what is rust?")
        self.assertEqual(web_search.extract_query(body)[0], "what is rust?")
        body["messages"][0]["content"] = "arbitrary instruction text"
        self.assertEqual(web_search.extract_query(body)[0], "arbitrary instruction text")
        body["messages"][0]["content"] = [{"type": "image", "source": {}}]
        self.assertEqual(web_search.extract_query(body)[0], "")

    def test_identical_queries_cached(self):
        calls = []
        web_search._cache.clear()

        def counting(query, include_domains=None, exclude_domains=None):
            calls.append(query)
            return fake_results()

        with mock.patch.object(web_search, "run_tavily", counting):
            asyncio.run(run_middleware(json.dumps(subrequest()).encode()))
            asyncio.run(run_middleware(json.dumps(subrequest()).encode()))
        self.assertEqual(len(calls), 1)

    def test_non_messages_paths_untouched(self):
        payload = json.dumps(subrequest()).encode()
        seen = asyncio.run(run_middleware(payload, path="/v1/chat/completions"))
        self.assertEqual(seen["recv"]["body"], payload)


if __name__ == "__main__":
    unittest.main()
