"""Live streaming/image regression probes for the text-only middleware."""
import json
import urllib.request
from pathlib import Path

from test_text_only import PNG_B64

BASE = "http://10.78.202.30:8000"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
results = {}
prompt = "What is 17 times 23? Answer briefly."

for api in ("chat/completions", "responses", "messages"):
    body = {"model": "glm-5.3", "stream": True, "temperature": 0}
    if api == "responses":
        body.update(max_output_tokens=512, reasoning={"effort": "low"}, input=[
            {"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": "data:image/png;base64," + PNG_B64},
            ]}])
    else:
        image = ({"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG_B64}}
                 if api == "chat/completions" else
                 {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}})
        body.update(max_tokens=512, messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt}, image]}])
        if api == "messages":
            body["output_config"] = {"effort": "low"}
        else:
            body["reasoning_effort"] = "low"
    req = urllib.request.Request(BASE + "/v1/" + api, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    with opener.open(req, timeout=45) as response:
        with opener.open(BASE + "/health", timeout=5) as health:
            assert health.status == 200
        raw = response.read().decode()
    assert "391" in raw, raw
    # "error":null is a normal field of every Responses stream event, so check
    # for actual failure markers instead of the bare substring "error".
    for marker in ("response.failed", "event: error", '"error":{'):
        assert marker not in raw, (marker, raw)
    with opener.open(BASE + "/health", timeout=5) as health:
        assert health.status == 200
    results[api] = {"stream": raw, "health_during_and_after": 200}
    print("PASS image placeholder, streaming, health during/after:", api, flush=True)

Path("audit/text-only-live.json").write_text(json.dumps(results, indent=2))
