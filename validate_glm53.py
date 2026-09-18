#!/usr/bin/env python3
"""Exercise the deployed API; save full responses and request timings."""
import concurrent.futures
import json
import pathlib
import time
import urllib.request

BASE = 'http://10.78.202.30:8000'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def request(path, payload=None):
    req = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    return OPENER.open(req, timeout=180)

def chat(prompt, **kwargs):
    payload = dict(model='glm-5.3', messages=[dict(role='user', content=prompt)], max_tokens=512, temperature=0, reasoning_effort='low', chat_template_kwargs={'clear_thinking': True})
    payload.update(kwargs)
    start = time.monotonic()
    with request('/v1/chat/completions', payload) as response:
        result = json.load(response)
    return dict(seconds=time.monotonic()-start, response=result)

results = {}
results['models'] = json.load(request('/v1/models'))
assert results['models']['data'][0]['id'] == 'glm-5.3'
results['arithmetic'] = chat('What is 17 times 23? Answer briefly.', reasoning_effort='high')
msg = results['arithmetic']['response']['choices'][0]['message']
assert '391' in msg['content'], msg
assert '</think>' not in msg['content'], msg
assert msg.get('reasoning_content'), msg
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results['concurrent'] = list(pool.map(chat, ['Reply with the capital of France.', 'What is 6 times 7?', 'Translate hello into Spanish.', 'Name the largest planet in our solar system.']))
for item in results['concurrent']:
    assert item['response']['choices'][0]['message']['content'], item
results['tool_call'] = chat('Use get_weather to look up the weather in Paris.', tools=[{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get current weather for a city.', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}], tool_choice='auto')
tool_msg = results['tool_call']['response']['choices'][0]['message']
assert tool_msg['tool_calls'][0]['function']['name'] == 'get_weather', tool_msg
assert 'Paris' in json.loads(tool_msg['tool_calls'][0]['function']['arguments'])['city'], tool_msg
payload = dict(model='glm-5.3', messages=[dict(role='user', content='Explain tensor parallelism in three sentences.')], max_tokens=512, temperature=0, stream=True, stream_options={'include_usage': True}, reasoning_effort='low')
t0 = time.monotonic(); first = None; chunks = []; content = ''
with request('/v1/chat/completions', payload) as response:
    for raw in response:
        if not raw.startswith(b'data: '): continue
        raw = raw[6:].strip()
        if raw == b'[DONE]': break
        event = json.loads(raw); chunks.append(event)
        for choice in event.get('choices', []):
            delta = choice['delta']
            if first is None and (delta.get('content') or delta.get('reasoning_content')): first = time.monotonic()
            content += delta.get('content') or ''
assert content and '</think>' not in content, content
results['stream'] = dict(ttft_seconds=None if first is None else first-t0, elapsed_seconds=time.monotonic()-t0, content=content, chunks=chunks)
pathlib.Path('audit/validation.json').write_text(json.dumps(results, indent=2))
print(json.dumps({k: v['seconds'] for k,v in results.items() if isinstance(v,dict) and 'seconds' in v}, indent=2))
print('Streaming TTFT:',results['stream']['ttft_seconds'],'elapsed:',results['stream']['elapsed_seconds'])
print('PASS: model alias, arithmetic, reasoning separation, four concurrent requests, tool call, streaming')
