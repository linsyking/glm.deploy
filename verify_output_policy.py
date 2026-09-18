"""Capture live answer/tool JSON and SSE before/after transport redaction."""
import json, sys, urllib.request
from pathlib import Path
from hide_thinking import redact, StreamFilter

phase=sys.argv[1]
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
results={}
tool={'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}
for api,path in [('chat','chat/completions'),('responses','responses'),('messages','messages')]:
    for use_tool in (False,True):
        prompt='Use get_weather for Paris.' if use_tool else 'What is 17 times 23? Answer briefly.'
        p={'model':'glm-5.3','temperature':0}
        if api=='responses':
            p.update(input=prompt,max_output_tokens=512,reasoning={'effort':'high'})
            if use_tool:p['tools']=[dict(type='function',**tool)]
        else:
            p.update(messages=[{'role':'user','content':prompt}],max_tokens=512)
            if api=='chat':
                p['reasoning_effort']='high'
                if use_tool:p['tools']=[{'type':'function','function':tool}]
            else:
                p['output_config']={'effort':'high'}
                if use_tool:p['tools']=[{'name':tool['name'],'description':tool['description'],'input_schema':tool['parameters']}]
        for stream in (False,True):
            p['stream']=stream
            req=urllib.request.Request('http://10.78.202.30:8000/v1/'+path,data=json.dumps(p).encode(),headers={'Content-Type':'application/json','anthropic-version':'2023-06-01'})
            with op.open(req,timeout=120) as r:
                raw=r.read().decode()
            key=f'{api}-{use_tool}-{stream}'
            if stream:
                events=[json.loads(line[5:].strip()) for line in raw.splitlines() if line.startswith('data:') and line[5:].strip()!='[DONE]']
                f=StreamFilter(api)
                if phase=='hidden':
                    for e in events:
                        original=json.dumps(e,sort_keys=True)
                        out=f.event(json.loads(original))
                        assert out is not None, (key,e)
                        # Sequence/index renumbering may differ; ensure payload is already redacted.
                        assert '"thinking_delta"' not in original and '"reasoning_content"' not in original,(key,e)
                results[key]=events
            else:
                d=json.loads(raw)
                if phase=='hidden':assert redact(json.loads(raw),api)==d,(key,d)
                results[key]=d
            if not use_tool:assert '391' in raw,(key,raw)
            else:assert 'get_weather' in raw and 'Paris' in raw,(key,raw)
            print('PASS',phase,key,flush=True)
Path(f'audit/output-policy-{phase}.json').write_text(json.dumps(results,indent=2))
