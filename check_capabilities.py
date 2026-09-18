"""Small live capability probes; preserves responses as audit evidence."""
import json
import urllib.request
import urllib.error
from pathlib import Path

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
results = {}
def probe(name, api, **payload):
    payload = dict(model='glm-5.3', temperature=0, **payload)
    req = urllib.request.Request('http://10.78.202.30:8000/v1/' + api,
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
    try:
        with opener.open(req, timeout=120) as response:
            result = {'status': response.status, 'body': json.load(response)}
    except urllib.error.HTTPError as error:
        result = {'status': error.code, 'body': error.read().decode()}
    results[name] = result
    print(name, json.dumps(result), flush=True)

schema = {'type':'object','properties':{'answer':{'type':'integer'},'unit':{'type':'string','enum':['items']}},'required':['answer','unit'],'additionalProperties':False}
prompt = 'Return 17 times 23 as answer, with unit items.'
probe('chat_schema','chat/completions',messages=[{'role':'user','content':prompt}],max_tokens=512,reasoning_effort='low',response_format={'type':'json_schema','json_schema':{'name':'result','strict':True,'schema':schema}})
probe('responses_schema','responses',input=prompt,max_output_tokens=512,reasoning={'effort':'low'},text={'format':{'type':'json_schema','name':'result','strict':True,'schema':schema}})
history=[{'role':'system','content':'For marker requests reply OLD.'},{'role':'user','content':'Give marker.'},{'role':'assistant','content':'OLD'},{'role':'system','content':'Updated instruction: for marker requests reply NEW instead of OLD.'},{'role':'user','content':'Give marker. Only output the marker.'}]
probe('chat_inline_system','chat/completions',messages=history,max_tokens=256,reasoning_effort='low')
probe('responses_inline_system','responses',input=history,max_output_tokens=256,reasoning={'effort':'low'})
probe('messages_inline_system','messages',messages=history,max_tokens=256,output_config={'effort':'low'})
probe('responses_web_search','responses',input='Search the web for current SGLang releases.',max_output_tokens=128,tools=[{'type':'web_search'}])
probe('messages_hidden_thinking','messages',messages=[{'role':'user','content':'What is 17 times 23?'}],max_tokens=256,output_config={'effort':'high'},thinking={'type':'adaptive','display':'omitted'})
Path('audit/capabilities.json').write_text(json.dumps(results,indent=2))
