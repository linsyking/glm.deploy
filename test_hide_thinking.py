import copy
import json
import unittest
import asyncio
from hide_thinking import redact, StreamFilter, HideThinkingMiddleware

class FilteringTests(unittest.TestCase):
    def test_chat_preserves_answer_tools_usage(self):
        raw={'choices':[{'message':{'content':'ANSWER','reasoning_content':'SECRET','tool_calls':[{'function':{'arguments':'{"thinking":"legitimate argument"}'}}]}}], 'usage':{'reasoning_tokens':42}}
        out=redact(copy.deepcopy(raw),'chat')
        self.assertNotIn('SECRET',json.dumps(out))
        self.assertEqual(out['usage'],raw['usage'])
        self.assertEqual(out['choices'][0]['message']['tool_calls'],raw['choices'][0]['message']['tool_calls'])
        self.assertEqual(raw['choices'][0]['message']['reasoning_content'],'SECRET')

    def test_anthropic_indexes(self):
        f=StreamFilter('messages')
        for event in [{'type':'content_block_start','index':0,'content_block':{'type':'thinking','thinking':'SECRET'}},{'type':'content_block_delta','index':0,'delta':{'type':'thinking_delta','thinking':'SECRET'}},{'type':'content_block_stop','index':0}]:
            self.assertIsNone(f.event(event))
        out=f.event({'type':'content_block_start','index':1,'content_block':{'type':'tool_use','name':'read','input':{}}})
        self.assertEqual(out['index'],0)
        self.assertEqual(f.event({'type':'content_block_stop','index':1})['index'],0)

    def test_responses_indexes_and_completed(self):
        f=StreamFilter('responses')
        self.assertIsNone(f.event({'type':'response.output_item.added','output_index':0,'item':{'type':'reasoning'}}))
        self.assertIsNone(f.event({'type':'response.reasoning_text.delta','delta':'SECRET'}))
        out=f.event({'type':'response.output_item.added','output_index':1,'sequence_number':4,'item':{'type':'message'}})
        self.assertEqual(out['output_index'],0)
        self.assertEqual(out['sequence_number'],0)
        out=f.event({'type':'response.completed','response':{'output':[{'type':'reasoning','content':'SECRET'},{'type':'message','content':'ANSWER'}],'usage':{'output_tokens':42}}})
        self.assertEqual(len(out['response']['output']),1)
        self.assertEqual(out['response']['usage']['output_tokens'],42)

    def test_transport_fragmentation_and_request_unchanged(self):
        source=('data: '+json.dumps({'choices':[{'delta':{'reasoning_content':'SECRET'}}]})+'\n\n'+
                'data: '+json.dumps({'choices':[{'delta':{'content':'391'}}]})+'\n\n'+'data: [DONE]\n\n').encode()
        for chunk_size in (1,7,len(source)):
            sent=[]
            async def receive(): return {'type':'http.request','body':b'original request'}
            async def send(message):sent.append(message)
            async def app(scope,recv,send_):
                self.assertIs(recv,receive)
                self.assertEqual((await recv())['body'],b'original request')
                await send_({'type':'http.response.start','status':200,'headers':[(b'content-type',b'text/event-stream')]})
                for offset in range(0,len(source),chunk_size):
                    await send_({'type':'http.response.body','body':source[offset:offset+chunk_size],'more_body':True})
                await send_({'type':'http.response.body','body':b'','more_body':False})
            asyncio.run(HideThinkingMiddleware(app)({'type':'http','path':'/v1/chat/completions'},receive,send))
            output=b''.join(m.get('body',b'') for m in sent)
            self.assertNotIn(b'SECRET',output)
            self.assertIn(b'391',output)
            self.assertIn(b'[DONE]',output)

if __name__=='__main__': unittest.main()
