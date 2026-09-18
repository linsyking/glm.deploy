"""Transport-only reasoning redaction for SGLang's three client APIs.

Never modifies requests, generation parameters, engine results, or stored history.
Only copies serialized outbound JSON/SSE before delivery to the HTTP client.
"""
import json

HIDDEN = {'reasoning', 'thinking', 'redacted_thinking'}

def redact(body, api):
    if api == 'chat':
        for choice in body.get('choices', []):
            for key in ('message', 'delta'):
                message = choice.get(key)
                if isinstance(message, dict):
                    message.pop('reasoning_content', None)
    elif api == 'responses':
        if isinstance(body.get('output'), list):
            body['output'] = [x for x in body['output'] if x.get('type') not in HIDDEN]
    elif api == 'messages' and isinstance(body.get('content'), list):
        body['content'] = [x for x in body['content'] if x.get('type') not in HIDDEN]
        if not body['content']:
            body['content'] = [{'type':'text','text':''}]
    return body

class StreamFilter:
    def __init__(self, api):
        self.api = api
        self.hidden = set()
        self.indexes = {}
        self.sequence = 0

    def event(self, data):
        kind = data.get('type', '')
        if self.api == 'messages':
            if kind == 'message_start':
                redact(data.get('message', {}), self.api)
            if kind.startswith('content_block_'):
                index = data['index']
                if kind == 'content_block_start':
                    if data.get('content_block', {}).get('type') in HIDDEN:
                        self.hidden.add(index)
                    else:
                        self.indexes[index] = len(self.indexes)
                if index in self.hidden:
                    return None
                data['index'] = self.indexes[index]
        elif self.api == 'responses':
            if 'response' in data:
                redact(data['response'], self.api)
            if kind.startswith('response.reasoning'):
                return None
            if data.get('item', {}).get('type') in HIDDEN:
                self.hidden.add(data.get('output_index'))
                return None
            if 'output_index' in data:
                index = data['output_index']
                if index in self.hidden:
                    return None
                if index not in self.indexes:
                    self.indexes[index] = len(self.indexes)
                data['output_index'] = self.indexes[index]
            if 'sequence_number' in data:
                data['sequence_number'] = self.sequence
                self.sequence += 1
        else:
            redact(data, self.api)
        return data

    def frame(self, frame):
        lines = frame.decode('utf-8').splitlines()
        payload = '\n'.join(x[5:].lstrip() for x in lines if x.startswith('data:'))
        if not payload or payload == '[DONE]':
            return frame + b'\n\n'
        data = self.event(json.loads(payload))
        if data is None:
            return b''
        prefix = [x for x in lines if not x.startswith('data:')]
        return ('\n'.join(prefix + ['data: '+json.dumps(data,ensure_ascii=False)])+'\n\n').encode()

class HideThinkingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get('path','')
        api = ('chat' if path == '/v1/chat/completions' else
               'messages' if path == '/v1/messages' else
               'responses' if path == '/v1/responses' or path.startswith('/v1/responses/') else None)
        if scope['type'] != 'http' or api is None:
            return await self.app(scope, receive, send)
        mode = None
        buffer = b''
        filter_ = StreamFilter(api)

        async def filtered_send(message):
            nonlocal mode, buffer
            if message['type'] == 'http.response.start':
                headers = dict(message.get('headers', []))
                content_type = headers.get(b'content-type',b'')
                if 200 <= message['status'] < 300:
                    mode = 'sse' if b'text/event-stream' in content_type else 'json' if b'json' in content_type else None
                if mode:
                    message = dict(message, headers=[(k,v) for k,v in message['headers'] if k.lower() not in (b'content-length', b'etag')])
                await send(message)
            elif message['type'] == 'http.response.body' and mode:
                buffer += message.get('body',b'')
                final = not message.get('more_body',False)
                if mode == 'json':
                    if final:
                        data = redact(json.loads(buffer),api)
                        await send(dict(message,body=json.dumps(data,ensure_ascii=False).encode()))
                else:
                    # SGLang emits LF SSE; also accept CRLF from other adapters.
                    buffer = buffer.replace(b'\r\n',b'\n')
                    while b'\n\n' in buffer:
                        frame, buffer = buffer.split(b'\n\n',1)
                        output = filter_.frame(frame)
                        if output:
                            await send({'type':'http.response.body','body':output,'more_body':True})
                    if final:
                        if buffer.strip():
                            await send({'type':'http.response.body','body':filter_.frame(buffer),'more_body':True})
                        await send({'type':'http.response.body','body':b'','more_body':False})
            else:
                await send(message)
        await self.app(scope, receive, filtered_send)
