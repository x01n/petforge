from __future__ import annotations

import asyncio
import sys

from core.adapters.direct import OpenAIChatSSEAdapter
from core.events import AudioChunk
from services.conversation import ConversationService
from services.model_routing import ChannelConfig, ModelRouter
from services.tts import SubprocessTTSBackend, TTSCoordinator


class _StreamingResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, first_audio: asyncio.Event, trace: list[str]) -> None:
        self.first_audio = first_audio
        self.trace = trace

    async def __aenter__(self) -> _StreamingResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"第一句。"}}]}'
        yield ""
        await asyncio.wait_for(self.first_audio.wait(), timeout=2.0)
        self.trace.append("model:second")
        yield 'data: {"choices":[{"delta":{"content":"第二句。"}}]}'
        yield ""
        yield "data: [DONE]"
        yield ""


class _StreamingClient:
    def __init__(self, response: _StreamingResponse) -> None:
        self.response = response

    def stream(self, *_args: object, **_kwargs: object) -> _StreamingResponse:
        return self.response


def test_first_model_sentence_reaches_stdio_worker_before_second_packet() -> None:
    async def scenario() -> None:
        first_audio = asyncio.Event()
        trace: list[str] = []
        audio: list[AudioChunk] = []
        response = _StreamingResponse(first_audio, trace)
        channel = ChannelConfig(
            "pipeline",
            base_url="https://pipeline.invalid/v1",
            model="demo",
            capabilities=frozenset({"streaming"}),
        )
        adapter = OpenAIChatSSEAdapter(channel, client=_StreamingClient(response))
        router = ModelRouter([channel], adapter_factory=lambda _channel: adapter)
        worker = (
            "import base64,json,sys; "
            "print(json.dumps({'type':'ready','protocol':'meapet.tts.jsonl',"
            "'version':1,'streaming':True}),flush=True); "
            "\nfor line in sys.stdin:"
            "\n p=json.loads(line); data=(p['language']+'|'+p['text']).encode('utf-8');"
            "\n print(json.dumps({'type':'audio','protocol':'meapet.tts.jsonl','version':1,"
            "'request_id':p['request_id'],"
            "'data':base64.b64encode(data).decode('ascii'),'sample_rate':24000,"
            "'channels':1,'sample_format':'s16le','final':False}),flush=True);"
            "\n print(json.dumps({'type':'done','protocol':'meapet.tts.jsonl','version':1,"
            "'request_id':p['request_id']}),flush=True)"
        )
        backend = SubprocessTTSBackend((sys.executable, "-u", "-c", worker))
        loop = asyncio.get_running_loop()

        def receive_audio(chunk: AudioChunk) -> None:
            if not chunk.data:
                return
            audio.append(chunk)
            trace.append(f"audio:{chunk.data.decode('utf-8')}")
            # TTS 的同步 audio sink 在 daemon 线程调用；跨线程唤醒 asyncio
            # Event 必须通过 loop 的线程安全入口，debug 模式下不能直接 set。
            loop.call_soon_threadsafe(first_audio.set)

        tts = TTSCoordinator(backend, audio_sink=receive_audio)
        service = ConversationService(router, tts=tts, tts_language="zh")
        try:
            health = await tts.start()
            assert health.available
            result = await service.complete("开始")
        finally:
            await service.aclose()
            await tts.aclose()
            await router.aclose()

        sentences: list[str] = []
        while sentence := service.presentation.pop_next_sentence():
            sentences.append(sentence.text)

        assert result.text == "第一句。第二句。"
        assert sentences == ["第一句。", "第二句。"]
        assert [chunk.data.decode("utf-8") for chunk in audio] == [
            "zh|第一句。",
            "zh|第二句。",
        ]
        assert trace.index("audio:zh|第一句。") < trace.index("model:second")

    asyncio.run(scenario())
