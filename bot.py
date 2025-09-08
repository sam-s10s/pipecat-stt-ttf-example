import asyncio
import os
from datetime import datetime, timedelta

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)


class TranscriptionMetricsLogger(FrameProcessor):
    def __init__(self, rtvi: RTVIProcessor, vad_analyzer: VADAnalyzer, prefix: str = None):
        super().__init__()
        self._last_final_time = None
        self._last_user_stopped_speaking_time = None
        self._rtvi = rtvi
        self._vad_analyzer = vad_analyzer
        self._prefix = prefix

    async def maybe_emit_metrics(self):
        if self._last_user_stopped_speaking_time and self._last_final_time:
            elapsed_td = self._last_final_time - self._last_user_stopped_speaking_time
            elapsed_seconds_str = f"{elapsed_td.total_seconds():.2f}"
            self._last_user_stopped_speaking_time = None
            self._last_final_time = None

            logger.info(f"[{self._prefix} TTF] {elapsed_seconds_str}s")
            await self._rtvi.push_frame(
                RTVIServerMessageFrame(
                    data={
                        "type": "time to final transcript",
                        "payload": {"elapsed": float(elapsed_seconds_str)},
                    }
                )
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._last_user_stopped_speaking_time = datetime.now() - timedelta(
                seconds=self._vad_analyzer.params.stop_secs
            )

        elif isinstance(frame, InterimTranscriptionFrame):
            logger.info(f"[{self._prefix} interim] {frame.text}")

        elif isinstance(frame, TranscriptionFrame):
            logger.info(f"[{self._prefix} final] {frame.text}")
            self._last_final_time = datetime.now()

        await self.maybe_emit_metrics()
        await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    async with aiohttp.ClientSession() as session:
        stt_deepgram = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        )

        stt_speechmatics = SpeechmaticsSTTService(
            api_key=os.getenv("SPEECHMATICS_API_KEY"),
        )

        stt = stt_speechmatics

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                ParallelPipeline(
                    [
                        Pipeline(
                            [
                                stt,
                                TranscriptionMetricsLogger(
                                    rtvi, transport._params.vad_analyzer, "🚀"
                                ),
                            ]
                        )
                    ],
                    [
                        Pipeline(
                            [
                                stt,
                                TranscriptionMetricsLogger(
                                    rtvi, transport._params.vad_analyzer, "🦊"
                                ),
                            ]
                        )
                    ],
                ),
                transport.output(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            logger.info("Client ready")

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            logger.info("Client connected")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False, force_gc=True)

        await runner.run(task)


transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
