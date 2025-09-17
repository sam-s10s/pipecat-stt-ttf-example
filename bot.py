import json
import logging
import os
import sys
from datetime import datetime, timedelta

import aiohttp
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
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
from pipecat.services.speechmatics.stt import (
    EndOfUtteranceMode,
    OperatingPoint,
    SpeechmaticsSTTService,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

# Enable SMX Voice API logging
logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger("speechmatics.voice").setLevel(logging.DEBUG)

# Default output directory (set by env variable)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/default")


class SMXLogger:
    """SMX logger."""

    def __init__(self, name: str):
        """Initialize the SMX logger."""
        self._name = name
        self._full_path = os.path.join(OUTPUT_DIR, self._name + ".jsonl")

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    def log_json(self, data):
        """Log a JSON object to the console."""
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        elif hasattr(data, "model_dump"):
            data = data.model_dump()
        elif isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
        elif isinstance(data, bytes):
            data = np.frombuffer(data, dtype=np.int16).tolist()
        with open(self._full_path, "a") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(), "payload": data}) + "\n")


class TranscriptionMetricsLogger(FrameProcessor):
    def __init__(
        self, rtvi: RTVIProcessor, vad_analyzer: VADAnalyzer, prefix: str = None, log: str = None
    ):
        super().__init__()
        self._last_final_time = None
        self._last_user_stopped_speaking_time = None
        self._rtvi = rtvi
        self._vad_analyzer = vad_analyzer
        self._prefix = prefix

        self._buffer: list[str] = []

        if log:
            self._log = SMXLogger(log)

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

        if isinstance(frame, UserStartedSpeakingFrame):
            if self._log:
                result = " ".join(self._buffer)
                if result:
                    self._log.log_json({"type": "turn", "text": result})
            self._buffer = []

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._vad_analyzer:
                self._last_user_stopped_speaking_time = datetime.now() - timedelta(
                    seconds=self._vad_analyzer.params.stop_secs
                )

        elif isinstance(frame, InterimTranscriptionFrame):
            logger.info(f"[{self._prefix} interim] {frame.text}")
            if self._log:
                result = self.as_json(frame.result)
                self._log.log_json({"type": "interim", "text": frame.text, "result": result})

        elif isinstance(frame, TranscriptionFrame):
            logger.info(f"[{self._prefix} final] {frame.text}")
            self._buffer.append(frame.text)
            if self._log:
                result = self.as_json(frame.result)
                self._log.log_json({"type": "final", "text": frame.text, "result": result})
            self._last_final_time = datetime.now()

        await self.maybe_emit_metrics()
        await self.push_frame(frame, direction)

    def as_json(self, data):
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        elif hasattr(data, "model_dump"):
            data = data.model_dump()
        elif isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
        return data


class SpeakingStartStopLogger(FrameProcessor):
    def __init__(self, snaffle: bool = False):
        super().__init__()
        self._smx_logger = SMXLogger("speaking")
        self._snaffle = snaffle

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._smx_logger.log_json({"speaking": False, "direction": direction.value})
            if self._snaffle:
                return

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._smx_logger.log_json({"speaking": True, "direction": direction.value})
            if self._snaffle:
                return
        await self.push_frame(frame, direction)


class AudioLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._smx_logger = SMXLogger("audio")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            self._smx_logger.log_json(frame.audio)

        await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    async with aiohttp.ClientSession() as session:
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                os.remove(os.path.join(OUTPUT_DIR, file))

        audio_logger = AudioLogger()
        speaking_logger = SpeakingStartStopLogger(snaffle=False)

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                audio_logger,
                speaking_logger,
                rtvi,
                ParallelPipeline(
                    [
                        Pipeline(
                            [
                                SpeechmaticsSTTService(
                                    api_key=os.getenv("SPEECHMATICS_API_KEY"),
                                    params=SpeechmaticsSTTService.InputParams(
                                        operating_point=OperatingPoint.STANDARD,
                                        max_delay=0.7,
                                        end_of_utterance_silence_trigger=0.2,
                                        end_of_utterance_mode=EndOfUtteranceMode.FIXED,
                                        enable_vad=False,
                                        enable_diarization=False,
                                    ),
                                ),
                                TranscriptionMetricsLogger(
                                    rtvi,
                                    transport._params.vad_analyzer,
                                    "🚀",
                                    "stt_smx_fast",
                                ),
                            ]
                        )
                    ],
                    [
                        Pipeline(
                            [
                                SpeechmaticsSTTService(
                                    api_key=os.getenv("SPEECHMATICS_API_KEY"),
                                    params=SpeechmaticsSTTService.InputParams(
                                        max_delay=1.5,
                                        end_of_utterance_silence_trigger=0.5,
                                        end_of_utterance_mode=EndOfUtteranceMode.FIXED,
                                        enable_vad=False,
                                    ),
                                ),
                                TranscriptionMetricsLogger(
                                    rtvi,
                                    transport._params.vad_analyzer,
                                    "🚀",
                                    "stt_smx_agent",
                                ),
                            ]
                        )
                    ],
                    [
                        Pipeline(
                            [
                                SpeechmaticsSTTService(
                                    api_key=os.getenv("SPEECHMATICS_API_KEY"),
                                    params=SpeechmaticsSTTService.InputParams(
                                        max_delay=1.5,
                                        end_of_utterance_silence_trigger=0.4,
                                        end_of_utterance_mode=EndOfUtteranceMode.EXTERNAL,
                                        enable_vad=True,
                                    ),
                                ),
                                TranscriptionMetricsLogger(
                                    rtvi,
                                    transport._params.vad_analyzer,
                                    "🚀",
                                    "stt_smx_captions",
                                ),
                            ]
                        )
                    ],
                    [
                        Pipeline(
                            [
                                DeepgramSTTService(
                                    api_key=os.getenv("DEEPGRAM_API_KEY"),
                                ),
                                TranscriptionMetricsLogger(
                                    rtvi, transport._params.vad_analyzer, "🦊", "stt_deepgram"
                                ),
                            ]
                        )
                    ],
                    # [
                    #     Pipeline(
                    #         [
                    #             AssemblyAISTTService(
                    #                 api_key=os.getenv("ASSEMBLYAI_API_KEY"),
                    #             ),
                    #             TranscriptionMetricsLogger(
                    #                 rtvi, transport._params.vad_analyzer, "🧤", "stt_assemblyai"
                    #             ),
                    #         ]
                    #     )
                    # ],
                    # [
                    #     Pipeline(
                    #         [
                    #             CartesiaSTTService(
                    #                 api_key=os.getenv("CARTESIA_API_KEY"),
                    #             ),
                    #             TranscriptionMetricsLogger(
                    #                 rtvi, transport._params.vad_analyzer, "🐸", "stt_cartesia"
                    #             ),
                    #         ]
                    #     )
                    # ],
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
