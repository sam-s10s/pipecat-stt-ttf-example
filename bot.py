import json
import os
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
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.speechmatics.stt import (
    EndOfUtteranceMode,
    OperatingPoint,
    SpeechmaticsSTTService,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

# logger.remove(0)
# logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", level=logging.DEBUG)
# logging.getLogger("speechmatics.voice").setLevel(logging.DEBUG)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/default")


class SMXLogger:
    """SMX logger."""

    def __init__(self, name: str):
        """Initialize the SMX logger."""
        self._path = os.path.join(OUTPUT_DIR, name)
        self._name = name

        self._full_path = os.path.join(self._path, self._name + ".jsonl")

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def log_json(self, data):
        """Log a JSON object to the console."""
        if hasattr(data, "to_dict"):
            data = data.to_dict()
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

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._last_user_stopped_speaking_time = datetime.now() - timedelta(
                seconds=self._vad_analyzer.params.stop_secs
            )

        elif isinstance(frame, InterimTranscriptionFrame):
            logger.info(f"[{self._prefix} interim] {frame.text}")
            if self._log:
                self._log.log_json({"final": False, "text": frame.text})

        elif isinstance(frame, TranscriptionFrame):
            logger.info(f"[{self._prefix} final] {frame.text}")
            if self._log:
                self._log.log_json({"final": True, "text": frame.text})
            self._last_final_time = datetime.now()

        await self.maybe_emit_metrics()
        await self.push_frame(frame, direction)


class SpeakingStartStopLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._smx_logger = SMXLogger("speaking")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._smx_logger.log_json({"speaking": False})

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._smx_logger.log_json({"speaking": True})

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
        if os.getenv("SMX_LOG_PATH"):
            smx_log_path = OUTPUT_DIR
            if os.path.exists(smx_log_path):
                for file in os.listdir(smx_log_path):
                    os.remove(os.path.join(smx_log_path, file))

        stt_speechmatics = SpeechmaticsSTTService(
            api_key=os.getenv("SPEECHMATICS_API_KEY"),
            params=SpeechmaticsSTTService.InputParams(
                max_delay=4.0,
                end_of_utterance_silence_trigger=0.5,
                end_of_utterance_mode=EndOfUtteranceMode.ADAPTIVE,
                operating_point=OperatingPoint.ENHANCED,
                enable_vad=True,
            ),
        )

        stt_deepgram = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        )

        stt_assemblyai = AssemblyAISTTService(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
        )

        stt_cartesia = CartesiaSTTService(
            api_key=os.getenv("CARTESIA_API_KEY"),
        )

        audio_logger = AudioLogger()
        speaking_logger = SpeakingStartStopLogger()

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
                                stt_speechmatics,
                                TranscriptionMetricsLogger(
                                    rtvi, transport._params.vad_analyzer, "🚀", "stt_speechmatics"
                                ),
                            ]
                        )
                    ],
                    [
                        Pipeline(
                            [
                                stt_deepgram,
                                TranscriptionMetricsLogger(
                                    rtvi, transport._params.vad_analyzer, "🦊", "stt_deepgram"
                                ),
                            ]
                        )
                    ],
                    [
                        Pipeline(
                            [
                                stt_assemblyai,
                                TranscriptionMetricsLogger(
                                    rtvi, transport._params.vad_analyzer, "🧤", "stt_assemblyai"
                                ),
                            ]
                        )
                    ],
                    [
                        Pipeline(
                            [
                                stt_cartesia,
                                TranscriptionMetricsLogger(
                                    rtvi, transport._params.vad_analyzer, "🐸", "stt_cartesia"
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
