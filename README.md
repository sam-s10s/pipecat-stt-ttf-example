# Example code: STT timing and Python aiortc SmallWebRTCTransport connection

Note that the first bot run will need to install dependencies and download model weights. This may take 30s or so.

Set up environment variables in an .env file.

```bash
cp env.example .env
# put in your API key(s)
```

If you are using local copies of Pipecat and the SMX Voice SDK, you need to install them as editable dependencies before running the bot.

```bash
# install editable LOCAL dependencies (optional)
uv add --editable "../../../speechmatics-python-sdk/sdk/voice"
uv add --editable "../../pipecat"
uv add --editable "../../pipecat[webrtc,deepgram,speechmatics,silero,runner]"
```

Start the bot. Then send an audio file. `send-audio-file.py` accepts a `--file` argument, which defaults to `test.m4a`.

```bash
# in terminal 1
uv run bot.py

# in terminal 2
uv run send-audio-file.py
```

The time-to-final-transcript metric is the time it took for the STT service to return a final transcript after the user stopped speaking event fires.

The bot logs transcription frames and a time-to-final-transcript metric. The bot also sends the time-to-final-transcript metric as an RTVI server message frame.

The bot also logs the audio frames to SMX.

```bash
# view results
uv run plot_results.py output/test --sample-rate 16000
```
