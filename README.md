# Example code: STT timing and Python aiortc SmallWebRTCTransport connection

Note that the first bot run will need to install dependencies and download model weights. This may take 30s or so.

```python
# in terminal 1
uv run bot.py

# in terminal 2
uv run send-audio-file.py
```

The time-to-final-transcript metric is the time it took for the STT service to return a final transcript after the user stopped speaking event fires.

The bot logs transcription frames and a time-to-final-transcript metric. The bot also sends the time-to-final-transcript metric as an RTVI server message frame.

