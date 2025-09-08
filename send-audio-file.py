import argparse
import asyncio
import json
import platform
import uuid
from typing import Optional

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from loguru import logger


async def wait_for_ice_gathering_complete(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    async def _on_ice_gathering_state_change() -> None:  # type: ignore[no-redef]
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


async def stream_file(host: str, file_path: str, data_channel_label: Optional[str] = "rtvi") -> int:
    pc = RTCPeerConnection()

    # Log state changes for visibility
    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:  # type: ignore[no-redef]
        logger.info(f"connectionState={pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()

    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change() -> None:  # type: ignore[no-redef]
        logger.info(f"iceConnectionState={pc.iceConnectionState}")
        if pc.iceConnectionState in ("failed", "disconnected"):
            await pc.close()

    @pc.on("track")
    def on_track(track) -> None:  # type: ignore[no-redef]
        logger.info(f"Remote track received: kind={getattr(track, 'kind', '?')}")

        @track.on("ended")
        async def on_ended() -> None:  # type: ignore[no-redef]
            logger.info("Remote track ended")

    # Create a data channel and send a proper RTVI client-ready message if requested
    channel = None
    if data_channel_label:
        channel = pc.createDataChannel(data_channel_label)

        @channel.on("open")
        def on_open() -> None:  # type: ignore[no-redef]
            try:
                client_ready = {
                    "label": "rtvi-ai",
                    "type": "client-ready",
                    "id": str(uuid.uuid4()),
                    "data": {
                        "version": "1.0.0",
                        "about": {
                            "library": "stt-timing-client",
                            "library_version": "0.1.0",
                            "platform": "python",
                            "platform_version": platform.python_version(),
                            "platform_details": {
                                "system": platform.system(),
                                "release": platform.release(),
                            },
                        },
                    },
                }
                msg = json.dumps(client_ready)
                channel.send(msg)
                logger.info(f"Sent RTVI client-ready on data channel '{data_channel_label}'")
            except Exception as e:
                logger.warning(f"Failed sending client-ready on data channel: {e}")

        @channel.on("message")
        def on_message(message) -> None:  # type: ignore[no-redef]
            try:
                text = (
                    message.decode("utf-8") if isinstance(message, (bytes, bytearray)) else message
                )
            except Exception:
                text = str(message)
            logger.info(f"RTVI: {text}")

    # Prepare local media (audio only)
    player = MediaPlayer(file_path)
    if player.audio is None:
        logger.error(f"No audio stream found in file: {file_path}")
        await pc.close()
        return 2

    pc.addTrack(player.audio)

    # Create offer and gather ICE candidates
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await wait_for_ice_gathering_complete(pc)

    # Exchange SDP with the server's runner endpoint
    async with aiohttp.ClientSession() as session:
        url = host.rstrip("/") + "/api/offer"
        logger.info(f"POST {url}")
        async with session.post(
            url,
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"/api/offer status={resp.status} body={text}")
                await pc.close()
                return 3
            data = await resp.json()

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=data["sdp"], type=data["type"])  # type: ignore[index]
    )

    # Wait for local audio to finish streaming
    try:
        logger.info("Streaming audio...")
        while getattr(player.audio, "readyState", "live") == "live" and pc.connectionState not in (
            "failed",
            "closed",
        ):
            await asyncio.sleep(0.2)
    finally:
        logger.info("Shutting down peer connection")
        await pc.close()

    logger.info("Done.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream an audio file to a local WebRTC bot.")
    parser.add_argument(
        "--host",
        default="http://localhost:7860",
        help="Bot base URL hosting /api/offer (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--file",
        default="test.m4a",
        help="Audio file to stream (default: test.m4a)",
    )
    parser.add_argument(
        "--no-data-channel",
        action="store_true",
        help="Do not create a data channel or send client-ready message.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = None if args.no_data_channel else "rtvi"
    exit_code = asyncio.run(stream_file(args.host, args.file, label))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
