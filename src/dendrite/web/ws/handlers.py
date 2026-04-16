"""
WebSocket endpoint handlers.
"""

import asyncio

import msgpack
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from dendrite.utils.logger_central import get_logger
from dendrite.web.ws.bridge import QueueBridge

router = APIRouter()
logger = get_logger("WebSocket")


async def _relay(
    websocket: WebSocket,
    bridge: QueueBridge,
    channel: str,
    maxsize: int = 100,
    binary: bool = False,
) -> None:
    """Relay bridge channel to WebSocket with clean disconnect detection.

    Runs websocket.receive() concurrently with q.get() so that client
    disconnects are detected even when the channel is idle (no data flowing).
    Without this, a blocked q.get() would keep the subscriber alive forever.
    """
    q, history = bridge.subscribe(channel, maxsize=maxsize)
    disconnect = asyncio.ensure_future(websocket.receive())
    try:
        # Send history snapshot in batches, yielding between to avoid blocking
        BATCH = 50
        for i in range(0, len(history), BATCH):
            if disconnect.done():
                return
            for item in history[i : i + BATCH]:
                if binary:
                    await websocket.send_bytes(msgpack.packb(item, use_bin_type=True))  # type: ignore[arg-type]
                else:
                    await websocket.send_json(item)
            await asyncio.sleep(0)

        # Live relay loop
        while True:
            get_task = asyncio.ensure_future(q.get())
            done, _ = await asyncio.wait(
                [get_task, disconnect],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect in done:
                get_task.cancel()
                break
            data = get_task.result()
            if binary:
                await websocket.send_bytes(msgpack.packb(data, use_bin_type=True))  # type: ignore[arg-type]
            else:
                await websocket.send_json(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WS relay error on '{channel}': {e}")
    finally:
        disconnect.cancel()
        bridge.unsubscribe(channel, q)


@router.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    """Real-time telemetry stream (JSON, ~1Hz)."""
    await websocket.accept()
    await _relay(websocket, _get_bridge(websocket), "telemetry")


@router.websocket("/ws/visualization")
async def visualization_ws(websocket: WebSocket):
    """Real-time visualization data stream (msgpack binary, ~100Hz)."""
    await websocket.accept()
    await _relay(websocket, _get_bridge(websocket), "visualization", maxsize=200, binary=True)


@router.websocket("/ws/mode_data")
async def mode_data_ws(websocket: WebSocket):
    """Mode output stream (msgpack binary, event-driven)."""
    await websocket.accept()
    await _relay(websocket, _get_bridge(websocket), "mode_data", maxsize=100, binary=True)


@router.websocket("/ws/training")
async def training_ws(websocket: WebSocket):
    """Real-time training progress stream (JSON, per-epoch)."""
    await websocket.accept()
    await _relay(websocket, _get_bridge(websocket), "training")


def _get_bridge(websocket: WebSocket) -> QueueBridge:
    """Get the QueueBridge from app state."""
    return websocket.app.state.queue_bridge
