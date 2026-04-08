"""
Telemetry Poller

Polls SharedState and psutil for system metrics at 1Hz,
broadcasts telemetry JSON to the QueueBridge "telemetry" channel.
"""

import asyncio
import time

import psutil

from dendrite.utils.logger_central import get_logger
from dendrite.utils.state_keys import (
    channel_quality_key,
    mode_metric_key,
    stream_latency_key,
    stream_timestamp_key,
)
from dendrite.web.ws.bridge import QueueBridge


async def run_telemetry_poller(
    bridge: QueueBridge,
    get_pipeline_service,
    interval: float = 1.0,
) -> None:
    """Poll telemetry and broadcast to subscribers.

    Args:
        bridge: QueueBridge to broadcast on "telemetry" channel.
        get_pipeline_service: Callable that returns PipelineService.
        interval: Polling interval in seconds.
    """
    logger = get_logger("TelemetryPoller")
    logger.info("Telemetry poller started")

    # Cache psutil.Process objects for accurate readings
    process_cache: dict[int, psutil.Process] = {}

    while True:
        try:
            service = get_pipeline_service()

            if not service.is_recording:
                await asyncio.sleep(interval)
                continue

            shared_state = service.shared_state
            if shared_state is None:
                await asyncio.sleep(interval)
                continue

            telemetry = _build_telemetry(
                shared_state, service, process_cache
            )

            await bridge.broadcast("telemetry", telemetry)

        except Exception as e:
            logger.warning(f"Telemetry poll error: {e}")

        await asyncio.sleep(interval)


def _build_telemetry(shared_state, service, process_cache: dict) -> dict:
    """Build telemetry snapshot from SharedState and process info."""
    mem = psutil.virtual_memory()
    data = {
        "type": "telemetry",
        "timestamp": time.time(),
        "elapsed_s": service.elapsed_seconds,
        "streams": [],
        "modes": [],
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": mem.percent,
            "memory_used_gb": mem.used / (1024 ** 3),
            "memory_total_gb": mem.total / (1024 ** 3),
            "processes": [],
        },
    }

    # Stream latencies
    for stream_type in service.configured_stream_types:
        latency_key = stream_latency_key(stream_type)
        latency = shared_state.get(latency_key)
        if latency is not None:
            ts_key = stream_timestamp_key(stream_type)
            data["streams"].append({
                "type": stream_type,
                "latency_ms": latency,
                "last_update": shared_state.get(ts_key),
            })

    # Mode metrics
    for mode_name in service.mode_pids:
        mode_data = {"name": mode_name}
        for metric in ["accuracy", "confidence", "kappa", "internal_ms", "inference_ms"]:
            key = mode_metric_key(mode_name, metric)
            val = shared_state.get(key)
            if val is not None:
                mode_data[metric] = val
        data["modes"].append(mode_data)

    # Channel quality (live monitoring)
    quality = shared_state.get(channel_quality_key())
    if quality:
        data["channel_quality"] = quality

    # Process resource usage
    all_pids = {**service.system_pids, **{f"Mode:{k}": v for k, v in service.mode_pids.items()}}
    for name, pid in all_pids.items():
        try:
            if pid not in process_cache:
                process_cache[pid] = psutil.Process(pid)
            proc = process_cache[pid]
            data["system"]["processes"].append({
                "name": name,
                "pid": pid,
                "cpu_percent": proc.cpu_percent(),
                "memory_mb": proc.memory_info().rss / (1024 * 1024),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_cache.pop(pid, None)

    return data
