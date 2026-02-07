"""
Dendrite Constants

"""

from pathlib import Path

from dendrite import DATA_DIR, __version__
from dendrite.data.stream_schemas import StreamConfig

# --- Version & Identity ---
VERSION = __version__
APP_NAME = "Dendrite"

# --- Mode Types ---
MODE_SYNCHRONOUS = "synchronous"
MODE_ASYNCHRONOUS = "asynchronous"
MODE_NEUROFEEDBACK = "neurofeedback"

# --- Timeouts (seconds) ---
STREAM_RESOLVE_TIMEOUT = 30  # Full LSL stream resolution
LSL_DISCOVERY_TIMEOUT = 2.0  # Quick LSL discovery scan
LSL_INLET_OPEN_TIMEOUT = 2.0  # LSL inlet connection (preflight)
SAMPLE_PULL_TIMEOUT = 0.1  # Per-sample pull timeout
THREAD_JOIN_TIMEOUT = 2.0  # Thread join timeout
TIMEOUT_DATA_ACQUISITION = 5  # Process shutdown
TIMEOUT_DATA_SAVER = 5
TIMEOUT_DATA_PROCESSOR = 3
TIMEOUT_MODE_PROCESS = 2
TIMEOUT_METRICS_SAVER = 2
TIMEOUT_MAIN_PROCESS = 10  # Main processing process shutdown
TIMEOUT_VISUALIZATION = 3  # Visualization streamer shutdown
MODE_GPU_EMIT_INTERVAL = 2.0  # Seconds between GPU telemetry emits
MODE_THREAD_JOIN_TIMEOUT = 2.0  # Mode thread cleanup

# --- Buffer Sizes ---
LSL_BUFFER_SIZE_SECONDS = 1  # LSL inlet buffer duration
DEFAULT_BUFFER_SIZE = 5000  # Dashboard visualization buffer (samples)
QUEUE_SIZE_LARGE = 1000  # Large multiprocessing queue capacity

# --- GUI Timers ---
PID_COLLECTION_INTERVAL_MS = 500  # PID polling interval

# --- Data Processing ---
PLOT_DECIMATION_FACTOR = 5  # 500Hz -> 100Hz for visualization
LOG_INTERVAL_SAMPLES = 30000  # ~60s @ 500Hz between log messages
DROPPED_SAMPLE_WARNING_INTERVAL = 100  # Log every N dropped samples
DEFAULT_CALIBRATION_DURATION = 3.0  # Seconds for baseline calibration
DEFAULT_QUALITY_Z_THRESHOLD = 5.0  # Z-score for bad channel detection

# --- Data Storage ---
STUDIES_DIR = DATA_DIR / "studies"
TEMP_REPORTS_DIR = DATA_DIR / "temp" / "reports"
DATABASE_PATH = DATA_DIR / "dendrite.db"


def get_study_paths(study_name: str) -> dict[str, Path]:
    """Get all data paths for a study."""
    base = STUDIES_DIR / study_name
    return {
        "config": base / "config",
        "raw": base / "raw",
        "metrics": base / "metrics",
        "decoders": base / "decoders",
    }


# --- Data I/O ---
TIMESTAMP_COLS = ["timestamp", "local_timestamp"]
UV_TO_V = 1e-6  # Unit scaling: µV to V for MNE
DEFAULT_MONTAGE = "standard_1005"  # Default MNE montage

# --- LSL ---
LSL_FORMAT_MAP = {
    0: "undefined",
    1: "float32",
    2: "double64",
    3: "string",
    4: "int32",
    5: "int16",
    6: "int8",
    7: "int64",
}

# --- BIDS Defaults ---
BIDS_VERSION = "1.8.0"
DEFAULT_STUDY_NAME = "default_study"
DEFAULT_SUBJECT_ID = "001"
DEFAULT_SESSION_ID = "01"
DEFAULT_RECORDING_NAME = "recording"

# --- Output Streams ---
PREDICTION_STREAM_INFO = StreamConfig(
    name="PredictionStream",
    type="PredictionStream",
    channel_count=1,
    sample_rate=0,
    channel_format="string",
    labels=["Data"],
    source_id="dendrite_prediction_stream",
)
VISUALIZATION_STREAM_INFO = StreamConfig(
    name="Dendrite_Visualization",
    type="Visualization",
    channel_count=1,
    sample_rate=0,
    channel_format="string",
    labels=["Payload"],
    source_id="dendrite_visualization",
)
LATENCY_EVENT_TYPE = "latency_update"

# --- Telemetry ---
STALE_DATA_THRESHOLD_SEC = 3.0  # Data considered stale after this
METRIC_THRESHOLDS = {  # (low, high) - green/yellow/red boundaries
    "internal_ms": (50, 100),
    "inference_ms": (20, 50),
    "e2e_ms": (80, 150),
    "stream_latency_ms": (10, 30),
    "cpu_percent": (50, 80),
    "memory_percent": (50, 80),
    "output_bandwidth_kbps": (50, 200),
}

# --- GUI ---
APPLICATION_TITLE = f"{APP_NAME} GUI"
DEFAULT_WINDOW_SIZE = (1600, 1000)
DEFAULT_WINDOW_POSITION = (100, 100)
