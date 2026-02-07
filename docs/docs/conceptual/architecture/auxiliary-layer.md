---
id: auxiliary-layer
title: Auxiliary Services Layer
sidebar_label: Auxiliary Layer
---

# Auxiliary Services Layer

Tools for development, testing, and offline analysis: Dashboard, Stream Manager, ML Workbench, Database Explorer.

---

## Visualization Dashboard

Dashboard (`src/dendrite/auxiliary/dashboard/app.py`) provides real-time visualization of signal data and mode outputs via LSL stream subscription.

Subscribes to LSL visualization streams via background thread, buffering in bounded queues. Plot managers render synchronized traces at 20 FPS.

Mode-specific views: Synchronous shows ERP stacks and performance metrics. Asynchronous shows prediction traces. Neurofeedback shows band power panels.


---

## Offline Stream Manager

OfflineDataStreamer (`src/dendrite/auxiliary/stream_manager/backend/streamer.py`) launches reproducible LSL streams from files or synthetic generators. Supports .fif and .h5 formats via MNE. Enables development, demos, and testing without hardware.

**File Streaming:** Extracts metadata from MNE-compatible formats and streams sample-by-sample to LSL outlet at original sample rate.

**Synthetic Generation:** Generates test EEG signals with sine wave components plus noise and periodic event markers for testing.

**Process Management:** Each stream runs as independent process with start/stop controls and cross-stream isolation.

**Usage:** Hardware-free development, reproducible testing, demo preparation, session replay, multi-modal testing.

---

## ML Workbench

:::note Work in Progress
The ML Workbench is under active development. Features and interfaces may change.
:::

ML Workbench (`src/dendrite/auxiliary/ml_workbench/app.py`) provides offline model development, validation, and optimization separate from real-time sessions.

Loads datasets from Dendrite HDF5, FIF, and MOABB. Configure epoching and preprocessing, then review epochs before training.

Uses the same Decoder infrastructure as real-time modes for preprocessing parity. Runs k-fold cross-validation, hyperparameter search, and augmentation. Trained decoders are registered in the database with complete lineage.



---

## Database Explorer

Database (`src/dendrite/data/storage/database.py`) implements repository pattern using SQLite (`data/dendrite.db`). Tracks studies, recording sessions, datasets, and trained decoders with complete metadata for experimental reproducibility.

**Schema:** Four tables with foreign key relationships:
- **studies** - Master table for organizing experiments (name, description, config)
- **recordings** - Raw EEG session metadata with BIDS fields (subject, session, run) and file paths
- **datasets** - Imported FIF files for offline training with epoching parameters
- **decoders** - Trained model metadata (architecture, accuracy, channels)

Recordings and decoders cascade-delete with their parent study. Datasets can be unassociated or linked to a study.

**Repository Operations:** Four repository classes (Study, Recording, Dataset, Decoder) provide CRUD operations with parameterized queries for SQL injection protection.

**Usage:** Main Window registers sessions via `add_recording()`, ML Workbench registers models via `add_decoder()`, DB Explorer provides browsing GUI.

---

**Related Documentation:**
- **[Data Layer](data-layer.md)** - Real-time data acquisition and storage
- **[Processing Layer](processing-layer.md)** - Real-time data processing workflows
- **[ML Layer](ml-layer.md)** - Decoder and model architecture
