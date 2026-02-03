# Dendrite

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

A Python application for brain-computer interface research and development. Handles signal acquisition, real-time processing, and decoder training while external applications drive paradigm design via network events. Three composable processing modes can be configured through a GUI. Runs on Windows, macOS, and Linux.

## Features

- **Three Processing Modes** - Trial-based training (synchronous), continuous inference (asynchronous), and neurofeedback - run individually or combined for hybrid paradigms
- **Hardware-Agnostic** - Connect to any LSL-compatible amplifier; offline stream manager enables development without hardware
- **Multimodal & Multi-Rate** - Synchronized acquisition across EEG, EMG, and other modalities at native sampling rates
- **Research-to-Deployment** - Train with 36+ MOABB datasets, optimize with Optuna, deploy to real-time inference
- **Multiple Output Protocols** - LSL, ROS2, TCP/UDP, ZeroMQ for games, robotics, and distributed systems
- **Integrated Storage** - HDF5 for signals, SQLite for experiment tracking and decoder versioning

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [LSL binaries](https://github.com/sccn/liblsl/releases) (liblsl)

### Installation

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/dendrite-bci/dendrite.git
cd dendrite
uv sync
```

### Launch

```bash
uv run python main.py
```

## Documentation

- [Quickstart Guide](https://dendrite-bci.github.io/dendrite/quickstart) - Installation and setup
- [Introduction](https://dendrite-bci.github.io/dendrite/) - System overview and capabilities
- [User Guides](https://dendrite-bci.github.io/dendrite/user-guides) - Feature tutorials and workflows
- [API Reference](https://dendrite-bci.github.io/dendrite/api) - Technical documentation

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support

- [GitHub Issues](https://github.com/dendrite-bci/dendrite/issues) - Bug reports and feature requests
