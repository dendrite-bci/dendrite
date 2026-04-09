# Dendrite

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

An open-source platform for BCI research and development. Handles signal acquisition, real-time processing, and decoder training via a web interface. External applications drive paradigm design via network events. 

## Features

- **Three Processing Modes** - Trial-based training (synchronous), continuous inference (asynchronous), and neurofeedback - run individually or combined for hybrid paradigms
- **Hardware-Agnostic** - Connect to any LSL-compatible amplifier or custom hardware
- **Multimodal & Multi-Rate** - Synchronized acquisition across EEG, EMG, and other modalities at native sampling rates
- **Multiple Output Protocols** - LSL, ROS2, TCP/UDP, ZeroMQ for games, robotics, and distributed systems
- **Integrated Storage** - HDF5 for signals, SQLite for experiment tracking 
- **Web Architecture** - FastAPI backend + Vue 3 SPA frontend, accessible from any device on the network

## Quick Start

See the [Quickstart guide](https://dendrite-bci.github.io/dendrite/quickstart) for installation and setup.

## Documentation

- [Guides](https://dendrite-bci.github.io/dendrite/guides/)
- [Architecture](https://dendrite-bci.github.io/dendrite/architecture/)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support

- [GitHub Issues](https://github.com/dendrite-bci/dendrite/issues) - Bug reports and feature requests
