---
id: api-overview
title: API Reference
sidebar_label: API Overview
slug: /api/
---

# API Reference

Programmatic interfaces for extending and integrating with Dendrite. These APIs are for developers building custom components, task applications, or offline analysis workflows.

**→ [Architecture Overview](../conceptual/)** for system design and conceptual documentation.

## Public API Modules

### dendrite.data
Event broadcasting and data handling:
- **[EventOutlet](generated/dendrite/data/event_outlet.md)** - LSL event stream for task applications

### dendrite.ml.decoders
Decoder factory functions for model training and inference:
- **[decoders](generated/dendrite/ml/decoders/)** - Factory functions (`create_decoder`, `load_decoder`, etc.)
- **[Decoder](generated/dendrite/ml/decoders/decoder.md)** - Main decoder class with fit/predict interface
- **[DecoderConfig](generated/dendrite/ml/decoders/decoder_schemas.md)** - Decoder configuration schema
- **[Registry](generated/dendrite/ml/decoders/registry.md)** - Available decoder types

### dendrite.ml.models
Model factory and base class:
- **[models](generated/dendrite/ml/models/)** - Factory functions (`create_model`, `get_available_models`)
- **[ModelBase](generated/dendrite/ml/models/base.md)** - Base class for extending with custom models

