# ML Training

Train decoders for real-time BCI using the ML Workbench.

## Data Sources

The **Data** tab offers three sources:

### MOABB Public Datasets

Browse public BCI datasets from [MOABB](https://moabb.neurotechx.com). Select a dataset, choose a subject, and click to load. Data downloads automatically on first use.

### Internal Datasets

Import your own recordings via the Data Explorer (**Data** page → **Datasets** tab → **Import**). Supports `.fif`, `.h5`, and `.hdf5` files. After importing, datasets appear in the ML Workbench for loading.

### Recordings

Browse recordings in the **Recordings** tab. Select a recording and click **Load** to epoch with the configured preprocessing. Toggle event chips to select classes for training.

## Preprocessing

Before loading, configure:

- **Low/High Cut (Hz)** — Bandpass filter bounds (e.g., 1–40 Hz for EEG, 20–200 Hz for EMG)
- **Common Average Reference** — Subtract mean across channels
- **Epoch Window** — `tmin` and `tmax` define the time window around each event (e.g., -0.2s to 0.8s)

## Event Selection

Toggle event chips to select which classes are used for training.

## Training

Switch to the **Training** tab:

1. **Select a model** — EEGNet, CSP+LDA, or any registered decoder type
2. **Configure hyperparameters** — epochs, batch size, learning rate, validation split, early stopping
3. **Start Training** — runs in background with live per-epoch progress

## Saving Decoders

Click **Save Decoder** on a completed job to save to `data/studies/<study>/decoders/`.

## Output Format

Decoders save as a JSON config + model file pair:
```
decoder_name.json      # Config: model type, input shape, class mapping
decoder_name.pt        # Neural model weights (PyTorch state dict)
decoder_name.joblib    # Classical pipeline (CSP+LDA/SVM) — used instead of .pt
```
