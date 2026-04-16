# ML Training

Train decoders for real-time BCI using the ML Workbench.

## Data Sources

The ML Workbench's left data panel offers two sources, chosen via the **Recordings** / **MOABB** tabs at the top:

### Recordings

Recordings owned by a study — either autosaved from a live Dendrite session or imported as existing `.h5` files. To import, open the Data Explorer (`/data`), select a study, and use **Import folder** to scan a directory: the importer walks the folder recursively and registers every `*.h5` file it finds. Only `.h5` is supported on this path — `.fif`, `.hdf5`, and other formats are not picked up.

In the **Recordings** tab, filter by study, select one or more recordings, and click **Load Data**. With a single recording you get an eval-split slider; with multiple recordings, click a chip to flip any between *train* and *eval*.

### MOABB Public Datasets

Browse public BCI datasets from [MOABB](https://moabb.neurotechx.com). Pick a dataset, choose a subject, and click **Load Dataset** — data downloads on first use. Toggle **Use paradigm preprocessing** to apply the dataset's published bandpass and epoch window.

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
