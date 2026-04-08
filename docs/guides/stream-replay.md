# Stream Replay

Replay recorded data as live LSL streams for hardware-free development and testing.

## Opening the Stream Manager

Click the play-circle icon in the top navigation bar to open the Stream Manager panel.

## File Replay

1. Switch to the **File** tab
2. Click **Browse File...** to open the file picker
3. Select a recording file (`.fif`, `.h5`, `.hdf5`)
4. Review the file metadata (duration, sample rate, channels, events)
5. Toggle **Create separate events stream** if you want events on a dedicated LSL outlet (shown when events exist)
6. Click **Start Replay**

## Recording Replay

The **Recordings** tab lists recordings from the database. Select a recording and click **Start Replay** to stream it.

## MOABB Replay

1. Switch to the **MOABB** tab and click **Load Datasets** to discover available datasets
2. Select a dataset and subject
3. Click **Start Stream**

MOABB data downloads automatically on first use.

## Active Streams

Running streams appear at the top of the panel with a source badge (FILE/MOABB), source label, progress bar, and stop button. Multiple streams can run simultaneously. Stream names are prefixed to avoid conflicts with live hardware streams.

Replay streams appear in **Configure Streams** just like real hardware — select and start recording as usual.
