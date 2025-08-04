# Vidar Speech Processing Pipeline

This repository contains a Minimum Viable Product (MVP) for a speech processing pipeline developed to detect and label speech segments from audio files. The pipeline processes XML files through multiple stages, generating speech probabilities and Audacity-compatible labels. Below, you'll find an overview of what has been accomplished, how to run the pipeline, and ideas for future enhancements.

## What I Have Done

**Stage 1: XML to WAV Conversion**  
Converted XML files containing audio metadata into WAV audio segments.

**Stage 2: Speech Probability Detection**  
Utilized the Silero VAD model to compute speech probabilities for each audio segment, saving results as JSON files.

**Stage 3: Speech Label Generation**  
Transformed speech probabilities into Audacity-compatible label files (.txt) with segments filtered by a probability threshold.

The pipeline is configured to process files sequentially, with outputs saved in the `labels` directory (e.g., `labels/recording_positions` and `labels/speech_detection`).

## How to Run the Pipeline

### Prerequisites

- Install Python 3.10+.
- Install required dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

### Setup

1. Clone this repository.  
2. Create a `data` directory and place your XML input folder inside `data/xml_input`  
   (e.g., `data/xml_input/AuDroKSoundData-23-02-22 Measurements_1-19`).  
3. Ensure the `labels` directory exists or will be created automatically.

### Configuration

Parameters (e.g., `threshold`, `frame_step`) are configurable via `params.yaml`.  
Edit this file to adjust settings as needed.

### Run the Pipeline

- Initialize DVC:  
  ```bash
  dvc init
  ```

- Reproduce the entire pipeline:  
  ```bash
  dvc repro
  ```

- Reproduce a specific stage:  
  ```bash
  dvc repro <concrete_stage>  # e.g., dvc repro getting_speech_probabilities
  ```

- Outputs will be saved in the `labels` directory.

---

## Future Enhancements (WIP)

Here are ideas for improving the pipeline, which are currently work-in-progress:

- **Local ONNX Weight Loading**:  
  Implement loading of ONNX weights locally, potentially using DVC for version control.

- **DVC Advanced 1**:  
  Enhance DVC to add new subfolders in `/xml_input` and `/wav_input`, ensuring only new files are processed while skipping previously transformed ones.

- **DVC Advanced 2**:  
  Integrate two DVC pipelines into one, with shared yet distinct stages (e.g., skipping `decoding_xml_to_wav` for WAV files) and merging XML and WAV processing at a common stage like `preparing_recording_positions`.

### Pytest Quality Tests (WIP)

- Verify that the output sampling rate is 16 kHz.  
- Check for removal of frequencies below 20 Hz, generating a spectrogram if violations are detected.  
- Validate the `preparing_speech_labels` stage with ground truth labels (requires dataset).  
- Ensure Audacity label format compliance.

### Multiprocessing

Add multiprocessing for stages with independent processing  
(e.g., `getting_speech_probabilities` across segments).  
Current chunk-based parallelization may be suboptimal due to potential temporal dependencies (e.g., RNN model behavior).

---

## MLflow Integration

The pipeline uses MLflow to track metrics (primarily errors) and artifacts.  
All DVC stages are executed within a single MLflow session, with each stage running as a child run under a parent run initialized in the `initializing_mlflow` stage.

**Viewing Metrics**:  
Run the following command to launch the MLflow UI dashboard and monitor metrics and artifacts:  
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlruns.db
```

---

## DVC Remote Storage

The pipeline is configured with a local DVC remote (e.g., tied to Google Drive in this setup).  
To set up your own remote storage:

1. Install DVC and configure a remote storage provider (e.g., Google Drive, AWS S3).
2. For Google Drive, install `dvc[gdrive]` and authenticate:

```bash
dvc remote add --default myremote gdrive://<your_folder_id>
dvc remote modify myremote gdrive_use_service_account true
```

3. Provide a service account key (optional for personal use, follow Google Drive API setup).  
4. Store credentials securely in `~/.dvc/config` or a local `config.local` file (not shared).

Adjust the remote in `dvc remote modify` if needed.  
If no remote is configured, `dvc pull/push` will be unavailable until set up.

> **Note**: The current setup uses a local GDrive remote (`credentials in config.local`, not shared).  
> Users must configure their own remote storage to enable `dvc pull/push`.

---

## Configurable Pipeline

The pipeline is driven by a `params.yaml` file, allowing users to tweak parameters  
(e.g., `threshold`, `min_duration`) to suit their needs.
