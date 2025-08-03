import os
import torch
import librosa
import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import click
import mlflow
from .cli import cli
from .utils.mlflow_tracking import init_mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='speech_detection.log', filemode='w')
logger = logging.getLogger(__name__)

class SpeechDetector:
    def __init__(self, input_folder, output_folder):
        # Initialize with input/output folders
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
        
        # Log initialization parameters
        mlflow.log_param("input_folder", str(input_folder))
        mlflow.log_param("output_folder", str(output_folder))

    def detect_speech(self, audio_path):
        # Detect speech in an audio file and return probabilities for each chunk
        try:
            logger.info(f"Starting processing for file: {audio_path}")
            # Load audio at 16,000 Hz, mono
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            logger.info(f"Loaded audio: sample rate {sr} Hz, length {len(audio)} samples")
            
            # Log audio duration
            audio_duration = len(audio) / sr
            mlflow.log_metric(f"audio_duration_{audio_path.name}", audio_duration)

            # Normalize audio to avoid clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9

            # Define chunk size for 16,000 Hz
            chunk_size = 512
            num_samples = len(audio)
            num_chunks = (num_samples + chunk_size - 1) // chunk_size  # Ceiling division

            # Pad audio if necessary
            if num_samples % chunk_size != 0:
                padding = chunk_size - (num_samples % chunk_size)
                audio = np.pad(audio, (0, padding), mode='constant')

            # Reshape into [num_chunks, 512]
            audio_chunks = audio.reshape(num_chunks, chunk_size)
            audio_tensor = torch.from_numpy(audio_chunks).float()

            # Process with the model
            probabilities = self.model(audio_tensor, sr)
            probabilities = probabilities.tolist()  # Convert tensor to list
            
            logger.info(f"Processing complete, obtained {len(probabilities)} probabilities")
            mlflow.log_metric("successful_detections", 1)
            return audio_path.name, probabilities

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            mlflow.log_metric("detection_errors", 1)
            return audio_path.name, [str(e)]

    def process(self):
        # Sequentially process all WAV files
        logger.info(f"Starting speech detection for files in {self.input_folder}")
        file_paths = list(self.input_folder.glob('*.wav'))
        total_files = len(file_paths)
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            filename, result = self.detect_speech(file_path)
            output_path = self.output_folder / f"{Path(filename).stem}_speech_probabilities.json"
            if isinstance(result[0], str):  # Handle errors
                logger.error(f"Error processing {filename}: {result[0]}")
                with open(output_path, 'w') as f:
                    json.dump({"error": result}, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Saved probabilities: {output_path} (length: {len(result)})")
                mlflow.log_artifact(str(output_path))
        
        # Log total files processed and success rate
        mlflow.log_metric("total_files_processed", total_files)
        success_rate = (total_files - mlflow.active_run().data.metrics.get("detection_errors", 0)) / total_files if total_files > 0 else 0
        mlflow.log_metric("success_rate", success_rate)
        
        logger.info(f"Speech detection complete. Results saved in {self.output_folder}")

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing WAV segments")
@click.option("--output-folder", type=click.Path(), default="test_data/speech_detection_probabilities",
              help="Output directory for speech probability JSON files")
def detect_speech(input_folder, output_folder):
    """Detect speech in WAV segments and save probabilities as JSON files."""
    # Initialize MLflow tracking
    init_mlflow()
    
    # Initialize detector and process files
    detector = SpeechDetector(input_folder, output_folder)
    detector.process()
    
    click.echo(f"Speech detection complete. Results saved in {output_folder}")