import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
import click
import mlflow
from .cli import cli
from .utils.mlflow_tracking import init_mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioConcatenator:
    def __init__(self, input_folder, output_folder, metadata_path, target_duration=1800, tolerance=180):
        # Initialize with input/output folders, metadata path, and duration settings
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.metadata_path = Path(metadata_path)
        self.target_duration = target_duration  # Target duration in seconds (default: 30 minutes)
        self.tolerance = tolerance  # Tolerance in seconds (default: ±3 minutes)
        self.min_duration = target_duration - tolerance
        self.max_duration = target_duration + tolerance
        self.output_folder.mkdir(exist_ok=True)  # Create output folder if it doesn't exist
        
        # Log initialization parameters
        mlflow.log_param("input_folder", str(input_folder))
        mlflow.log_param("output_folder", str(output_folder))
        mlflow.log_param("metadata_path", str(metadata_path))
        mlflow.log_param("target_duration", target_duration)
        mlflow.log_param("tolerance", tolerance)

    def load_metadata(self):
        # Load metadata from CSV file
        try:
            df = pd.read_csv(self.metadata_path)
            # Ensure required columns are present
            required_cols = ['filename', 'duration_seconds']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            return df
        except Exception as e:
            logger.error(f"Error loading metadata from {self.metadata_path}: {e}")
            mlflow.log_metric("metadata_load_errors", 1)
            raise

    def concatenate_files(self):
        # Concatenate WAV files into segments
        metadata = self.load_metadata()
        total_files = len(metadata)
        current_segment = []
        current_duration = 0
        segment_count = 1

        for index, row in metadata.iterrows():
            filename = row['filename']
            duration = row['duration_seconds']
            file_path = self.input_folder / filename

            if not file_path.exists():
                logger.warning(f"File {filename} not found, skipping.")
                mlflow.log_metric("missing_files", 1)
                continue

            # Check if file fits in current segment
            if current_duration + duration > self.max_duration and current_duration > 0:
                # Save current segment and start new one
                self.save_segment(current_segment, segment_count)
                current_segment = []
                current_duration = 0
                segment_count += 1

            current_segment.append(file_path)
            current_duration += duration

            # Save segment if minimum duration is reached and no more files remain
            if current_duration >= self.min_duration and index == total_files - 1:
                self.save_segment(current_segment, segment_count)
                break

        # Save final segment if not empty
        if current_segment and current_duration > 0:
            self.save_segment(current_segment, segment_count)
        
        # Log total segments created
        mlflow.log_metric("total_segments_created", segment_count)

    def save_segment(self, file_paths, segment_count):
        # Save concatenated segment to file
        try:
            # Initialize empty array for audio data
            combined_audio = None
            output_path = self.output_folder / f"segment_{segment_count}.wav"

            for file_path in file_paths:
                audio, sr = librosa.load(file_path, sr=None, mono=True)
                if combined_audio is None:
                    combined_audio = audio
                else:
                    combined_audio = np.concatenate((combined_audio, audio))

            # Save concatenated audio
            sf.write(output_path, combined_audio, sr)
            logger.info(f"Saved segment {segment_count}: {output_path} (duration {len(combined_audio) / sr:.2f} sec)")
            
            # Log segment duration and file as artifact
            mlflow.log_metric(f"segment_{segment_count}_duration", len(combined_audio) / sr)
            mlflow.log_artifact(str(output_path))
        
        except Exception as e:
            logger.error(f"Error saving segment {segment_count}: {e}")
            mlflow.log_metric("segment_save_errors", 1)
            raise

    def process(self):
        # Run the concatenation process
        logger.info(f"Starting concatenation of files from {self.input_folder}")
        self.concatenate_files()
        logger.info(f"Concatenation complete. Results saved in {self.output_folder}")

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing WAV files")
@click.option("--output-folder", type=click.Path(), default="test_data/concatenated_wav_segments",
              help="Output directory for concatenated WAV segments")
@click.option("--metadata-path", type=click.Path(exists=True), required=True,
              help="Path to CSV file with metadata")
@click.option("--target-duration", type=int, default=1800,
              help="Target duration for each segment in seconds (default: 1800)")
@click.option("--tolerance", type=int, default=180,
              help="Tolerance for segment duration in seconds (default: 180)")
def concatenate_audio(input_folder, output_folder, metadata_path, target_duration, tolerance):
    """Concatenate WAV files into segments based on target duration and metadata."""
    # Initialize MLflow tracking
    init_mlflow()
    
    # Initialize concatenator and process files
    concatenator = AudioConcatenator(input_folder, output_folder, metadata_path, target_duration, tolerance)
    concatenator.process()
    
    click.echo(f"Concatenation complete. Results saved in {output_folder}")