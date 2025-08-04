import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
import click
from .cli import cli
from .utils.mlflow_tracking import (
    get_mlflow_client, 
    load_parent_run_id,
    create_child_run,
    terminate_run
)
from mlflow.tracking import MlflowClient
import dvc.api
from .utils.logger import setup_logger
from .utils.constants import LOG_STORAGE_PATH

logger = setup_logger(
    name=__name__,
    log_file=LOG_STORAGE_PATH,
)

class AudioConcatenator:
    def __init__(self, input_folder, output_folder, metadata_path, 
                 client: MlflowClient, run_id: str,
                 target_duration=1800, tolerance=180):
        # Initialize with input/output folders, metadata path, and duration settings
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.metadata_path = Path(metadata_path)
        self.target_duration = target_duration  # Target duration in seconds (default: 30 minutes)
        self.tolerance = tolerance  # Tolerance in seconds (default: ±3 minutes)
        self.min_duration = target_duration - tolerance
        self.max_duration = target_duration + tolerance
        self.output_folder.mkdir(exist_ok=True)  # Create output folder if it doesn't exist
        self.client = client
        self.run_id = run_id

        # Log initialization parameters
        self.client.log_metric(run_id=self.run_id, key="target_duration", value=target_duration)
        self.client.log_metric(run_id=self.run_id, key="tolerance", value=tolerance)

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
            self.client.log_metric(run_id=self.run_id, key="metadata_load_errors", value=1)
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
                self.client.log_metric(run_id=self.run_id, key="missing_files", value=1)
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
        self.client.log_metric(run_id=self.run_id, key="total_segments_created", value=segment_count)

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
            segment_duration = len(combined_audio) / sr
            self.client.log_metric(run_id=self.run_id, key=f"segment_{segment_count}_duration", value=segment_duration)
            self.client.log_artifact(self.run_id, str(output_path))
        
        except Exception as e:
            logger.error(f"Error saving segment {segment_count}: {e}")
            self.client.log_metric(run_id=self.run_id, key="segment_save_errors", value=1)
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
def concatenate_audio(input_folder, output_folder, metadata_path):
    """Concatenate WAV files into segments based on target duration and metadata."""
    # Connect to MLflow client
    client = get_mlflow_client()  

    # Load parent run_id
    parent_run_id = load_parent_run_id()
    
    try:
        # Create child run for this stage
        child_run_id = create_child_run(client, parent_run_id, "concatenating_segments")
        
        # Fetch parameters from DVC
        params = dvc.api.params_show()
        target_duration = params["concat_duration"]
        tolerance = params["concat_tolerance"]
        
        print(f"Target duration: {target_duration} seconds")
        print(f"Tolerance: {tolerance} seconds")
        
        # Initialize concatenator and process files
        concatenator = AudioConcatenator(input_folder, output_folder, metadata_path, 
                                         client, child_run_id,
                                         target_duration, tolerance)
        concatenator.process()
        
        click.echo(f"Concatenation complete. Results saved in {output_folder}")
    
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")
    finally:
        # Terminate child run
        terminate_run(client, child_run_id)