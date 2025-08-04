import os
import pandas as pd
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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioLabelGenerator:
    def __init__(self, input_folder, output_folder, metadata_path, client: MlflowClient, run_id: str):
        # Initialize with input/output folders and metadata path
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.metadata_path = Path(metadata_path)
        self.output_folder.mkdir(parents=True, exist_ok=True)  # Create output folder if it doesn't exist
        self.client = client
        self.run_id = run_id

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

    def generate_labels(self):
        # Generate labels for each segment
        metadata = self.load_metadata()
        total_duration = 0
        current_segment_labels = []
        segment_count = 1

        for index, row in metadata.iterrows():
            filename = row['filename']
            duration = row['duration_seconds']
            start_time = total_duration
            end_time = total_duration + duration

            current_segment_labels.append((start_time, end_time, filename))

            total_duration += duration

            # Assume segments are ~30 minutes (1800 sec) with tolerance
            if total_duration >= 1500 and (index == len(metadata) - 1 or total_duration >= 2100):
                self.save_labels(current_segment_labels, segment_count)
                current_segment_labels = []
                total_duration = 0
                segment_count += 1

        # Save final segment if not empty
        if current_segment_labels:
            self.save_labels(current_segment_labels, segment_count)
        
        # Log total segments created
        self.client.log_metric(run_id=self.run_id, key="total_segments_created", value=segment_count)

    def save_labels(self, labels, segment_count):
        # Save labels to a text file
        try:
            output_path = self.output_folder / f"segment_{segment_count}.txt"
            with open(output_path, 'w') as f:
                for start_time, end_time, filename in labels:
                    f.write(f"{start_time:.3f}\t{end_time:.3f}\t{filename}\n")
            logger.info(f"Saved labels file: {output_path}")
            self.client.log_artifact(self.run_id, str(output_path))
        except Exception as e:
            logger.error(f"Error saving labels for segment {segment_count}: {e}")
            self.client.log_metric(run_id=self.run_id, key="label_save_errors", value=1)
            raise

    def process(self):
        # Run the label generation process
        logger.info(f"Starting label generation for segments from {self.input_folder}")
        self.generate_labels()
        logger.info(f"Label generation complete. Results saved in {self.output_folder}")

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing WAV segments")
@click.option("--output-folder", type=click.Path(), default="labels/recording_positions",
              help="Output directory for label files")
@click.option("--metadata-path", type=click.Path(exists=True), required=True,
              help="Path to CSV file with metadata")
def generate_labels(input_folder, output_folder, metadata_path):
    """Generate label files for WAV segments based on metadata."""
    # Connect to MLflow client
    client = get_mlflow_client()  

    # Load parent run_id
    parent_run_id = load_parent_run_id()
    
    try:
        # Create child run for this stage
        child_run_id = create_child_run(client, parent_run_id, "preparing_recording_positions")
    
        # Initialize label generator and process
        label_generator = AudioLabelGenerator(input_folder, output_folder, metadata_path, 
                                              client, child_run_id)
        label_generator.process()
        
        click.echo(f"Label generation complete. Results saved in {output_folder}")
    
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")
    finally:
        # Terminate child run
        terminate_run(client, child_run_id)