import os
import pandas as pd
from pathlib import Path
import logging
import click
import mlflow
from .cli import cli
from .utils.mlflow_tracking import init_mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioLabelGenerator:
    def __init__(self, input_folder, output_folder, metadata_path):
        # Initialize with input/output folders and metadata path
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.metadata_path = Path(metadata_path)
        self.output_folder.mkdir(parents=True, exist_ok=True)  # Create output folder if it doesn't exist
        
        # Log initialization parameters
        mlflow.log_param("input_folder", str(input_folder))
        mlflow.log_param("output_folder", str(output_folder))
        mlflow.log_param("metadata_path", str(metadata_path))

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
        mlflow.log_metric("total_segments_created", segment_count)

    def save_labels(self, labels, segment_count):
        # Save labels to a text file
        try:
            output_path = self.output_folder / f"segment_{segment_count}.txt"
            with open(output_path, 'w') as f:
                for start_time, end_time, filename in labels:
                    f.write(f"{start_time:.3f}\t{end_time:.3f}\t{filename}\n")
            logger.info(f"Saved labels file: {output_path}")
            mlflow.log_artifact(str(output_path))
        except Exception as e:
            logger.error(f"Error saving labels for segment {segment_count}: {e}")
            mlflow.log_metric("label_save_errors", 1)
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
    # Initialize MLflow tracking
    init_mlflow()
    
    # Initialize label generator and process
    label_generator = AudioLabelGenerator(input_folder, output_folder, metadata_path)
    label_generator.process()
    
    click.echo(f"Label generation complete. Results saved in {output_folder}")