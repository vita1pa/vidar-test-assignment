import os
import json
from pathlib import Path
import logging
from tqdm import tqdm
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='speech_labeling.log', filemode='w')
logger = logging.getLogger(__name__)

class SpeechLabeler:
    def __init__(self, input_folder, output_folder, 
                 client: MlflowClient, run_id: str,
                 threshold=0.5, frame_step=512 / 16000):
        # Initialize with input/output folders, probability threshold, and frame step
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.threshold = threshold
        self.frame_step = frame_step
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.run_id = run_id
        
        # Log parameters
        self.client.log_metric(run_id=self.run_id, key="threshold", value=threshold)
        self.client.log_metric(run_id=self.run_id, key="frame_step", value=frame_step)

    def create_labels(self, probabilities):
        # Create labels based on speech probabilities
        # Extract probabilities from nested lists
        flat_probabilities = [prob[0] for prob in probabilities]
        logger.info(f"Processed {len(flat_probabilities)} flat probabilities")

        labels = []
        start_time = None

        for idx, prob in enumerate(flat_probabilities):
            current_time = idx * self.frame_step
            if prob >= self.threshold and start_time is None:
                start_time = current_time
            elif prob < self.threshold and start_time is not None:
                end_time = current_time
                labels.append((start_time, end_time, "speech"))
                start_time = None

        # Handle speech continuing until the end
        if start_time is not None:
            end_time = len(flat_probabilities) * self.frame_step
            labels.append((start_time, end_time, "speech"))

        # Log number of detected speech segments
        self.client.log_metric(run_id=self.run_id, key="speech_segments_detected", value=len(labels))
        
        return labels

    def process_file(self, json_path):
        # Process a single JSON file and create labels
        try:
            logger.info(f"Starting processing for file: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and "error" in data:
                    logger.error(f"Error in data {json_path}: {data['error']}")
                    self.client.log_metric(run_id=self.run_id, key="json_load_errors", value=1)
                    return
                logger.info(f"Data structure: list with {len(data)} nested elements")

            probabilities = data
            labels = self.create_labels(probabilities)

            output_path = self.output_folder / f"{json_path.stem.replace('_speech_probabilities', '')}_labels.txt"
            with open(output_path, 'w') as f:
                for start, end, label in labels:
                    f.write(f"{start:.3f}\t{end:.3f}\t{label}\n")
            logger.info(f"Saved labels: {output_path} (number of segments: {len(labels)})")
            self.client.log_artifact(self.run_id, str(output_path))
            self.client.log_metric(run_id=self.run_id, key="successful_label_files", value=1)
        
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            self.client.log_metric(run_id=self.run_id, key="label_processing_errors", value=1)

    def process(self):
        # Process all JSON files
        logger.info(f"Starting label creation for files in {self.input_folder}")
        json_files = list(self.input_folder.glob('*.json'))
        total_files = len(json_files)
        
        for json_path in tqdm(json_files, desc="Creating labels"):
            self.process_file(json_path)
        
        # Log total files processed and success rate
        self.client.log_metric(run_id=self.run_id, key="total_files_processed", value=total_files)
        success_rate = (total_files - self.client.get_run(self.run_id).data.metrics.get("label_processing_errors", 0)) / total_files if total_files > 0 else 0
        self.client.log_metric(run_id=self.run_id, key="success_rate", value=success_rate)
        
        logger.info(f"Label creation complete. Results saved in {self.output_folder}")

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing speech probability JSON files")
@click.option("--output-folder", type=click.Path(), default="labels/speech_detection",
              help="Output directory for label files")
def label_speech(input_folder, output_folder):
    """Generate speech labels from probability JSON files."""
    # Connect to MLflow client
    client = get_mlflow_client()  

    # Load parent run_id
    parent_run_id = load_parent_run_id()
    
    try:
        # Create child run for this stage
        child_run_id = create_child_run(client, parent_run_id, "preparing_speech_labels")

        # Fetch parameters from DVC
        params = dvc.api.params_show()
        threshold = params["prob_threshold"]
        frame_step = params["frame_step"]
        
        print(f"Probability threshold: {threshold}")
        print(f"Frame step: {frame_step} seconds")
        
        # Initialize labeler and process files
        labeler = SpeechLabeler(input_folder, output_folder, client, child_run_id, threshold, frame_step)
        labeler.process()
        
        click.echo(f"Label creation complete. Results saved in {output_folder}")

    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")
    finally:
        # Terminate child run
        terminate_run(client, child_run_id)