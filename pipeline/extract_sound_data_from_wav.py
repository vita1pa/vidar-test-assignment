import os
from pydub import AudioSegment
import mutagen
from pathlib import Path
import csv
import click
from .utils.mlflow_tracking import (
    get_mlflow_client, 
    load_parent_run_id,
    create_child_run,
    terminate_run
)
from mlflow.tracking import MlflowClient
from datetime import datetime
from .cli import cli
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_wav(file_path, client: MlflowClient, run_id: str):
    # Analyze a single WAV file and extract metadata
    start_time = datetime.now()
    try:
        # Load audio file
        audio = AudioSegment.from_wav(file_path)
        # Extract metadata using mutagen
        audio_info = mutagen.File(file_path)

        # Basic audio characteristics
        result = {
            "filename": os.path.basename(file_path),
            "size_bytes": os.path.getsize(file_path),
            "duration_seconds": audio.duration_seconds,
            "sample_rate_hz": audio.frame_rate,
            "channels": audio.channels,
            "bit_depth": audio.sample_width * 8  # Convert to bits
        }

        # Extract RIFF INFO metadata if available
        if audio_info.tags:
            for key, value in audio_info.tags.items():
                result[key] = value
                client.log_metric(run_id=run_id, key=f"metadata_{key}_{result['filename']}", value=str(value))

        # Calculate and log processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        client.log_metric(run_id=run_id, key=f"processing_time_{result['filename']}", value=processing_time)

        return result
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        client.log_metric(run_id=run_id, key="analysis_errors", value=1)
        return {"filename": os.path.basename(file_path), "error": str(e)}

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing WAV files")
@click.option("--output-folder", type=click.Path(), default="wav_analysis",
              help="Output directory for analysis results")
def analyze_wav_files(input_folder, output_folder):
    """Analyze WAV files and save metadata to a CSV file."""
    # Connect to MLflow client
    client = get_mlflow_client()  

    # Load parent run_id
    parent_run_id = load_parent_run_id()

    try:
        # Create child run for this stage
        child_run_id = create_child_run(client, parent_run_id, "filtering_outdoor")
    
        # Create output directory if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Analyze all WAV files in the input folder
        results = []
        total_files = 0
        for wav_file in os.listdir(input_folder):
            if wav_file.endswith(".wav"):
                total_files += 1
                file_path = os.path.join(input_folder, wav_file)
                result = analyze_wav(file_path, client, child_run_id)
                results.append(result)
        
        # Save results to CSV
        output_csv = os.path.join(output_folder, "wav_metadata.csv")
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            # Determine headers from all keys in results
            headers = set()
            for res in results:
                headers.update(res.keys())
            headers = sorted(headers)  # Sort for consistency
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for res in results:
                writer.writerow([res.get(h, "") for h in headers])
        
        # Log CSV as artifact
        client.log_artifact(child_run_id, output_csv)

        # Log summary metrics
        client.log_metric(run_id=child_run_id, key="total_files_processed", value=total_files)
        success_rate = len([r for r in results if "error" not in r]) / total_files if total_files > 0 else 0
        client.log_metric(run_id=child_run_id, key="success_rate", value=success_rate)
    
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")
    finally:
        # Terminate child run
        terminate_run(client, child_run_id)

    click.echo(f"Analysis complete. Results saved to {output_csv}")