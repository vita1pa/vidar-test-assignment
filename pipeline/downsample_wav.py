import os
import librosa
import soundfile as sf
from pathlib import Path
import logging
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
import dvc.api
from .utils.logger import setup_logger
from .utils.constants import LOG_STORAGE_PATH

logger = setup_logger(
    name=__name__,
    log_file=LOG_STORAGE_PATH,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioDownsampler:
    def __init__(self, input_folder, output_folder,
                 client: MlflowClient, run_id: str, 
                 target_sr=16000,):
        # Initialize with input/output folders and target sampling rate
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.target_sr = target_sr
        self.output_folder.mkdir(exist_ok=True)  # Create output folder if it doesn't exist
        self.client = client
        self.run_id = run_id

        # Log target sampling rate
        self.client.log_metric(run_id=self.run_id, key="target_sampling_rate", value=target_sr)

    def downsample_file(self, file_path):
        # Process a single WAV file by downsampling
        start_time = datetime.now()
        try:
            # Load audio file with its original sampling rate
            audio, sr = librosa.load(file_path, sr=None)
            logger.info(f"Processing file {file_path}: original sampling rate {sr} Hz")
            
            # Downsample to target sampling rate
            audio_downsampled = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

            # Generate output file path
            output_path = self.output_folder / f"{file_path.stem}_16kHz.wav"

            # Save downsampled audio
            sf.write(output_path, audio_downsampled, self.target_sr)
            logger.info(f"Saved downsampled file: {output_path}")

            # Calculate processing time and audio duration
            processing_time = (datetime.now() - start_time).total_seconds()
            audio_duration = len(audio) / sr
            self.client.log_metric(run_id=self.run_id, key=f"processing_time_{file_path.name}", value=processing_time)

            return {
                'filename': file_path.name,
                'original_sr': sr,
                'downsampled_sr': self.target_sr,
                'output_path': str(output_path)
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.client.log_metric(run_id=self.run_id, key="downsample_errors", value=1)
            return None

    def process_folder(self):
        # Process all WAV files in the input folder
        results = []
        total_files = 0
        for file_path in self.input_folder.glob('*.wav'):
            total_files += 1
            result = self.downsample_file(file_path)
            if result:
                results.append(result)
        
        # Log total files processed and success rate
        self.client.log_metric(run_id=self.run_id, key="total_files_processed", value=total_files)
        success_rate = len(results) / total_files if total_files > 0 else 0
        self.client.log_metric(run_id=self.run_id, key="success_rate", value=success_rate)
        
        return results

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing WAV files")
@click.option("--output-folder", type=click.Path(), default="test_data/downsampled_wav",
              help="Output directory for downsampled WAV files")
def downsample_audio(input_folder, output_folder):
    """Downsample WAV files in the input folder to the target sampling rate."""
    # Connect to MLflow client
    client = get_mlflow_client()  

    # Load parent run_id
    parent_run_id = load_parent_run_id()

    try:
        # Create child run for this stage
        child_run_id = create_child_run(client, parent_run_id, "downsampling_audio")

        # Fetch target sampling rate from DVC parameters
        params = dvc.api.params_show()
        target_sr = params.get("target_sr")  # Fallback to 16000 if not found
        print(f"Target sampling rate: {target_sr} Hz")
        
        # Initialize downsampler and process files
        downsampler = AudioDownsampler(input_folder, output_folder,
                                       client, child_run_id, target_sr)
        results = downsampler.process_folder()

        # Print results
        for result in results:
            click.echo(f"File: {result['filename']}")
            click.echo(f"  Original sampling rate: {result['original_sr']} Hz")
            click.echo(f"  Downsampled rate: {result['downsampled_sr']} Hz")
            click.echo(f"  Saved to: {result['output_path']}")
        
        click.echo(f"Processing complete. {len(results)} files downsampled successfully.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")
    finally:
        # Terminate child run
        terminate_run(client, child_run_id)

if __name__ == "__main__":
    cli()