import os
import librosa
import soundfile as sf
from pathlib import Path
import logging
import click
import mlflow
from .utils.mlflow_tracking import init_mlflow
from datetime import datetime
from .cli import cli


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioDownsampler:
    def __init__(self, input_folder, output_folder, target_sr=16000):
        # Initialize with input/output folders and target sampling rate
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.target_sr = target_sr
        self.output_folder.mkdir(exist_ok=True)  # Create output folder if it doesn't exist
        
        # Log initialization parameters
        mlflow.log_param("input_folder", str(input_folder))
        mlflow.log_param("output_folder", str(output_folder))
        mlflow.log_param("target_sampling_rate", target_sr)

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
            mlflow.log_metric(f"processing_time_{file_path.name}", processing_time)

            return {
                'filename': file_path.name,
                'original_sr': sr,
                'downsampled_sr': self.target_sr,
                'output_path': str(output_path)
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            mlflow.log_metric("downsample_errors", 1)
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
        mlflow.log_metric("total_files_processed", total_files)
        success_rate = len(results) / total_files if total_files > 0 else 0
        mlflow.log_metric("success_rate", success_rate)
        
        return results

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing WAV files")
@click.option("--output-folder", type=click.Path(), default="test_data/downsampled_wav",
              help="Output directory for downsampled WAV files")
@click.option("--target-sr", type=int, default=16000,
              help="Target sampling rate for downsampling (in Hz)")
def downsample_audio(input_folder, output_folder, target_sr):
    """Downsample WAV files in the input folder to the target sampling rate."""
    # Initialize MLflow tracking
    init_mlflow()   
    
    # Initialize downsampler and process files
    downsampler = AudioDownsampler(input_folder, output_folder, target_sr)
    results = downsampler.process_folder()

    # Print results
    for result in results:
        click.echo(f"File: {result['filename']}")
        click.echo(f"  Original sampling rate: {result['original_sr']} Hz")
        click.echo(f"  Downsampled rate: {result['downsampled_sr']} Hz")
        click.echo(f"  Saved to: {result['output_path']}")
    
    click.echo(f"Processing complete. {len(results)} files downsampled successfully.")

if __name__ == "__main__":
    downsample_audio()