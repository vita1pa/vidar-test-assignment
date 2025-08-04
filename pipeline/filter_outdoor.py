import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import logging
import click
from .utils.mlflow_tracking import (
    get_mlflow_client, 
    load_parent_run_id,
    create_child_run,
    terminate_run
)
from mlflow.tracking import MlflowClient
import mlflow
from datetime import datetime
from .cli import cli
import dvc.api
from .utils.logger import setup_logger
from .utils.constants import LOG_STORAGE_PATH

logger = setup_logger(
    name=__name__,
    log_file=LOG_STORAGE_PATH,
)

class AudioHighPassFilter:
    def __init__(self, input_folder, output_folder, 
                 client: MlflowClient, run_id: str, 
                 cutoff_freq=20, sample_rate=16000):
        # Initialize with input/output folders, cutoff frequency, and sampling rate
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.output_folder.mkdir(exist_ok=True)  # Create output folder if it doesn't exist
        self.client = client
        self.run_id = run_id

        # Log parameters
        self.client.log_metric(run_id=self.run_id, key="cutoff_frequency", value=cutoff_freq)
        self.client.log_metric(run_id=self.run_id, key="sample_rate", value=sample_rate)

    def apply_highpass_filter(self, audio, sr):
        # Apply high-pass filter to audio data
        sos = signal.butter(5, self.cutoff_freq, btype='highpass', fs=sr, output='sos')
        filtered_audio = signal.sosfilt(sos, audio)
        return filtered_audio

    @staticmethod
    def plot_spectrogram(audio, sr, title, output_path):
        # Generate and save spectrogram for the audio
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Spectrogram saved: {output_path}")
        mlflow.log_artifact(str(output_path))

    def analyze_spectrum(self, audio, sr):
        # Analyze frequency spectrum (amplitudes in 0-100 Hz range)
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        low_freq_mask = freqs <= 100
        low_freq_amplitudes = fft[low_freq_mask]
        mean_amplitude = np.mean(low_freq_amplitudes) if low_freq_amplitudes.size > 0 else 0
        return mean_amplitude

    def process_file(self, file_path, plot_spectrogram=False):
        # Process a single WAV file with high-pass filtering
        start_time = datetime.now()
        try:
            # Load audio file with its original sampling rate
            audio, sr = librosa.load(file_path, sr=None)
            logger.info(f"Processing file {file_path}: original sampling rate {sr} Hz")

            # Check and resample if sampling rate differs
            if sr != self.sample_rate:
                logger.warning(f"Sampling rate {sr} Hz differs from expected {self.sample_rate} Hz. Resampling...")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate

            # Analyze spectrum before filtering
            original_low_freq_amplitude = self.analyze_spectrum(audio, sr)

            # Apply high-pass filter
            filtered_audio = self.apply_highpass_filter(audio, sr)

            # Analyze spectrum after filtering
            filtered_low_freq_amplitude = self.analyze_spectrum(filtered_audio, sr)
            logger.info(f"Mean amplitude (0-100 Hz): before={original_low_freq_amplitude:.2f}, after={filtered_low_freq_amplitude:.2f}")

            # Generate output file path
            output_path = self.output_folder / f"{file_path.stem}_filtered.wav"

            # Save filtered audio
            sf.write(output_path, filtered_audio, sr)
            logger.info(f"Saved filtered file: {output_path}")

            # Generate spectrograms if requested
            if plot_spectrogram:
                spectrogram_path = self.output_folder / f"{file_path.stem}_spectrogram.png"
                self.plot_spectrogram(audio, sr, f"Spectrogram: {file_path.name} (Original)", spectrogram_path)
                filtered_spectrogram_path = self.output_folder / f"{file_path.stem}_filtered_spectrogram.png"
                self.plot_spectrogram(filtered_audio, sr, f"Spectrogram: {file_path.name} (Filtered)", filtered_spectrogram_path)

            return {
                'filename': file_path.name,
                'original_sr': sr,
                'cutoff_freq': self.cutoff_freq,
                'output_path': str(output_path),
                'original_low_freq_amplitude': original_low_freq_amplitude,
                'filtered_low_freq_amplitude': filtered_low_freq_amplitude
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.client.log_metric(run_id=self.run_id, key="filter_errors", value=1)
            return None

    def process_folder(self, plot_spectrogram_for_first=False):
        # Process all WAV files in the input folder
        results = []
        first_file = True
        total_files = 0
        for file_path in self.input_folder.glob('*.wav'):
            total_files += 1
            result = self.process_file(file_path, plot_spectrogram=plot_spectrogram_for_first and first_file)
            if result:
                results.append(result)
            if first_file:
                first_file = False
        
        # Log total files processed and success rate
        self.client.log_metric(run_id=self.run_id, key="total_files_processed", value=total_files)
        success_rate = len(results) / total_files if total_files > 0 else 0
        self.client.log_metric(run_id=self.run_id, key="success_rate", value=success_rate)
        
        return results

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing WAV files")
@click.option("--output-folder", type=click.Path(), default="test_data/filtered_wav",
              help="Output directory for filtered WAV files")
@click.option("--plot-spectrogram", is_flag=True, default=False,
              help="Generate spectrograms for the first file")
def highpass_filter(input_folder, output_folder, plot_spectrogram):
    """Apply high-pass filter to WAV files and optionally generate spectrograms."""
    # Connect to MLflow client
    client = get_mlflow_client()  

    # Load parent run_id
    parent_run_id = load_parent_run_id()
    
    try:
        # Create child run for this stage
        child_run_id = create_child_run(client, parent_run_id, "filtering_outdoor")

        # Fetch parameters from DVC
        params = dvc.api.params_show()
        cutoff_freq = params["cutoff_freq"]  # No fallback, assume it's defined in params.yaml
        sample_rate = params["target_sr"]    # No fallback, assume it's defined in params.yaml
        
        print(f"Cutoff frequency: {cutoff_freq} Hz")
        print(f"Sample rate: {sample_rate} Hz")
        
        # Initialize high-pass filter and process files
        highpass_filter = AudioHighPassFilter(input_folder, output_folder, client, child_run_id, 
                                              cutoff_freq, sample_rate)
        results = highpass_filter.process_folder(plot_spectrogram_for_first=plot_spectrogram)

        # Print results
        for result in results:
            click.echo(f"File: {result['filename']}")
            click.echo(f"  Original sampling rate: {result['original_sr']} Hz")
            click.echo(f"  Cutoff frequency: {result['cutoff_freq']} Hz")
            click.echo(f"  Mean amplitude (0-100 Hz): before={result['original_low_freq_amplitude']:.2f}, after={result['filtered_low_freq_amplitude']:.2f}")
            click.echo(f"  Saved to: {result['output_path']}")
        
        click.echo(f"Processing complete. {len(results)} files filtered successfully.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")
    finally:
        # Terminate child run
        terminate_run(client, child_run_id)


if __name__ == "__main__":
    highpass_filter()