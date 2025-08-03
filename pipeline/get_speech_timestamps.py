import os
import torch
import librosa
import json
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import logging
import click
import mlflow
from .cli import cli
from .utils.mlflow_tracking import init_mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeechDetector:
    def __init__(self, input_folder, output_folder, sample_rate=16000, threshold=0.5, chunk_duration_seconds=10):
        # Initialize with input/output folders and speech detection parameters
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.chunk_duration_seconds = chunk_duration_seconds
        self.chunk_size_samples = int(sample_rate * chunk_duration_seconds)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Load Silero VAD model and utilities
        self.model, self.utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
        self.get_speech_timestamps = self.utils[0]
        logger.info(f"Loaded function: {self.get_speech_timestamps.__name__}")
        
        # Log initialization parameters
        mlflow.log_param("input_folder", str(input_folder))
        mlflow.log_param("output_folder", str(output_folder))
        mlflow.log_param("sample_rate", sample_rate)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("chunk_duration_seconds", chunk_duration_seconds)

    def detect_speech(self, audio, sr):
        # Detect speech in audio by processing in chunks
        logger.info(f"Original sample rate: {sr} Hz")
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
            logger.info(f"Resampled to {sr} Hz")

        # Normalize audio to avoid clipping
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9

        # Process audio in chunks
        timestamps_all = []
        for i in range(0, len(audio) - self.chunk_size_samples + 1, self.chunk_size_samples):
            chunk = audio[i:i + self.chunk_size_samples]
            if len(chunk) == self.chunk_size_samples:
                audio_tensor = torch.from_numpy(chunk).float()
                try:
                    timestamps = self.get_speech_timestamps(
                        audio_tensor,
                        self.model,
                        self.threshold,
                        sampling_rate=self.sample_rate,
                        return_seconds=True
                    )
                    if timestamps:
                        # Adjust timestamps to absolute time
                        timestamps_adjusted = [
                            {'start': ts['start'] + (i / sr), 'end': ts['end'] + (i / sr)} for ts in timestamps
                        ]
                        timestamps_all.extend(timestamps_adjusted)
                except ValueError as e:
                    logger.error(f"Error processing chunk at {i / sr:.2f} sec: {e}")
                    mlflow.log_metric("chunk_processing_errors", 1)

        # Process remainder if any
        remainder_start = len(audio) - self.chunk_size_samples
        if remainder_start > 0:
            chunk = audio[remainder_start:]
            audio_tensor = torch.from_numpy(chunk).float()
            try:
                timestamps = self.get_speech_timestamps(
                    audio_tensor,
                    self.model,
                    self.threshold,
                    sampling_rate=self.sample_rate,
                    return_seconds=True
                )
                if timestamps:
                    timestamps_adjusted = [
                        {'start': ts['start'] + (remainder_start / sr), 'end': ts['end'] + (remainder_start / sr)} for ts in timestamps
                    ]
                    timestamps_all.extend(timestamps_adjusted)
            except ValueError as e:
                logger.error(f"Error processing remainder at {remainder_start / sr:.2f} sec: {e}")
                mlflow.log_metric("chunk_processing_errors", 1)

        return timestamps_all

    def process_file(self, file_path):
        # Process a single audio file
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            logger.info(f"Processing file {file_path}: sample rate {sr} Hz")
            
            # Log audio duration
            audio_duration = len(audio) / sr
            mlflow.log_metric(f"audio_duration_{file_path.name}", audio_duration)

            speech_segments = self.detect_speech(audio, sr)

            # Convert to list of tuples (start, end)
            speech_segments_list = [(seg['start'], seg['end']) for seg in speech_segments]

            output_path = self.output_folder / f"{file_path.stem}_speech_segments.json"
            with open(output_path, 'w') as f:
                json.dump(speech_segments_list, f, indent=2)
            logger.info(f"Saved speech segments: {output_path} (number of segments: {len(speech_segments_list)})")
            
            # Log segment count and output file
            mlflow.log_metric("speech_segments_detected", len(speech_segments_list))
            mlflow.log_artifact(str(output_path))
            mlflow.log_metric("successful_detections", 1)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            mlflow.log_metric("file_processing_errors", 1)

    def process(self):
        # Process all files in the input folder
        logger.info(f"Starting speech detection for files in {self.input_folder}")
        file_paths = list(self.input_folder.glob('*.wav'))
        total_files = len(file_paths)
        
        for file_path in file_paths:
            self.process_file(file_path)
        
        # Log total files processed and success rate
        mlflow.log_metric("total_files_processed", total_files)
        success_rate = (total_files - mlflow.active_run().data.metrics.get("file_processing_errors", 0)) / total_files if total_files > 0 else 0
        mlflow.log_metric("success_rate", success_rate)
        
        logger.info(f"Speech detection complete. Results saved in {self.output_folder}")

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True,
              help="Path to folder containing WAV segments")
@click.option("--output-folder", type=click.Path(), default="labels/speech_detection",
              help="Output directory for speech segment JSON files")
@click.option("--sample-rate", type=int, default=16000,
              help="Target sample rate for audio files (default: 16000 Hz)")
@click.option("--threshold", type=float, default=0.5,
              help="Probability threshold for speech detection (default: 0.5)")
@click.option("--chunk-duration-seconds", type=float, default=10,
              help="Duration of each audio chunk in seconds (default: 10)")
def detect_speech_segments(input_folder, output_folder, sample_rate, threshold, chunk_duration_seconds):
    """Detect speech segments in WAV files and save timestamps as JSON files."""
    # Initialize MLflow tracking
    init_mlflow()
    
    # Initialize detector and process files
    detector = SpeechDetector(input_folder, output_folder, sample_rate, threshold, chunk_duration_seconds)
    detector.process()
    
    click.echo(f"Speech segment detection complete. Results saved in {output_folder}")