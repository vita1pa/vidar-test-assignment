import os
import base64
import xml.etree.ElementTree as ET
from pathlib import Path
import click
import mlflow
from datetime import datetime
from .utils.mlflow_tracking import init_mlflow
from .cli import cli


def decode_wav_from_xml(xml_file_path, output_wav_path):
    # Track processing start time
    start_time = datetime.now()
    
    try:
        # Verify read permissions for XML file
        if not os.access(xml_file_path, os.R_OK):
            mlflow.log_metric("permission_denied_read", 1)
            raise PermissionError(f"No read permissions for file: {xml_file_path}")

        # Parse XML with namespace handling
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Define XML namespace
        namespaces = {'ns3': 'http://ws.bast.de/container/TrafficDataService'}
        
        # Locate binary element in XML
        binary_element = root.find(".//ns3:binary", namespaces)
        if binary_element is None:
            mlflow.log_metric("missing_binary_tag", 1)
            raise ValueError(f"Binary tag not found in file {xml_file_path}")
        
        # Extract and decode Base64 data
        encoded_data = binary_element.text.strip()
        decoded_data = base64.b64decode(encoded_data)
        
        # Log size of decoded data
        mlflow.log_metric("decoded_data_size_bytes", len(decoded_data))
        
        # Verify write permissions for output directory
        output_dir = os.path.dirname(output_wav_path)
        if not os.access(output_dir, os.W_OK):
            mlflow.log_metric("permission_denied_write", 1)
            raise PermissionError(f"No write permissions for directory: {output_dir}")
        
        # Save decoded data as WAV file
        with open(output_wav_path, "wb") as wav_file:
            wav_file.write(decoded_data)
        
        # Calculate and log processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("file_processing_time_seconds", processing_time)
        mlflow.log_metric("successful_conversions", 1)
        
        click.echo(f"Successfully saved: {output_wav_path}")
        return True
    
    except PermissionError as e:
        mlflow.log_metric("permission_errors", 1)
        click.echo(f"Permission error processing {xml_file_path}: {e}")
        return False
    except base64.binascii.Error:
        mlflow.log_metric("base64_decode_errors", 1)
        click.echo(f"Base64 decoding error in file {xml_file_path}")
        return False
    except Exception as e:
        mlflow.log_metric("general_errors", 1)
        click.echo(f"General error processing {xml_file_path}: {e}")
        return False

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True, 
              help="Path to folder containing XML files")
@click.option("--output-folder", type=click.Path(), default="test_data/extracted_wav",
              help="Output directory for WAV files")
def process_xml_files(input_folder, output_folder):
    # Initialize MLflow tracking
    init_mlflow()    
    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Initialize counters for MLflow metrics
    total_files = 0
    successful_files = 0
    
    # Process all XML files in input folder recursively
    for xml_file in Path(input_folder).glob("**/*.xml"):
        total_files += 1
        output_wav_path = os.path.join(output_folder, f"{xml_file.stem}.wav")
        click.echo(f"Processing file: {xml_file}")
        
        if decode_wav_from_xml(str(xml_file), output_wav_path):
            successful_files += 1
    
    # Log summary metrics
    mlflow.log_metric("total_files_processed", total_files)
    mlflow.log_metric("success_rate", successful_files / total_files if total_files > 0 else 0)
    
    click.echo(f"Processing complete. {successful_files}/{total_files} files converted successfully.")

if __name__ == "__main__":
    process_xml_files()