import os
import base64
import xml.etree.ElementTree as ET
from pathlib import Path
import click
from datetime import datetime
from .utils.mlflow_tracking import (
    get_mlflow_client, 
    load_parent_run_id,
    create_child_run,
    terminate_run
)
from mlflow.tracking import MlflowClient
from .cli import cli


def decode_wav_from_xml(xml_file_path: str, output_wav_path: str, 
                        client: MlflowClient, run_id: str
                        ):
    # Track processing start time
    start_time = datetime.now()
    
    try:
        # Verify read permissions for XML file
        if not os.access(xml_file_path, os.R_OK):
            client.log_metric(run_id=run_id, key="permission_denied_read", value=1)
            raise PermissionError(f"No read permissions for file: {xml_file_path}")

        # Parse XML with namespace handling
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Define XML namespace
        namespaces = {'ns3': 'http://ws.bast.de/container/TrafficDataService'}
        
        # Locate binary element in XML
        binary_element = root.find(".//ns3:binary", namespaces)
        if binary_element is None:
            client.log_metric(run_id=run_id, key="missing_binary_tag", value=1)
            raise ValueError(f"Binary tag not found in file {xml_file_path}")
        
        # Extract and decode Base64 data
        encoded_data = binary_element.text.strip()
        decoded_data = base64.b64decode(encoded_data)
        
        # Log size of decoded data
        client.log_metric(run_id=run_id, key="decoded_data_size_bytes", value=len(decoded_data))
        
        # Verify write permissions for output directory
        output_dir = os.path.dirname(output_wav_path)
        if not os.access(output_dir, os.W_OK):
            client.log_metric(run_id=run_id, key="permission_denied_write", value=1)
            raise PermissionError(f"No write permissions for directory: {output_dir}")
        
        # Save decoded data as WAV file
        with open(output_wav_path, "wb") as wav_file:
            wav_file.write(decoded_data)
        
        # Calculate and log processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        client.log_metric(run_id=run_id, key="file_processing_time_seconds", value=processing_time)
        client.log_metric(run_id=run_id, key="successful_conversions", value=1)
        
        click.echo(f"Successfully saved: {output_wav_path}")
        return True
    
    except PermissionError as e:
        client.log_metric(run_id=run_id, key="permission_errors", value=1)
        click.echo(f"Permission error processing {xml_file_path}: {e}")
        return False
    except base64.binascii.Error:
        client.log_metric(run_id=run_id, key="base64_decode_errors", value=1)
        click.echo(f"Base64 decoding error in file {xml_file_path}")
        return False
    except Exception as e:
        client.log_metric(run_id=run_id, key="general_errors", value=1)
        click.echo(f"General error processing {xml_file_path}: {e}")
        return False

@cli.command()
@click.option("--input-folder", type=click.Path(exists=True), required=True, 
              help="Path to folder containing XML files")
@click.option("--output-folder", type=click.Path(), default="test_data/extracted_wav",
              help="Output directory for WAV files")
def process_xml_files(input_folder, output_folder):
    """Decode WAV files from XML and save them to the output directory."""
    # Connect to MLflow client
    client = get_mlflow_client()  

    # Load parent run_id
    parent_run_id = load_parent_run_id()
    
    try:
        # Create child run for this stage
        child_run_id = create_child_run(client, parent_run_id, "decoding_xml_to_wav")

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
            
            if decode_wav_from_xml(str(xml_file), output_wav_path, client, child_run_id):
                successful_files += 1
        
        # Log summary metrics
        success_rate = successful_files / total_files if total_files > 0 else 0
        client.log_metric(run_id=parent_run_id, key="total_files_processed", value=total_files)
        client.log_metric(run_id=parent_run_id, key="success_rate", value=success_rate)
        
        click.echo(f"Processing complete. {successful_files}/{total_files} files converted successfully.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")
    finally:
        # Terminate child run
        terminate_run(client, child_run_id)

if __name__ == "__main__":
    process_xml_files()