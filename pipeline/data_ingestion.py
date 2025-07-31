import sqlite3
import xml.etree.ElementTree as ET

import click
import mlflow
from .utils.mlflow_tracking import init_mlflow
from .cli import cli


def parse_xml_metadata(xml_file):
    """Parse Angebotsbeschreibung.xml to extract metadata."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        metadata = []
        for record in root.findall(".//record"):
            data = {
                "filename": record.find("FileName").text,
                "sampling_rate": int(record.find("SamplingRate").text),
                "duration": float(record.find("Replay-Time").text),
                "is_outdoor": record.find("Class").text in ["B", "C", "E", "F"],
                "drone_type": record.find("DroneType").text,
                "weight": float(record.find("Weight").text),
                "origin": record.find("Origin").text,
                "file_path": f"data/audio_files/{record.find('FileName').text}"
            }
            metadata.append(data)
        mlflow.log_metric("num_metadata_records", len(metadata))
        return metadata
    except Exception as e:
        mlflow.log_metric("xml_parse_errors", 1)
        raise click.ClickException(f"Failed to parse XML: {e}")


def create_or_update_sqlite_db(metadata, db_path):
    """Create or update SQLite database with metadata."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                sampling_rate INTEGER,
                duration REAL,
                is_outdoor BOOLEAN,
                drone_type TEXT,
                weight REAL,
                origin TEXT,
                file_path TEXT
            )
        """)
        for data in metadata:
            cursor.execute("""
                INSERT OR REPLACE INTO audio_files (filename, sampling_rate, duration, is_outdoor, drone_type, weight, origin, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (data["filename"], data["sampling_rate"], data["duration"], data["is_outdoor"],
                  data["drone_type"], data["weight"], data["origin"], data["file_path"]))
        conn.commit()
        conn.close()
        mlflow.log_artifact(db_path)
    except Exception as e:
        mlflow.log_metric("sqlite_errors", 1)
        raise click.ClickException(f"Failed to update SQLite DB: {e}")


# def download_dataset(url, output_dir):
#     """Download dataset from URL to output directory."""
#     try:
#         os.makedirs(output_dir, exist_ok=True)
#         # Placeholder: Implement actual download logic
#         click.echo(f"Downloading dataset from {url} to {output_dir}")
#         # Example: requests.get(url) to download files
#     except Exception as e:
#         mlflow.log_metric("download_errors", 1)
#         raise click.ClickException(f"Failed to download dataset: {e}")


@cli.command()
@click.option("--metadata_xml", type=click.Path(exists=True), required=True, help="Path to Angebotsbeschreibung.xml")
@click.option("--output", default="data/audio_files", type=click.Path(), help="Output directory for audio files")
@click.option("--db", default="data/metadata.db", type=click.Path(), help="SQLite database path")
def ingest_data(metadata_xml, output, db):
    """Ingest dataset and create/update SQLite database."""
    init_mlflow()
    with mlflow.start_run():
        metadata = parse_xml_metadata(metadata_xml)
        create_or_update_sqlite_db(metadata, db)
        click.echo(f"Successfully ingested dataset to {output} and created/updated {db}")

if __name__ == "__main__":
    ingest_data()