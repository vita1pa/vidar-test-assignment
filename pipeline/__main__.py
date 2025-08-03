from pipeline import (
    cli, 
    data_ingestion, 
    decode_xml_to_wav,
    downsample_wav,
    filter_outdoor
)

if __name__ == "__main__":
    cli.cli()