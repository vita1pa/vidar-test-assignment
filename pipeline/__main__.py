from pipeline import (
    cli, 
    data_ingestion, 
    decode_xml_to_wav,
    downsample_wav,
    filter_outdoor,
    extract_sound_data_from_wav,
    concatenate_into_segments
)

if __name__ == "__main__":
    cli.cli()