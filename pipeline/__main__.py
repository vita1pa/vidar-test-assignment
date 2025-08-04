from pipeline import (
    cli, 
    decode_xml_to_wav,
    downsample_wav,
    filter_outdoor,
    extract_sound_data_from_wav,
    concatenate_into_segments,
    prepare_recording_positions,
    get_speech_probabilities,
    prepare_speech_labels
)

if __name__ == "__main__":
    cli.cli()