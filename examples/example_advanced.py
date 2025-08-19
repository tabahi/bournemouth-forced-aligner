

import torch
import os

from bournemouth_aligner import PhonemeTimestampAligner


def example_audio_file_processing():
    """Example of single file processing usage.
    Use whisper style .json files for loading transcription.
    For batch processing, it's better to create a single Instance of PhonemeTimestampAligner, and use its `process_srt_file` method for each file. 
    Using this method is not recommended for batch processing since it will create a new instance of the extractor for each file, which is inefficient."""

    cupe_ckpt_path="models/m_uj01d_epoch=199_step=1035200_val_GER=0.2307.ckpt"

    # This would typically come from your data collator
    example_paths = [
        {
            "audio": "examples/samples/LJSpeech/LJ001-0001.wav",
            "srt": "examples/samples/LJSpeech/LJ001-0001.srt.json", 
            "vs": "examples/samples/LJSpeech/LJ001-0001.vs.json", # timestamps json output
            "txg": "examples/samples/LJSpeech/LJ001-0001.TextGrid", # textgrid output for Praat
        },
        {
            "audio": "examples/samples/LJSpeech/LJ001-0002.wav",
            "srt": "examples/samples/LJSpeech/LJ001-0002.srt.json", 
            "vs": "examples/samples/LJSpeech/LJ001-0002.vs.json", # timestamps json output
            "txg": "examples/samples/LJSpeech/LJ001-0002.TextGrid" # textgrid output for Praat
        },
        # ... more paths
    ]


    extractor = PhonemeTimestampAligner(cupe_ckpt_path=cupe_ckpt_path, lang='en-us', duration_max=10, device='cpu')



    for paths in example_paths:
        audio_path = paths["audio"]
        srt_path = paths["srt"]
        ts_out_path = paths["vs"]



        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file does not exist: {srt_path}")

            
        timestamps_dict = extractor.process_srt_file(srt_path, audio_path, ts_out_path, extract_embeddings=False, vspt_path=None, debug=False)
    
        if timestamps_dict and paths.get("txg"):
            extractor.convert_to_textgrid(timestamps_dict, paths["txg"], include_confidence=False)




if __name__ == "__main__":
    torch.random.manual_seed(42)
    example_audio_file_processing()

