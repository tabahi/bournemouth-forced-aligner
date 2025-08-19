import whisper
import torch
import os
import json
from bournemouth_aligner import PhonemeTimestampAligner


model = whisper.load_model("turbo")


audio_path = "audio.wav"
srt_path = "whisper_output.srt.json"
ts_out_path = "timestamps.vs2.json"

result = model.transcribe(audio_path)

print("Transcription:")
print(result["text"])

# save whisper output
with open(srt_path, "w") as srt_file:
    json.dump(result, srt_file)

extractor = PhonemeTimestampAligner(model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt", lang='en-us', duration_max=10, device='cpu')
timestamps_dict = extractor.process_srt_file(srt_path, audio_path, ts_out_path, extract_embeddings=False, vspt_path=None, debug=False)

print(timestamps_dict)  

with open(ts_out_path, "w") as ts_file:
    json.dump(timestamps_dict, ts_file)
