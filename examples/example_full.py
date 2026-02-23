import torch
import torchaudio
#from bournemouth_aligner import PhonemeTimestampAligner
import sys
sys.path.append('.')
from bournemouth_aligner.core import PhonemeTimestampAligner



# Step1: Initialize PhonemeTimestampAligner
device = 'cpu' # CPU is faster for single file processing
duration_max = 10 # it's only for padding and clipping. Set it more than your expected duration

# Using language preset (recommended) - automatically selects best English model
extractor = PhonemeTimestampAligner(duration_max=duration_max, device=device) # initialize without loading a model

# Alternative: explicit model selection
# model_name = "en_libri1000_ua01c_e4_val_GER=0.2186.ckpt" # Find more models at: https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt
# lang = 'en-us' # Each CUPE model is trained on a specific language(s)
# extractor = PhonemeTimestampAligner(model_name=model_name, lang=lang, duration_max=duration_max, device=device)

# Step 2a: Load and preprocess audio - manually

audio_path = "examples/samples/audio/Schwa-What.wav"
audio_wav, sr = torchaudio.load(audio_path, normalize=True) #  normalize=True is for torch dtype normalization, not for amplitude

# Stick with the CUPE's sample rate of 16000. For consistency, use the same audio loading and resampling pipeline same as the CUPE's training preprocessing:
resampler = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=16000,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
audio_wav = resampler(audio_wav)

rms = torch.sqrt(torch.mean(audio_wav ** 2)) # rms normalize (better to have at least 75% voiced duration)
audio_wav = (audio_wav / rms) if rms > 0 else audio_wav



# Step 2b: Load and preprocess audio - streamlining
audio_wav =  extractor.load_audio(audio_path)

# Step2: Load/create text transcriptions:

text_sentence = "ah What!"

extractor.load_model(preset="en-us") # load the model before alignment

timestamps = extractor.process_sentence(text_sentence, audio_wav, extract_embeddings=False, do_groups=True, debug=False)
# timestamps is a dict with a "segments" key containing a list of segment dicts

# Step4 (optional): Convert to textgrid
output_file = "output_timestamps.TextGrid"
extractor.convert_to_textgrid(timestamps, output_file=output_file, include_confidence=False)
print(f"TextGrid Timestamps saved to {output_file}")