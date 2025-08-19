import torch
import torchaudio
from bournemouth_aligner import PhonemeTimestampAligner



# Step1: Initialize PhonemeTimestampAligner
device = 'cpu' # CPU is faster for sigle file processing
duration_max = 10 # it's only for padding and clipping. Set it more than your expected duration
model_name = "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt" # Find more models at: https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt
lang = 'en-us' # Each CUPE model is trained on a specific language(s)
extractor = PhonemeTimestampAligner(model_name=model_name, lang=lang, duration_max=duration_max, device='cpu')

# Step 2a: Load and preprocess audio - manually

audio_path = "examples/samples/audio/Schwa-What.wav"
audio_wav, sr = torchaudio.load(audio_path, normalize=True) #  normalize=True is for torch dtype normalization, not for amplitude

# Stick with the CUPE's sample rate of 16000. For consistency, use the same audio loading and resampling pipeline same as the CUPE's training preprocessing:
resampler = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=160000,
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

transcription = "ah What!"



timestamps = extractor.process_transcription(transcription, audio_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True, debug=False)


# Step4 (optional): Convert to textgrid
extractor.convert_to_textgrid(timestamps, output_file="output_timestamps.TextGrid", include_confidence=False)