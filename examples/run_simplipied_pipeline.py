
import torch
import time

from bournemouth_aligner import PhonemeTimestampAligner

def example_simplified_pipeline():
    """
    Simplified pipeline: text -> phonemize -> CUPE -> Viterbi -> timestamps.
    No group alignment, no embeddings, no target boosting, no silence anchoring.
    Mirrors what a C++ port would do.
    """

    text = "butterfly"
    audio_path = "examples/samples/audio/109867__timkahn__butterfly.wav"

    aligner = PhonemeTimestampAligner(preset="en-us", duration_max=10, device="cpu")

    # 1. Load and preprocess audio
    audio_wav = aligner.load_audio(audio_path)

    # 2. Phonemize text to get phoneme index sequence
    ts_out = aligner.phonemize_sentence(text)
    phoneme_seq = ts_out[aligner.phonemes_key]       # list of int indices
    ipa_labels = ts_out.get("eipa", ts_out.get("ipa", []))

    print(f"Text: '{text}'")
    print(f"IPA: {ipa_labels}")
    print(f"Phoneme indices: {phoneme_seq}")

    # 3. Chop/pad audio segment (full clip in this case)
    wav, wav_len = aligner.chop_wav(audio_wav, 0, audio_wav.shape[1])
    wavs = wav.unsqueeze(0)          # [1, T]
    wav_lens = [wav_len]

    # 4. Run simplified alignment
    t0 = time.time()
    timestamp_dicts = aligner.extract_timestamps_from_segment_simplified(
        wavs, wav_lens,
        phoneme_sequences=[phoneme_seq],
        start_offset_times=[0.0],
        debug=False,
    )
    elapsed_ms = (time.time() - t0) * 1000

    # 5. Print results
    phoneme_ts = timestamp_dicts[0]["phoneme_timestamps"]
    print(f"\nAligned {len(phoneme_ts)} phonemes in {elapsed_ms:.1f} ms:\n")
    for i, tup in enumerate(phoneme_ts):
        ph_id = tup[0]
        start_ms, end_ms = tup[6], tup[7]
        label = aligner.phoneme_id_to_label.get(ph_id, f"UNK_{ph_id}")
        ipa = ipa_labels[i] if i < len(ipa_labels) else "?"
        print(f"  {i+1:>2}: {label:>6s} ({ipa:>3s})  {start_ms:7.1f} - {end_ms:7.1f} ms")


def example_simplified_batch():
    """
    Batch version: multiple sentences aligned in one call.
    """

    aligner = PhonemeTimestampAligner(preset="en-us", duration_max=10, device="cpu")

    texts = [
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "in being comparatively modern.",
    ]
    audio_paths = [
        "examples/samples/LJSpeech/LJ001-0001.wav",
        "examples/samples/LJSpeech/LJ001-0002.wav",
    ]

    # Prepare batch
    all_wavs = []
    all_wav_lens = []
    all_ph_seqs = []
    all_ipa = []
    all_offsets = []

    for text, audio_path in zip(texts, audio_paths):
        audio_wav = aligner.load_audio(audio_path)
        wav, wav_len = aligner.chop_wav(audio_wav, 0, audio_wav.shape[1])
        ts_out = aligner.phonemize_sentence(text)

        all_wavs.append(wav)
        all_wav_lens.append(wav_len)
        all_ph_seqs.append(ts_out[aligner.phonemes_key])
        all_ipa.append(ts_out.get("eipa", ts_out.get("ipa", [])))
        all_offsets.append(0.0)

    wavs = torch.stack(all_wavs, dim=0)

    t0 = time.time()
    timestamp_dicts = aligner.extract_timestamps_from_segment_simplified(
        wavs, all_wav_lens,
        phoneme_sequences=all_ph_seqs,
        start_offset_times=all_offsets,
        debug=False,
    )
    elapsed_ms = (time.time() - t0) * 1000
    print(f"\nBatch alignment took {elapsed_ms:.1f} ms\n")

    for b, (text, td) in enumerate(zip(texts, timestamp_dicts)):
        phoneme_ts = td["phoneme_timestamps"]
        ipa = all_ipa[b]
        print(f"[{b}] '{text[:60]}...' -> {len(phoneme_ts)} phonemes")
        for i, tup in enumerate(phoneme_ts[:5]):
            ph_id = tup[0]
            start_ms, end_ms = tup[6], tup[7]
            label = aligner.phoneme_id_to_label.get(ph_id, f"UNK_{ph_id}")
            ipa_lbl = ipa[i] if i < len(ipa) else "?"
            print(f"    {label:>6s} ({ipa_lbl}): {start_ms:.0f}-{end_ms:.0f} ms")
        if len(phoneme_ts) > 5:
            print(f"    ... ({len(phoneme_ts) - 5} more)")


if __name__ == "__main__":
    torch.random.manual_seed(42)

    print("=" * 60)
    print("Example 1: Single sentence (simplified pipeline)")
    print("=" * 60)
    example_simplified_pipeline()

    print("\n\n")
    print("=" * 60)
    print("Example 2: Batch alignment (simplified pipeline)")
    print("=" * 60)
    example_simplified_batch()
