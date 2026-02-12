import torch
import os
import json
import time

from bournemouth_aligner import PhonemeTimestampAligner



def example_batch_sentence_processing():
    """Example of batch processing with process_sentence_batch.
    Processes multiple text-audio pairs in a single batch call.
    All audio clips must fit within duration_max seconds."""

    aligner = PhonemeTimestampAligner(preset='en-us', duration_max=10, device='cpu')

    # Load audio files
    texts = [
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "in being comparatively modern.",
    ]
    audio_paths = [
        "examples/samples/LJSpeech/LJ001-0001.wav",
        "examples/samples/LJSpeech/LJ001-0002.wav",
    ]

    audio_wavs = [aligner.load_audio(p) for p in audio_paths]

    t0 = time.time()
    vs2_data = aligner.process_sentence_batch(texts, audio_wavs, do_groups=True, debug=True)
    print(f"Batch processing took {(time.time() - t0)*1000:.1f} ms")

    # Print results
    for seg in vs2_data["segments"]:
        print(f"\nText: '{seg['text']}'")
        print(f"  Phonemes aligned: {len(seg['phoneme_ts'])}")
        print(f"  Coverage: {seg['coverage_analysis']['coverage_ratio']:.0%}")
        for ph in seg["phoneme_ts"][:5]:
            print(f"    {ph['phoneme_label']} ({ph['ipa_label']}): {ph['start_ms']:.0f}-{ph['end_ms']:.0f} ms, conf={ph['confidence']:.3f}")
        if len(seg["phoneme_ts"]) > 5:
            print(f"    ... ({len(seg['phoneme_ts']) - 5} more)")


def example_batch_SRT_processing():
    """Example of batch processing with process_segments_batch.
    Uses whisper-style SRT json files with pre-segmented audio.
    Each segment gets its own audio tensor (chopped from the full audio)."""

    aligner = PhonemeTimestampAligner(preset='en-us', duration_max=10, device='cpu', silence_anchors=3)

    srt_paths = [
        "examples/samples/LJSpeech/LJ001-0001.srt.json",
        "examples/samples/LJSpeech/LJ001-0002.srt.json",
    ]
    audio_paths = [
        "examples/samples/LJSpeech/LJ001-0001.wav",
        "examples/samples/LJSpeech/LJ001-0002.wav",
    ]

    # Build flat lists of segments and their corresponding audio tensors
    all_segments = []
    all_audio_wavs = []

    for srt_path, audio_path in zip(srt_paths, audio_paths):
        with open(srt_path, 'r') as f:
            srt_data = json.load(f)
        audio_wav = aligner.load_audio(audio_path)

        for seg in srt_data["segments"]:
            all_segments.append(seg)
            all_audio_wavs.append(audio_wav)

    t0 = time.time()
    vs2_data = aligner.process_segments_batch(
        all_segments,
        all_audio_wavs,
        ts_out_path="examples/samples/LJSpeech/batch_output.vs.json",
        do_groups=True,
        debug=True,
    )
    print(f"Batch segments processing took {(time.time() - t0)*1000:.1f} ms")

    # Print word-level results
    for seg in vs2_data["segments"]:
        print(f"\nText: '{seg['text']}'")
        for w in seg.get("words_ts", []):
            print(f"  {w['word']}: {w['start_ms']:.0f}-{w['end_ms']:.0f} ms, conf={w['confidence']:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Batch sentence processing")
    print("=" * 60)
    example_batch_sentence_processing()

    print("\n\n")
    print("=" * 60)
    print("Example 2: Batch SRT processing (from SRT files)")
    print("=" * 60)
    example_batch_SRT_processing()
