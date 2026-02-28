
import json
import time

from bournemouth_aligner import PhonemeTimestampAligner
#import sys
#sys.path.append('.')
#from bournemouth_aligner.core import PhonemeTimestampAligner



def example_batch_sentence_processing():
    """Example of batch processing with process_sentence_batch.
    Processes multiple text-audio pairs in a single batch call.
    All audio clips must fit within duration_max seconds."""

    bfa_aligner = PhonemeTimestampAligner(preset='en-us', duration_max=10, device='auto')

    # Load audio files
    texts = [
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "in being comparatively modern.",
    ]
    audio_paths = [
        "examples/samples/LJSpeech/LJ001-0001.wav",
        "examples/samples/LJSpeech/LJ001-0002.wav",
    ]

    audio_wavs = [bfa_aligner.load_audio(p) for p in audio_paths]

    t0 = time.time()
    timestamps = bfa_aligner.process_sentences_batch(texts, audio_wavs, do_groups=True, debug=True)
    print(f"Batch processing took {(time.time() - t0)*1000:.1f} ms")

    # Print results
    for b in range((len(texts))):
        for seg in timestamps[b]["segments"]:
            print(f"\nText: '{seg['text']}'")
            print(f"  Phonemes aligned: {len(seg['phoneme_ts'])}")
            print(f"  Coverage: {seg['coverage_analysis']['coverage_ratio']:.0%}")
            for ph in seg["phoneme_ts"][:5]:
                print(f"    {ph['phoneme_label']} ({ph['ipa_label']}): {ph['start_ms']:.0f}-{ph['end_ms']:.0f} ms, conf={ph['confidence']:.3f}")
            if len(seg["phoneme_ts"]) > 5:
                print(f"    ... ({len(seg['phoneme_ts']) - 5} more)")



    # view detailed debug info
    print(f"Totals,  segments processed: {bfa_aligner.total_segments_processed}", f"\tPhonemes aligned: {bfa_aligner.total_phonemes_aligned}",
            f"\tTarget phonemes: {bfa_aligner.total_phonemes_target}", f"\tPhonemes missed: {bfa_aligner.total_phonemes_missed}",
            f"\tPhonemes extra: {bfa_aligner.total_phonemes_extra}", f"\tPhonemes aligned easily: {bfa_aligner.total_phonemes_aligned_easily}")

def example_batch_SRT_processing():
    """Example of batch processing with process_segments_batch.
    Uses whisper-style SRT json files with pre-segmented audio.
    Each segment gets its own audio tensor (chopped from the full audio)."""

    bfa_aligner = PhonemeTimestampAligner(preset='en-us', duration_max=10, device='cpu', silence_anchors=3)

    srt_paths = [
        "examples/samples/LJSpeech/LJ001-0001.srt.json",
        "examples/samples/LJSpeech/LJ001-0002.srt.json",
    ]
    audio_paths = [
        "examples/samples/LJSpeech/LJ001-0001.wav",
        "examples/samples/LJSpeech/LJ001-0002.wav",
    ]

    # Build flat lists of segments and their corresponding audio tensors
    all_texts_srts = []
    all_audio_wavs = []

    for srt_path, audio_path in zip(srt_paths, audio_paths):
        with open(srt_path, 'r') as f:
            srt_data = json.load(f)
        audio_wav = bfa_aligner.load_audio(audio_path)

        all_texts_srts.append(srt_data)
        all_audio_wavs.append(audio_wav)

    t0 = time.time()
    timestamps = bfa_aligner.process_segments(
        all_texts_srts,
        all_audio_wavs,
        do_groups=True,
        batch_size=16,  # Process 16 segments in a single batch if they are short enough (duration_max=10s), otherwise use batch_size=1 to process segments one by one
        debug=True,
    )
    ts_out_path="examples/samples/LJSpeech/batch_output.vs.json"
    with open(ts_out_path, 'w') as f:
        json.dump(timestamps, f, indent=4)
    
    print(f"Batch segments processing took {(time.time() - t0)*1000:.1f} ms")

    # Print word-level results
    for b in range(len(all_texts_srts)):
        print(f"\nBatch item {b}")
        for seg in timestamps[b]["segments"]:
            print(f"\nText: '{seg['text']}'")
            for w in seg.get("words_ts", []):
                print(f"  {w['word']}: {w['start_ms']:.0f}-{w['end_ms']:.0f} ms, conf={w['confidence']:.3f}")


    

    # view detailed debug info
    print(f"Totals,  segments processed: {bfa_aligner.total_segments_processed}", f"\tPhonemes aligned: {bfa_aligner.total_phonemes_aligned}",
            f"\tTarget phonemes: {bfa_aligner.total_phonemes_target}", f"\tPhonemes missed: {bfa_aligner.total_phonemes_missed}",
            f"\tPhonemes extra: {bfa_aligner.total_phonemes_extra}", f"\tPhonemes aligned easily: {bfa_aligner.total_phonemes_aligned_easily}")
    
    print ("\n\nBatch processing complete. Output saved to:", ts_out_path)

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
