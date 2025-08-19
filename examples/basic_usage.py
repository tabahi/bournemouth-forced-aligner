
import torch
import time
import json
from bournemouth_aligner import PhonemeTimestampAligner

def example_audio_timestamps():

    transcription = "butterfly"
    audio_path = "examples/samples/audio/109867__timkahn__butterfly.wav"
    
    model_name = "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt" 
    extractor = PhonemeTimestampAligner(model_name=model_name, lang='en-us', duration_max=10, device='cpu')

    audio_wav = extractor.load_audio(audio_path) # can replace it with custom audio source

    t0 = time.time()

    timestamps = extractor.process_transcription(transcription, audio_wav, ts_out_path=None, extract_embeddings=False, vspt_path=None, do_groups=True, debug=True)

    t1 = time.time()
    print("Timestamps:")
    print(json.dumps(timestamps, indent=4, ensure_ascii=False))
    print(f"Processing time: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    torch.random.manual_seed(42)
    example_audio_timestamps()















'''output
Model available at: /root/.cache/huggingface/hub/models--Tabahi--CUPE-2i/snapshots/5bb0124be864e01d12c90145863f727e490ab3fb/ckpt/en_libri1000_uj01d_e199_val_GER=0.2307.ckpt
Setting backend for language: en-us
Expected phonemes: ['b', 'ʌ', 'ɾ', 'ɚ', 'f', 'l', 'aɪ']
Target phonemes: 7, Expected: ['b', 'ʌ', 'ɾ', 'ɚ', 'f', 'l', 'aɪ']
Spectral length: 75
Forced alignment took 18.673 ms
Aligned phonemes: 7
Target phonemes: 7
SUCCESS: All target phonemes were aligned!
Predicted phonemes 7
Predicted groups 7
start_offset_time 0.0
 1:   b, voiced_stops  -> (33.568 - 50.352), Confidence: 0.991
 2:   ʌ, central_vowels  -> (100.705 - 117.489), Confidence: 0.845
 3:   ɾ, rhotics  -> (134.273 - 151.057), Confidence: 0.285
 4:   ɚ, central_vowels  -> (285.331 - 302.115), Confidence: 0.738
 5:   f, voiceless_fricatives  -> (352.467 - 402.820), Confidence: 0.988
 6:   l, laterals  -> (520.309 - 553.878), Confidence: 0.916
 7:  aɪ, diphthongs  -> (604.230 - 621.014), Confidence: 0.412
Alignment Coverage Analysis:
  Target phonemes: 7
  Aligned phonemes: 7
  Coverage ratio: 100.00%

============================================================
PROCESSING SUMMARY
============================================================
Total segments processed: 1
Perfect sequence matches: 1/1 (100.0%)
Total phonemes aligned: 7
Overall average confidence: 0.655
============================================================
Timestamps:
{
    "segments": [
        {
            "start": 0.0,
            "end": 1.2588125,
            "text": "butterfly",
            "ph66": [
                29,
                10,
                58,
                9,
                43,
                56,
                23
            ],
            "pg16": [
                7,
                2,
                14,
                2,
                8,
                13,
                5
            ],
            "coverage_analysis": {
                "target_count": 7,
                "aligned_count": 7,
                "missing_count": 0,
                "extra_count": 0,
                "coverage_ratio": 1.0,
                "missing_phonemes": [],
                "extra_phonemes": []
            },
            "ipa": [
                "b",
                "ʌ",
                "ɾ",
                "ɚ",
                "f",
                "l",
                "aɪ"
            ],
            "word_num": [
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "words": [
                "butterfly"
            ],
            "phoneme_ts": [
                {
                    "phoneme_idx": 29,
                    "phoneme_label": "b",
                    "start_ms": 33.56833267211914,
                    "end_ms": 50.35249710083008,
                    "confidence": 0.9849503040313721
                },
                {
                    "phoneme_idx": 10,
                    "phoneme_label": "ʌ",
                    "start_ms": 100.70499420166016,
                    "end_ms": 117.48916625976562,
                    "confidence": 0.8435571193695068
                },
                {
                    "phoneme_idx": 58,
                    "phoneme_label": "ɾ",
                    "start_ms": 134.27333068847656,
                    "end_ms": 151.0574951171875,
                    "confidence": 0.3894280791282654
                },
                {
                    "phoneme_idx": 9,
                    "phoneme_label": "ɚ",
                    "start_ms": 285.3308410644531,
                    "end_ms": 302.114990234375,
                    "confidence": 0.3299962282180786
                },
                {
                    "phoneme_idx": 43,
                    "phoneme_label": "f",
                    "start_ms": 369.2516784667969,
                    "end_ms": 386.03582763671875,
                    "confidence": 0.9150863289833069
                },
                {
                    "phoneme_idx": 56,
                    "phoneme_label": "l",
                    "start_ms": 520.3091430664062,
                    "end_ms": 553.8775024414062,
                    "confidence": 0.9060741662979126
                },
                {
                    "phoneme_idx": 23,
                    "phoneme_label": "aɪ",
                    "start_ms": 604.22998046875,
                    "end_ms": 621.01416015625,
                    "confidence": 0.21650740504264832
                }
            ],
            "group_ts": [
                {
                    "group_idx": 7,
                    "group_label": "voiced_stops",
                    "start_ms": 33.56833267211914,
                    "end_ms": 50.35249710083008,
                    "confidence": 0.9911064505577087
                },
                {
                    "group_idx": 2,
                    "group_label": "central_vowels",
                    "start_ms": 100.70499420166016,
                    "end_ms": 117.48916625976562,
                    "confidence": 0.8446590304374695
                },
                {
                    "group_idx": 14,
                    "group_label": "rhotics",
                    "start_ms": 134.27333068847656,
                    "end_ms": 151.0574951171875,
                    "confidence": 0.28526052832603455
                },
                {
                    "group_idx": 2,
                    "group_label": "central_vowels",
                    "start_ms": 285.3308410644531,
                    "end_ms": 302.114990234375,
                    "confidence": 0.7377423048019409
                },
                {
                    "group_idx": 8,
                    "group_label": "voiceless_fricatives",
                    "start_ms": 352.4674987792969,
                    "end_ms": 402.8199768066406,
                    "confidence": 0.9877637028694153
                },
                {
                    "group_idx": 13,
                    "group_label": "laterals",
                    "start_ms": 520.3091430664062,
                    "end_ms": 553.8775024414062,
                    "confidence": 0.9163824915885925
                },
                {
                    "group_idx": 5,
                    "group_label": "diphthongs",
                    "start_ms": 604.22998046875,
                    "end_ms": 621.01416015625,
                    "confidence": 0.4117060899734497
                }
            ],
            "words_ts": [
                {
                    "word": "butterfly",
                    "start_ms": 33.56833267211914,
                    "end_ms": 621.01416015625,
                    "confidence": 0.6550856615815844,
                    "ph66": [
                        29,
                        10,
                        58,
                        9,
                        43,
                        56,
                        23
                    ],
                    "ipa": [
                        "b",
                        "ʌ",
                        "ɾ",
                        "ɚ",
                        "f",
                        "l",
                        "aɪ"
                    ]
                }
            ]
        }
    ]
}
Processing time: 0.19 seconds
'''