
import torch
import time
import json
from bournemouth_aligner import BFA

def example_audio_timestamps():

    text_sentence = "butterfly"
    audio_path = "examples/samples/audio/109867__timkahn__butterfly.wav"

    # Using language preset (recommended) - picks the model for the language. This is the
    # default "std" phoneme set (standard_g2p phonemes, the mbfa-latin model); the first run
    # downloads the model (~55 MB) and the G2P model (~1.2 GB) from Hugging Face, once.
    extractor = BFA(preset="en-us", duration_max=10, device='cpu')

    # Alternative: the espeak-ng + ph66 phoneme set (needs espeak-ng installed; uses the
    # CUPE-3-en model for English and CUPE-3-mls7 for other languages, no G2P download)
    # extractor = BFA(preset="en-us", phoneme_set="ph66", duration_max=10, device='cpu')

    # Alternative: a published model by name - the model decides the phoneme set
    # extractor = BFA(model_name="mbfa-latin-benchmark", duration_max=10, device='cpu')
    # extractor = BFA(model_name="CUPE-3-en", duration_max=10, device='cpu')
    # extractor = BFA(model_name="CUPE-3-mls7", lang="de", duration_max=10, device='cpu')

    # Alternative: a model on disk - a release directory or a training checkpoint file
    # extractor = BFA(ckpt_path="path/to/model", lang='en-us', duration_max=10, device='cpu')

    audio_wav = extractor.load_audio(audio_path) # can replace it with custom audio source

    t0 = time.time()

    timestamps = extractor.process_sentence(text_sentence, audio_wav, extract_embeddings=False, do_groups=True, debug=True)

    t1 = time.time()
    print("Timestamps:")
    print(json.dumps(timestamps, indent=4, ensure_ascii=False))
    print(f"Processing time: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    torch.random.manual_seed(42)
    example_audio_timestamps()















'''output
Alignment Coverage Analysis:
  Target phonemes: 8
  Aligned phonemes: 8
  Coverage ratio: 100.00%

Segment: 'butterfly'
0  Phoneme ID: 22, label: b, IPA: b, Start: 0 ms, End: 52 ms, Confidence: 0.887
1  Phoneme ID: 21, label: ə, IPA: ə, Start: 52 ms, End: 167 ms, Confidence: 0.094
2  Phoneme ID: 9, label: t, IPA: t, Start: 167 ms, End: 257 ms, Confidence: 0.054
3  Phoneme ID: 97, label: ɝ, IPA: ɝ, Start: 257 ms, End: 350 ms, Confidence: 0.014
4  Phoneme ID: 19, label: f, IPA: f, Start: 350 ms, End: 497 ms, Confidence: 0.277
5  Phoneme ID: 43, label: ɫ, IPA: ɫ, Start: 497 ms, End: 572 ms, Confidence: 0.107
6  Phoneme ID: 4, label: a, IPA: a, Start: 572 ms, End: 825 ms, Confidence: 0.943
7  Phoneme ID: 17, label: ɪ, IPA: ɪ, Start: 825 ms, End: 982 ms, Confidence: 0.245

============================================================
PROCESSING SUMMARY
============================================================
Clip items: 1, Segments processed: 1
Totals, Segments processed: 1 	Segments bad: 0 	Phonemes aligned: 8 	Target phonemes: 8 	Phonemes missed: 0
============================================================
Timestamps:
{
    "phoneme_set": "std/latin",
    "lang": "en",
    "segments": [
        {
            "start": 0.0,
            "end": 1.2588125,
            "text": "butterfly",
            "local_ph": [
                22,
                21,
                9,
                97,
                19,
                43,
                4,
                17
            ],
            "gold_phg": [
                5,
                2,
                4,
                1,
                7,
                10,
                3,
                1
            ],
            "coverage_analysis": {
                "target_count": 8,
                "aligned_count": 8,
                "missing_count": 0,
                "extra_count": 0,
                "coverage_ratio": 1.0,
                "missing_phonemes": [],
                "extra_phonemes": [],
                "bad_alignment": false
            },
            "ipa": [
                "b",
                "ə",
                "t",
                "ɝ",
                "f",
                "ɫ",
                "a",
                "ɪ"
            ],
            "word_num": [
                0,
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
                    "phoneme_id": 22,
                    "phoneme_label": "b",
                    "ipa_label": "b",
                    "start_ms": 0.0,
                    "end_ms": 52.5,
                    "confidence": 0.8867923617362976,
                    "target_seq_idx": 0,
                    "index": 0,
                    "tone": 0,
                    "stress": 0,
                    "length": 0
                },
                {
                    "phoneme_id": 21,
                    "phoneme_label": "ə",
                    "ipa_label": "ə",
                    "start_ms": 52.5,
                    "end_ms": 167.5,
                    "confidence": 0.09439657628536224,
                    "target_seq_idx": 1,
                    "index": 1,
                    "tone": 0,
                    "stress": 1,
                    "length": 0
                },
                {
                    "phoneme_id": 9,
                    "phoneme_label": "t",
                    "ipa_label": "t",
                    "start_ms": 167.5,
                    "end_ms": 257.5,
                    "confidence": 0.05372203141450882,
                    "target_seq_idx": 2,
                    "index": 2,
                    "tone": 0,
                    "stress": 0,
                    "length": 0
                },
                {
                    "phoneme_id": 97,
                    "phoneme_label": "ɝ",
                    "ipa_label": "ɝ",
                    "start_ms": 257.5,
                    "end_ms": 350.8782196044922,
                    "confidence": 0.01361100748181343,
                    "target_seq_idx": 3,
                    "index": 3,
                    "tone": 0,
                    "stress": 0,
                    "length": 0
                },
                {
                    "phoneme_id": 19,
                    "phoneme_label": "f",
                    "ipa_label": "f",
                    "start_ms": 350.8782196044922,
                    "end_ms": 497.5,
                    "confidence": 0.2765592336654663,
                    "target_seq_idx": 4,
                    "index": 4,
                    "tone": 0,
                    "stress": 0,
                    "length": 0
                },
                {
                    "phoneme_id": 43,
                    "phoneme_label": "ɫ",
                    "ipa_label": "ɫ",
                    "start_ms": 497.5,
                    "end_ms": 572.5,
                    "confidence": 0.10689622163772583,
                    "target_seq_idx": 5,
                    "index": 5,
                    "tone": 0,
                    "stress": 0,
                    "length": 0
                },
                {
                    "phoneme_id": 4,
                    "phoneme_label": "a",
                    "ipa_label": "a",
                    "start_ms": 572.5,
                    "end_ms": 825.9669494628906,
                    "confidence": 0.943099856376648,
                    "target_seq_idx": 6,
                    "index": 6,
                    "tone": 0,
                    "stress": 2,
                    "length": 0
                },
                {
                    "phoneme_id": 17,
                    "phoneme_label": "ɪ",
                    "ipa_label": "ɪ",
                    "start_ms": 825.9669494628906,
                    "end_ms": 982.5,
                    "confidence": 0.24526582658290863,
                    "target_seq_idx": 7,
                    "index": 7,
                    "tone": 0,
                    "stress": 0,
                    "length": 0
                }
            ],
            "tone": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "stress": [
                0,
                1,
                0,
                0,
                0,
                0,
                2,
                0
            ],
            "length": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "group_ts": [
                {
                    "group_id": 5,
                    "group_label": "stop_voiced",
                    "start_ms": 0.0,
                    "end_ms": 52.5,
                    "confidence": 0.4602576494216919,
                    "target_seq_idx": 0,
                    "index": 0
                },
                {
                    "group_id": 2,
                    "group_label": "vowel_mid",
                    "start_ms": 52.5,
                    "end_ms": 167.5,
                    "confidence": 0.34621405601501465,
                    "target_seq_idx": 1,
                    "index": 1
                },
                {
                    "group_id": 4,
                    "group_label": "stop_voiceless",
                    "start_ms": 167.5,
                    "end_ms": 257.5,
                    "confidence": 0.05913054198026657,
                    "target_seq_idx": 2,
                    "index": 2
                },
                {
                    "group_id": 1,
                    "group_label": "vowel_close",
                    "start_ms": 257.5,
                    "end_ms": 350.8782196044922,
                    "confidence": 0.31947261095046997,
                    "target_seq_idx": 3,
                    "index": 3
                },
                {
                    "group_id": 7,
                    "group_label": "fricative_vl",
                    "start_ms": 350.8782196044922,
                    "end_ms": 497.5,
                    "confidence": 0.862213671207428,
                    "target_seq_idx": 4,
                    "index": 4
                },
                {
                    "group_id": 10,
                    "group_label": "lateral",
                    "start_ms": 497.5,
                    "end_ms": 572.5,
                    "confidence": 0.16395527124404907,
                    "target_seq_idx": 5,
                    "index": 5
                },
                {
                    "group_id": 3,
                    "group_label": "vowel_open",
                    "start_ms": 572.5,
                    "end_ms": 825.9669494628906,
                    "confidence": 0.9435704350471497,
                    "target_seq_idx": 6,
                    "index": 6
                },
                {
                    "group_id": 1,
                    "group_label": "vowel_close",
                    "start_ms": 825.9669494628906,
                    "end_ms": 982.5,
                    "confidence": 0.5733354687690735,
                    "target_seq_idx": 7,
                    "index": 7
                }
            ],
            "words_ts": [
                {
                    "word": "butterfly",
                    "start_ms": 0.0,
                    "end_ms": 982.5,
                    "confidence": 0.32754288939759135,
                    "local_ph": [
                        22,
                        21,
                        9,
                        97,
                        19,
                        43,
                        4,
                        17
                    ],
                    "ipa": [
                        "b",
                        "ə",
                        "t",
                        "ɝ",
                        "f",
                        "ɫ",
                        "a",
                        "ɪ"
                    ]
                }
            ]
        }
    ]
}
Processing time: 3.76 seconds
'''
