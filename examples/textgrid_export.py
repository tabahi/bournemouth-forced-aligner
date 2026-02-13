
from sqlalchemy import false
from bournemouth_aligner.utils import dict_to_textgrid


# Example usage
if __name__ == "__main__":
    # Sample data (your provided dictionary)
    sample_data = {
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
                "extra_phonemes": [],
                "bad_alignment": False
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
                    "phoneme_id": 29,
                    "phoneme_label": "b",
                    "ipa_label": "b",
                    "start_ms": 33.56833267211914,
                    "end_ms": 50.35249710083008,
                    "confidence": 0.9970603585243225,
                    "is_estimated": False,
                    "target_seq_idx": 0,
                    "index": 0
                },
                {
                    "phoneme_id": 10,
                    "phoneme_label": "ʌ",
                    "ipa_label": "ʌ",
                    "start_ms": 50.35249710083008,
                    "end_ms": 117.48916625976562,
                    "confidence": 0.4553969204425812,
                    "is_estimated": False,
                    "target_seq_idx": 1,
                    "index": 1
                },
                {
                    "phoneme_id": 58,
                    "phoneme_label": "ɾ",
                    "ipa_label": "ɾ",
                    "start_ms": 117.48916625976562,
                    "end_ms": 151.0574951171875,
                    "confidence": 0.039361510425806046,
                    "is_estimated": false,
                    "target_seq_idx": 2,
                    "index": 2
                },
                {
                    "phoneme_id": 9,
                    "phoneme_label": "ɚ",
                    "ipa_label": "ɚ",
                    "start_ms": 151.0574951171875,
                    "end_ms": 302.114990234375,
                    "confidence": 0.18375618755817413,
                    "is_estimated": false,
                    "target_seq_idx": 3,
                    "index": 3
                },
                {
                    "phoneme_id": 43,
                    "phoneme_label": "f",
                    "ipa_label": "f",
                    "start_ms": 302.114990234375,
                    "end_ms": 402.8199768066406,
                    "confidence": 0.952548086643219,
                    "is_estimated": false,
                    "target_seq_idx": 4,
                    "index": 4
                },
                {
                    "phoneme_id": 56,
                    "phoneme_label": "l",
                    "ipa_label": "l",
                    "start_ms": 402.8199768066406,
                    "end_ms": 553.8775024414062,
                    "confidence": 0.9266434907913208,
                    "is_estimated": false,
                    "target_seq_idx": 5,
                    "index": 5
                },
                {
                    "phoneme_id": 23,
                    "phoneme_label": "aɪ",
                    "ipa_label": "aɪ",
                    "start_ms": 553.8775024414062,
                    "end_ms": 939.913330078125,
                    "confidence": 0.11104730516672134,
                    "is_estimated": false,
                    "target_seq_idx": 6,
                    "index": 6
                }
            ],
            "group_ts": [
                {
                    "group_id": 7,
                    "group_label": "voiced_stops",
                    "start_ms": 33.56833267211914,
                    "end_ms": 50.35249710083008,
                    "confidence": 0.9979244470596313,
                    "is_estimated": false,
                    "target_seq_idx": 0,
                    "index": 0
                },
                {
                    "group_id": 2,
                    "group_label": "central_vowels",
                    "start_ms": 50.35249710083008,
                    "end_ms": 117.48916625976562,
                    "confidence": 0.4637269377708435,
                    "is_estimated": false,
                    "target_seq_idx": 1,
                    "index": 1
                },
                {
                    "group_id": 14,
                    "group_label": "rhotics",
                    "start_ms": 117.48916625976562,
                    "end_ms": 151.0574951171875,
                    "confidence": 0.0318431481719017,
                    "is_estimated": false,
                    "target_seq_idx": 2,
                    "index": 2
                },
                {
                    "group_id": 2,
                    "group_label": "central_vowels",
                    "start_ms": 151.0574951171875,
                    "end_ms": 302.114990234375,
                    "confidence": 0.5893039703369141,
                    "is_estimated": false,
                    "target_seq_idx": 3,
                    "index": 3
                },
                {
                    "group_id": 8,
                    "group_label": "voiceless_fricatives",
                    "start_ms": 302.114990234375,
                    "end_ms": 402.8199768066406,
                    "confidence": 0.9883034229278564,
                    "is_estimated": false,
                    "target_seq_idx": 4,
                    "index": 4
                },
                {
                    "group_id": 13,
                    "group_label": "laterals",
                    "start_ms": 402.8199768066406,
                    "end_ms": 553.8775024414062,
                    "confidence": 0.9555268287658691,
                    "is_estimated": false,
                    "target_seq_idx": 5,
                    "index": 5
                },
                {
                    "group_id": 5,
                    "group_label": "diphthongs",
                    "start_ms": 553.8775024414062,
                    "end_ms": 1023.834228515625,
                    "confidence": 0.42801225185394287,
                    "is_estimated": false,
                    "target_seq_idx": 6,
                    "index": 6
                }
            ],
            "words_ts": [
                {
                    "word": "butterfly",
                    "start_ms": 33.56833267211914,
                    "end_ms": 939.913330078125,
                    "confidence": 0.523687694221735,
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
    
    # Basic usage
    textgrid_str = dict_to_textgrid(data=sample_data, output_file=None, include_confidence=False)

    save_path = "examples/samples/outputs/butterfly.TextGrid"

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(textgrid_str)

    print(f"TextGrid saved to: {save_path}")