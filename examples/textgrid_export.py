
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
    
    # Basic usage
    textgrid_str = dict_to_textgrid(timestamps_dict=sample_data, output_file=None, include_confidence=False)

    save_path = "samples/outputs/butterfly.TextGrid"

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(textgrid_str)