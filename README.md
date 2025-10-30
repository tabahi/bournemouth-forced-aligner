# 🎯 Bournemouth Forced Aligner (BFA)

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/bournemouth-forced-aligner.svg)](https://badge.fury.io/py/bournemouth-forced-aligner)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub stars](https://img.shields.io/github/stars/tabahi/bournemouth-forced-aligner.svg)](https://github.com/tabahi/bournemouth-forced-aligner/stargazers)

**High-precision multi-lingual phoneme-level timestamp extraction from audio files**
> 🎯 **Find the exact time when any phoneme is spoken** - provided you have the audio and its text.


[🚀 Quick Start](#-getting-started) • [📚 Documentation](#-how-does-it-work) • [🔧 Installation](#-installation) • [💻 CLI](#-command-line-interface-cli) • [🤝 Contributing](https://github.com/tabahi/bournemouth-forced-aligner/issues)

</div>

---

## ✨ Overview

BFA is a lightning-fast Python library that extracts **phoneme/word timestamps** from audio files with millisecond precision. Built on  [Contextless Universal Phoneme Encoder (CUPE)](https://github.com/tabahi/contexless-phonemes-CUPE), it delivers accurate forced alignment for speech analysis, linguistics research, and audio processing applications.


## 🌟 Key Features

<div align="center">

| Feature | Description | Performance |
|---------|-------------|-------------|
| ⚡ **Ultra-Fast** | CPU-optimized processing | 0.2s for 10s audio |
| 🎯 **Phoneme-Level** | Millisecond-precision timestamps | High accuracy alignment |
| 🌍 **Multi-Language** | Via espeak phonemization | 80+ Indo-European + related |
| 🔧 **Easy Integration** | JSON & TextGrid output | Praat compatibility |



</div>


**Words+Phonemes aligned to Mel-spectrum frames:**

![Aligned mel-spectrum plot](examples/samples/images/LJ02_mel_words.png)
![Aligned mel-spectrum plot](examples/samples/images/LJ01_mel_phonemes.png)

Try [mel_spectrum_alignment.py](examples/mel_spectrum_alignment.py)


---
## 🚀 Installation

### 📦 From PyPI (Recommended)

```bash
# Install the package
pip install bournemouth-forced-aligner

# Alternatively, install the latest library directly from github:
# pip install git+https://github.com/tabahi/bournemouth-forced-aligner.git

# Install system dependencies
apt-get install espeak-ng ffmpeg
```

### ✅ Verify Installation

```bash
# Show help
balign --help

# Check version
balign --version

# Test installation
python -c "from bournemouth_aligner import PhonemeTimestampAligner; print('✅ Installation successful!')"
```

---

## 🎯 Getting Started

### 🔥 Quick Example

```python
import torch
import time
import json
from bournemouth_aligner import PhonemeTimestampAligner

# Configuration
text_sentence = "butterfly"
audio_path = "examples/samples/audio/109867__timkahn__butterfly.wav"

# Initialize aligner using language preset (recommended)
extractor = PhonemeTimestampAligner(
    preset="en-us",  # Automatically selects best English model
    duration_max=10,
    device='cpu'
)

# Alternative: explicit model selection
# extractor = PhonemeTimestampAligner(
#     model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt",
#     lang='en-us',
#     duration_max=10,
#     device='cpu'
# )

# Load and process
audio_wav = extractor.load_audio(audio_path) # use RMS normalization for preloaded wav `audio_wav = extractor._rms_normalize(audio_wav)`

t0 = time.time()
timestamps = extractor.process_sentence(
    text_sentence,
    audio_wav,
    ts_out_path=None,
    extract_embeddings=False,
    vspt_path=None,
    do_groups=True,
    debug=True
)
t1 = time.time()

print("🎯 Timestamps:")
print(json.dumps(timestamps, indent=4, ensure_ascii=False))
print(f"⚡ Processing time: {t1 - t0:.2f} seconds")
```



### 🌐 Multi-Language Examples

```python
# German with MLS8 model
aligner_de = PhonemeTimestampAligner(preset="de")

# Hindi with Universal model
aligner_hi = PhonemeTimestampAligner(preset="hi")

# French with MLS8 model
aligner_fr = PhonemeTimestampAligner(preset="fr")

```

### 📊 Sample Output

<details>
<summary>📋 Click to see detailed JSON output</summary>

```json
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
            ],
        }
    ]
}
```

</details>

### 🔑 Output Format Guide

| Key | Description | Format |
|-----|-------------|--------|
| `ph66` | Standardized 66 phoneme classes (including silence) | See [ph66_mapper.py](bournemouth_aligner/ipamappers/ph66_mapper.py) |
| `pg16` | 16 phoneme category groups (lateral, vowels, rhotics, etc.) | Grouped classifications |
| `ipa` | IPA sequences from espeak | Unicode IPA symbols |
| `words` | Word segmentation | Regex-based: `\b\w+\b` |
| `phoneme_ts` | Aligned phoneme timestamps | Millisecond precision |
| `group_ts` | Phoneme group timestamps | Often more accurate |
| `word_num` | Word index for each phoneme | Maps phonemes to words |
| `words_ts` | Word-level timestamps | Derived from phonemes |
| `coverage_analysis` | Alignment quality metrics | Insertions/deletions |

---

## 🛠️ Methods

### 🌍 Language Presets

BFA supports **80+ languages** through intelligent preset selection, focusing on Indo-European and closely related language families. Simply specify a language code as `preset` parameter for automatic model and language configuration.

**⚠️ Note**: Tonal languages (Chinese, Vietnamese, Thai) and distant language families (Japanese, Korean, Bantu, etc.) are not supported through presets due to CUPE model limitations.

```python
# Using presets (recommended)
aligner = PhonemeTimestampAligner(preset="de")  # German with MLS8 model
aligner = PhonemeTimestampAligner(preset="hi")  # Hindi with Universal model
aligner = PhonemeTimestampAligner(preset="fr")  # French with MLS8 model
```

#### 🎯 Parameter Priority
1. **Explicit `cupe_ckpt_path`** (highest priority)
2. **Explicit `model_name`**
3. **Preset values** (only if no explicit model specified)
4. **Default values**

#### 📋 Complete Preset Table

<details>
<summary>🔍 Click to view all 80+ supported language presets</summary>

| **Language** | **Preset Code** | **Model Used** | **Language Family** |
|--------------|-----------------|----------------|-------------------|
| **🇺🇸 ENGLISH VARIANTS** | | |
| English (US) | `en-us`, `en` | English Model | West Germanic |
| English (UK) | `en-gb` | English Model | West Germanic |
| English (Caribbean) | `en-029` | English Model | West Germanic |
| English (Lancastrian) | `en-gb-x-gbclan` | English Model | West Germanic |
| English (RP) | `en-gb-x-rp` | English Model | West Germanic |
| English (Scottish) | `en-gb-scotland` | English Model | West Germanic |
| English (West Midlands) | `en-gb-x-gbcwmd` | English Model | West Germanic |
| **🇪🇺 EUROPEAN LANGUAGES (MLS8)** | | |
| German | `de` | MLS8 Model | West Germanic |
| French | `fr` | MLS8 Model | Romance |
| French (Belgium) | `fr-be` | MLS8 Model | Romance |
| French (Switzerland) | `fr-ch` | MLS8 Model | Romance |
| Spanish | `es` | MLS8 Model | Romance |
| Spanish (Latin America) | `es-419` | MLS8 Model | Romance |
| Italian | `it` | MLS8 Model | Romance |
| Portuguese | `pt` | MLS8 Model | Romance |
| Portuguese (Brazil) | `pt-br` | MLS8 Model | Romance |
| Polish | `pl` | MLS8 Model | West Slavic |
| Dutch | `nl` | MLS8 Model | West Germanic |
| Danish | `da` | MLS8 Model | North Germanic |
| Swedish | `sv` | MLS8 Model | North Germanic |
| Norwegian Bokmål | `nb` | MLS8 Model | North Germanic |
| Icelandic | `is` | MLS8 Model | North Germanic |
| Czech | `cs` | MLS8 Model | West Slavic |
| Slovak | `sk` | MLS8 Model | West Slavic |
| Slovenian | `sl` | MLS8 Model | South Slavic |
| Croatian | `hr` | MLS8 Model | South Slavic |
| Bosnian | `bs` | MLS8 Model | South Slavic |
| Serbian | `sr` | MLS8 Model | South Slavic |
| Macedonian | `mk` | MLS8 Model | South Slavic |
| Bulgarian | `bg` | MLS8 Model | South Slavic |
| Romanian | `ro` | MLS8 Model | Romance |
| Hungarian | `hu` | MLS8 Model | Uralic |
| Estonian | `et` | MLS8 Model | Uralic |
| Latvian | `lv` | MLS8 Model | Baltic |
| Lithuanian | `lt` | MLS8 Model | Baltic |
| Catalan | `ca` | MLS8 Model | Romance |
| Aragonese | `an` | MLS8 Model | Romance |
| Papiamento | `pap` | MLS8 Model | Romance |
| Haitian Creole | `ht` | MLS8 Model | Romance |
| Afrikaans | `af` | MLS8 Model | West Germanic |
| Luxembourgish | `lb` | MLS8 Model | West Germanic |
| Irish Gaelic | `ga` | MLS8 Model | Celtic |
| Scottish Gaelic | `gd` | MLS8 Model | Celtic |
| Welsh | `cy` | MLS8 Model | Celtic |
| **🌏 INDO-EUROPEAN LANGUAGES (Universal)** | | |
| Russian | `ru` | Universal Model | East Slavic |
| Russian (Latvia) | `ru-lv` | Universal Model | East Slavic |
| Ukrainian | `uk` | Universal Model | East Slavic |
| Belarusian | `be` | Universal Model | East Slavic |
| Hindi | `hi` | Universal Model | Indic |
| Bengali | `bn` | Universal Model | Indic |
| Urdu | `ur` | Universal Model | Indic |
| Punjabi | `pa` | Universal Model | Indic |
| Gujarati | `gu` | Universal Model | Indic |
| Marathi | `mr` | Universal Model | Indic |
| Nepali | `ne` | Universal Model | Indic |
| Assamese | `as` | Universal Model | Indic |
| Oriya | `or` | Universal Model | Indic |
| Sinhala | `si` | Universal Model | Indic |
| Konkani | `kok` | Universal Model | Indic |
| Bishnupriya Manipuri | `bpy` | Universal Model | Indic |
| Sindhi | `sd` | Universal Model | Indic |
| Persian | `fa` | Universal Model | Iranian |
| Persian (Latin) | `fa-latn` | Universal Model | Iranian |
| Kurdish | `ku` | Universal Model | Iranian |
| Greek (Modern) | `el` | Universal Model | Greek |
| Greek (Ancient) | `grc` | Universal Model | Greek |
| Armenian (East) | `hy` | Universal Model | Indo-European |
| Armenian (West) | `hyw` | Universal Model | Indo-European |
| Albanian | `sq` | Universal Model | Indo-European |
| Latin | `la` | Universal Model | Italic |
| **🇹🇷 TURKIC LANGUAGES (Universal)** | | |
| Turkish | `tr` | Universal Model | Turkic |
| Azerbaijani | `az` | Universal Model | Turkic |
| Kazakh | `kk` | Universal Model | Turkic |
| Kyrgyz | `ky` | Universal Model | Turkic |
| Uzbek | `uz` | Universal Model | Turkic |
| Tatar | `tt` | Universal Model | Turkic |
| Turkmen | `tk` | Universal Model | Turkic |
| Uyghur | `ug` | Universal Model | Turkic |
| Bashkir | `ba` | Universal Model | Turkic |
| Chuvash | `cu` | Universal Model | Turkic |
| Nogai | `nog` | Universal Model | Turkic |
| **🇫🇮 URALIC LANGUAGES (Universal)** | | |
| Finnish | `fi` | Universal Model | Uralic |
| Lule Saami | `smj` | Universal Model | Uralic |
| **🕌 SEMITIC LANGUAGES (Universal)** | | |
| Arabic | `ar` | Universal Model | Semitic |
| Hebrew | `he` | Universal Model | Semitic |
| Amharic | `am` | Universal Model | Semitic |
| Maltese | `mt` | Universal Model | Semitic |
| **🏝️ MALAYO-POLYNESIAN LANGUAGES (Universal)** | | |
| Indonesian | `id` | Universal Model | Malayo-Polynesian |
| Malay | `ms` | Universal Model | Malayo-Polynesian |
| **🇮🇳 DRAVIDIAN LANGUAGES (Universal)** | | |
| Tamil | `ta` | Universal Model | Dravidian |
| Telugu | `te` | Universal Model | Dravidian |
| Kannada | `kn` | Universal Model | Dravidian |
| Malayalam | `ml` | Universal Model | Dravidian |
| **🇬🇪 SOUTH CAUCASIAN LANGUAGES (Universal)** | | |
| Georgian | `ka` | Universal Model | South Caucasian |
| **🗾 LANGUAGE ISOLATES & OTHERS (Universal)** | | |
| Basque | `eu` | Universal Model | Language Isolate |
| Quechua | `qu` | Universal Model | Quechuan |
| **🛸 CONSTRUCTED LANGUAGES (Universal)** | | |
| Esperanto | `eo` | Universal Model | Constructed |
| Interlingua | `ia` | Universal Model | Constructed |
| Ido | `io` | Universal Model | Constructed |
| Lingua Franca Nova | `lfn` | Universal Model | Constructed |
| Lojban | `jbo` | Universal Model | Constructed |
| Pyash | `py` | Universal Model | Constructed |
| Lang Belta | `qdb` | Universal Model | Constructed |
| Quenya | `qya` | Universal Model | Constructed |
| Klingon | `piqd` | Universal Model | Constructed |
| Sindarin | `sjn` | Universal Model | Constructed |

</details>

#### 🔧 Model Selection Guide

| **Model** | **Languages** | **Use Case** | **Performance** |
|-----------|---------------|--------------|-----------------|
| **English Model** | English variants | Best for English | Highest accuracy for English |
| **MLS8 Model** | 8 European + similar | European languages | High accuracy for European |
| **Universal Model** | 60+ Indo-European + related | Other supported languages | Good for Indo-European families |

**⚠️ Unsupported Language Types:**
- **Tonal languages**: Chinese (Mandarin, Cantonese), Vietnamese, Thai, Burmese
- **Distant families**: Japanese, Korean, most African languages (Swahili, etc.)
- **Indigenous languages**: Most Native American, Polynesian (except Indonesian/Malay)
- **Recommendation**: For unsupported languages, use explicit `model_name` parameter with caution

### Initialization

```python
PhonemeTimestampAligner(
    preset="en-us",  # Language preset (recommended)
    model_name=None,  # Optional: explicit model override
    cupe_ckpt_path=None,  # Optional: direct checkpoint path
    lang="en-us",  # Language for phonemization
    duration_max=10,
    output_frames_key="phoneme_idx",
    device="cpu",
    boost_targets=True,
    enforce_minimum=True,
    enforce_all_targets=True,
    ignore_noise=True
)
```

**Parameters:**
- `preset`: **[NEW]** Language preset for automatic model and language selection. Use language codes like "de", "fr", "hi", "ja", etc. Supports 127+ languages with intelligent model selection.
- `model_name`: Name of the CUPE model (see [HuggingFace models](https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt)). Overrides preset selection. Downloaded automatically if available.
- `cupe_ckpt_path`: Local path to model checkpoint. Highest priority - overrides both preset and model_name.
- `lang`: Language code for phonemization ([espeak lang codes](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)). Only overridden by preset if using default.
- `duration_max`: Maximum segment duration (seconds, for batch padding). Best to keep <30 seconds.
- `output_frames_key`: Output key for frame assortment (`phoneme_idx`, `phoneme_label`, `group_idx`, `group_label`).
- `device`: Inference device (`cpu` or `cuda`).
- `silence_anchors`: Number of silent frames to anchor pauses (i.e., split segments when at least `silence_anchors` frames are silent). Set `0` to disable. Default is `10`. Set a lower value to increase sensitivity to silences. Best set `enforce_all_targets=True` when using this.
- `boost_targets`: Boost target phoneme probabilities for better alignment.
- `enforce_minimum`: Enforce minimum probability for target phonemes.
- `enforce_all_targets`: Band-aid postprocessing patch. It will insert phonemes missed by viterbi decoding at their expected positions based on target positions.
- `ignore_noise`:  Whether to ignore the predicted "noise" in the alignment. If set to True, noise will be skipped over. If False, long noisy/silent segments will be included as "noise" timestamps.
---

**Models:**
- `model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt"` for best performance on English. This model is trained on 1000 hours LibriSpeech.
- `model_name="en_libri1000_uj01d_e62_val_GER=0.2438.ckpt"` for best performance on heavy accented English speech. This is the same as above, just unsettled weights.
- `model_name="multi_MLS8_uh02_e36_val_GER=0.2334.ckpt"` for best performance on 8 european languages including English, German, French, Dutch, Italian, Spanish, Italian, Portuguese, Polish. This model's accuracy on English (buckeye corpus) is on par with the above (main) English model. We can only assume that the performance will be the same on the rest of the 7 languages.
- `model_name="multi_mswc38_ug20_e59_val_GER=0.5611.ckpt"` universal model for all non-tonal languages. This model is extremely acoustic, if it hears /i/, it will mark an /i/ regardless of the language.
- Models for tonal languages (Mandarin, Vietnamese, Thai) will have to wait.

Do not forget to set `lang="en-us"` parameter based on your model and [Language Identifier](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md).


### Process SRT File

```python
PhonemeTimestampAligner.process_srt_file(
    srt_path,
    audio_path,
    ts_out_path=None,
    extract_embeddings=False,
    vspt_path=None,
    do_groups=True,
    debug=True
)
```

**Parameters:**
- `srt_path`: Path to input SRT file (whisper JSON format).
- `audio_path`: Path to audio file.
- `ts_out_path`: Output path for timestamps (vs2 format).
- `extract_embeddings`: Extract embeddings.
- `vspt_path`: Path to save embeddings (`.pt` file).
- `do_groups`: Extract group timestamps.
- `debug`: Enable debug output.

**Returns:**  
- `timestamps_dict`: Dictionary with extracted timestamps.

---

### Process text sentences

```python
PhonemeTimestampAligner.process_sentence(
    text,
    audio_wav,
    ts_out_path=None,
    extract_embeddings=False,
    vspt_path=None,
    do_groups=True,
    debug=False
)
```

**Parameters:**
- `text`: Sentence/text.
- `audio_wav`: Audio waveform tensor (`torch.Tensor`).
- `ts_out_path`: Output path for timestamps (optional).
- `extract_embeddings`: Extract embeddings (optional).
- `vspt_path`: Path to save embeddings (`.pt`, optional).
- `do_groups`: Extract group timestamps (optional).
- `debug`: Enable debug output (optional).

Returns: `timestamps_dict`

---
### 🗣️ Convert Text to Phonemes

Phonemization in BFA is powered by the [phonemizer](https://github.com/bootphon/phonemizer) package, using the [espeak-ng](https://github.com/espeak-ng/espeak-ng) backend for robust multi-language support.

```python
PhonemeTimestampAligner.phonemize_sentence(text)
```

**Optional:** Change the espeak language after initialization:
```python
PhonemeTimestampAligner.phonemizer.set_backend(language='en')
```

**Method Description:**

Phonemizes a sentence and returns a detailed mapping:

- `text`: Original input sentence
- `ipa`: List of phonemes in IPA format
- `ph66`: List of phoneme class indices (mapped to 66-class set)
- `pg16`: List of phoneme group indices (16 broad categories)
- `words`: List of words corresponding to phonemes
- `word_num`: Word indices for each phoneme

**Example Usage:**
```python
result = PhonemeTimestampAligner.phonemize_sentence("butterfly")
print(result["ipa"])    # ['b', 'ʌ', 'ɾ', 'ɚ', 'f', 'l', 'aɪ']
print(result["ph66"])  # [29, 10, 58, 9, 43, 56, 23]
print(result["pg16"])  # [7, 2, 14, 2, 8, 13, 5]
```



### Extract Timestamps from Segment

```python
PhonemeTimestampAligner.extract_timestamps_from_segment(
    wav,
    wav_len,
    phoneme_sequence,
    start_offset_time=0,
    group_sequence=None,
    extract_embeddings=True,
    do_groups=True,
    debug=True
)
```

**Parameters:**
- `wav`: Audio tensor for the segment. Shape: [1, samples]
- `wav_len`: Length of the audio segment (samples).
- `phoneme_sequence`: List/tensor of phoneme indices (ph66)
- `start_offset_time`: Segment start offset (seconds).
- `group_sequence`: Optional group indices (pg16).
- `extract_embeddings`: Extract pooled phoneme embeddings.
- `do_groups`: Extract phoneme group timestamps.
- `debug`: Enable debug output.

**Returns:**
- `timestamp_dict`: Contains phoneme and group timestamps.
- `pooled_embeddings_phonemes`: Pooled phoneme embeddings or `None`.
- `pooled_embeddings_groups`: Pooled group embeddings or `None`.

---

### Convert to TextGrid

```python
PhonemeTimestampAligner.convert_to_textgrid(
    timestamps_dict,
    output_file=None,
    include_confidence=False
)
```

**Description:**  
Converts VS2 timestamp data to [Praat TextGrid](https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_format.html) format.

**Parameters:**
- `timestamps_dict`: Timestamp dictionary (from alignment).
- `output_file`: Path to save TextGrid file (optional).
- `include_confidence`: Include confidence values in output (optional).

**Returns:**  
- `textgrid_content`: TextGrid file content as string.



---



## 🔧 Advanced Usage


### 🎙️ Mel-Spectrum Alignment

BFA provides advanced mel-spectrogram compatibility methods for audio synthesis workflows. These methods enable seamless integration with [HiFi-GAN](https://github.com/jik876/hifi-gan) and [BigVGAN vocoder](https://github.com/NVIDIA/BigVGAN) and other mel-based audio processing pipelines.

See full [example here](examples/mel_spectrum_alignment.py).

#### Extract Mel Spectrogram

```python
PhonemeTimestampAligner.extract_mel_spectrum(
    wav,
    wav_sample_rate,
    vocoder_config={'num_mels': 80, 'num_freq': 1025, 'n_fft': 1024, 'hop_size': 256, 'win_size': 1024, 'sampling_rate': 22050, 'fmin': 0, 'fmax': 8000, 'model': 'whatever_22khz_80band_fmax8k_256x'}
)
```

**Description:**  
Extracts mel spectrogram from audio with vocoder compatibility.

**Parameters:**
- `wav`: Input waveform tensor of shape `(1, T)`
- `wav_sample_rate`: Sample rate of the input waveform
- `vocoder_config`: Configuration dictionary for HiFiGAN/BigVGAN vocoder compatibility.

**Returns:**  
- `mel`: Mel spectrogram tensor of shape `(frames, mel_bins)` - transposed for easy frame-wise processing

#### Frame-wise Assortment

```python
PhonemeTimestampAligner.framewise_assortment(
    aligned_ts,
    total_frames,
    frames_per_second,
    gap_contraction=5,
    select_key="phoneme_idx"
)
```

**Description:**  
Converts timestamp-based phoneme alignment to frame-wise labels matching mel-spectrogram frames.

**Parameters:**
- `aligned_ts`: List of timestamp dictionaries (from `phoneme_ts`, `group_ts`, or `word_ts`)
- `total_frames`: Total number of frames in the mel spectrogram
- `frames_per_second`: Frame rate of the mel spectrogram
- `gap_contraction`: Number of frames to fill silent gaps on either side of segments (default: 5)
- `select_key`: Key to extract from timestamps (`"phoneme_idx"`, `"group_idx"`, etc.)

**Returns:**  
- List of frame labels with length `total_frames`

#### Frame Compression

```python
PhonemeTimestampAligner.compress_frames(frames_list)
```

**Description:**  
Compresses consecutive identical frame values into run-length encoded format.

**Example:**
```python
frames = [0,0,0,0,1,1,1,1,3,4,5,4,5,2,2,2]
compressed = compress_frames(frames)
# Returns: [(0,4), (1,4), (3,1), (4,1), (5,1), (4,1), (5,1), (2,3)]
```

**Returns:**  
- List of `(frame_value, count)` tuples

#### Frame Decompression

```python
PhonemeTimestampAligner.decompress_frames(compressed_frames)
```

**Description:**  
Decompresses run-length encoded frames back to full frame sequence.

**Parameters:**
- `compressed_frames`: List of `(phoneme_id, count)` tuples

**Returns:**  
- Decompressed list of frame labels

<details>
<summary>📊 Complete mel-spectrum alignment example</summary>

```python
# pip install librosa
import torch
from bournemouth_aligner import PhonemeTimestampAligner

# Initialize aligner
extractor = PhonemeTimestampAligner(model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt", 
                                  lang='en-us', duration_max=10, device='cpu')

# Process audio and get timestamps
audio_wav = extractor.load_audio("examples/samples/audio/109867__timkahn__butterfly.wav")
timestamps = extractor.process_sentence("butterfly", audio_wav)

# Extract mel spectrogram with vocoder compatibility
vocoder_config = {'num_mels': 80, 'hop_size': 256, 'sampling_rate': 22050}
segment_wav = audio_wav[:, :int(timestamps['segments'][0]['end'] * extractor.resampler_sample_rate)]
mel_spec = extractor.extract_mel_spectrum(segment_wav, extractor.resampler_sample_rate, vocoder_config)

# Create frame-wise phoneme alignment
total_frames = mel_spec.shape[0]
frames_per_second = total_frames / timestamps['segments'][0]['end']
frames_assorted = extractor.framewise_assortment(
    aligned_ts=timestamps['segments'][0]['phoneme_ts'], 
    total_frames=total_frames, 
    frames_per_second=frames_per_second
)

# Compress and visualize
compress_framesed = extractor.compress_frames(frames_assorted)
# Use provided plot_mel_phonemes() function to visualize
```

</details>



### 🔗 Integration Examples

<details>
<summary>🎙️ Whisper Integration</summary>

```python
# pip install git+https://github.com/openai/whisper.git 
import whisper, json
from bournemouth_aligner import PhonemeTimestampAligner

# Transcribe and align
model = whisper.load_model("turbo")
result = model.transcribe("audio.wav")
with open("whisper_output.srt.json", "w") as f:
    json.dump(result, f)

# Process with BFA
extractor = PhonemeTimestampAligner(model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt")
timestamps = extractor.process_srt_file("whisper_output.srt.json", "audio.wav", "timestamps.json")
```

</details>

<details>
<summary>🔬 Manual Processing Pipeline</summary>

```python
import torch
from bournemouth_aligner import PhonemeTimestampAligner

# Initialize and process
extractor = PhonemeTimestampAligner(model_name="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt")
audio_wav = extractor.load_audio("audio.wav")  # Handles resampling and normalization
timestamps = extractor.process_sentence("your text here", audio_wav)

# Export to Praat
extractor.convert_to_textgrid(timestamps, "output.TextGrid")
```

</details>

### 🤖 Machine Learning Integration

For phoneme embeddings in ML pipelines, check out our [embeddings example](examples/read_embeddings.py).

---

## 💻 Command Line Interface (CLI)

### 🚀 Quick CLI Usage

```bash
# Basic alignment
balign audio.wav transcription.srt.json output.json

# With debug output
balign audio.wav transcription.srt.json output.json --debug

# Extract embeddings
balign audio.wav transcription.srt.json output.json --embeddings embeddings.pt
```

### ⚙️ Command Syntax

```bash
balign [OPTIONS] AUDIO_PATH SRT_PATH OUTPUT_PATH
```

### 📝 Arguments & Options

<div align="center">

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `AUDIO_PATH` | **Required** | - | Audio file path (.wav, .mp3, .flac) |
| `SRT_PATH` | **Required** | - | SRT JSON file path |
| `OUTPUT_PATH` | **Required** | - | Output timestamps (.json) |

</div>

<details>
<summary>🔧 Advanced Options</summary>

| Option | Default | Description |
|--------|---------|-------------|
| `--model TEXT` | `en_libri1000_uj01d_e199_val_GER=0.2307.ckpt` | CUPE model from [HuggingFace](https://huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt) |
| `--lang TEXT` | `en-us` | Language code ([espeak codes](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)) |
| `--device TEXT` | `cpu` | Processing device (`cpu` or `cuda`) |
| `--embeddings PATH` | None | Save phoneme embeddings (.pt file) |
| `--duration-max FLOAT` | `10.0` | Max segment duration (seconds) |
| `--debug / --no-debug` | `False` | Enable detailed output |
| `--boost-targets / --no-boost-targets` | `True` | Enable target phoneme boosting |
| `--help` | | Show help message |
| `--version` | | Show version info |

</details>

### 🌟 CLI Examples

```bash
# Basic usage
balign audio.wav transcription.srt.json output.json

# With GPU and embeddings  
balign audio.wav transcription.srt.json output.json --device cuda --embeddings embeddings.pt

# Multi-language (English + 8 european langauges model available)
balign audio.wav transcription.srt.json output.json --lang es

# Batch processing
for audio in *.wav; do balign "$audio" "${audio%.wav}.srt" "${audio%.wav}.json"; done
```

### 📊 Input Format

SRT files must be in JSON format:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "hello world this is a test"
    },
    {
      "start": 3.5,
      "end": 7.2,  
      "text": "another segment of speech"
    }
  ]
}
```

### 🎯 Creating Input Files

Use Whisper for transcription (see [Integration Examples](#-integration-examples)) or create SRT JSON manually with the format shown above.

### 🔍 Debug Mode

Enable comprehensive processing information:

```bash
balign audio.wav transcription.srt.json output.json --debug
```

<details>
<summary>📊 Sample debug output</summary>

```
🚀 Bournemouth Forced Aligner
🎵 Audio: audio.wav
📄 SRT: transcription.srt.json  
💾 Output: output.json
🏷️  Language: en-us
🖥️  Device: cpu
🎯 Model: en_libri1000_uj01d_e199_val_GER=0.2307.ckpt
--------------------------------------------------
🔧 Initializing aligner...
Setting backend for language: en-us
✅ Aligner initialized successfully
🎵 Processing audio...
Loaded SRT file with 1 segments from transcription.srt.json
Resampling audio.wav from 22050Hz to 16000Hz
Expected phonemes: ['p', 'ɹ', 'ɪ', ...'ʃ', 'ə', 'n']
Target phonemes: 108, Expected: ['p', 'ɹ', 'ɪ', ..., 'ʃ', 'ə', 'n']
Spectral length: 600
Forced alignment took 135.305 ms
Aligned phonemes: 108
Target phonemes: 108
SUCCESS: All target phonemes were aligned!

============================================================
PROCESSING SUMMARY
============================================================
Total segments processed: 1
Perfect sequence matches: 1/1 (100.0%)
Total phonemes aligned: 108
Overall average confidence: 0.502
============================================================
Results saved to: output.json
✅ Timestamps extracted to output.json
📊 Processed 1 segments with 108 phonemes
🎉 Processing completed successfully!
```

</details>

---



---

## 🧠 How Does It Work?

Read full paper: [BFA: REAL-TIME MULTILINGUAL TEXT-TO-SPEECH FORCED ALIGNMENT](https://arxiv.org/pdf/2509.23147)

### 🔄 Processing Pipeline

```mermaid
graph TD
    A[Audio Input] --> B[RMS Normalization]
    B --> C[Audio Windowing]
    C --> D[CUPE Model Inference]
    D --> E[Phoneme/Group Probabilities]
    E --> F[Text Phonemization]
    F --> G[Target Boosting]
    G --> H[Viterbi Forced Alignment]
    H --> I[Missing Phoneme Recovery]
    I --> J[Confidence Calculation]
    J --> K[Frame-to-Time Conversion]
    K --> L[Output Generation]
```

**CTC Transition Rules:**
- **Stay**: Remain in current state (repeat phoneme or blank)
- **Advance**: Move to next state in sequence
- **Skip**: Jump over blank to next phoneme (when consecutive phonemes differ)

**Core Components:**

1. **🎵 Audio Preprocessing**: RMS normalization and windowing (120ms windows, 80ms stride)
2. **🧠 CUPE Model**: Contextless Universal Phoneme Encoder extracts frame-level phoneme probabilities
3. **📝 Phonemization**: espeak-ng converts text to 66-class phoneme indices (ph66) and 16 phoneme groups (pg16)
4. **🎯 Target Boosting**: Enhances probabilities of expected phonemes for better alignment
5. **🔍 CTC style Viterbi**: CTC-based forced alignment with minimum probability enforcement
6. **🛠️ Recovery Mechanism**: Ensures all target phonemes appear in alignment, even with low confidence
7. **📊 Confidence Scoring**: Frame-level probability averaging with adaptive thresholding
8. **⏱️ Timestamp Conversion**: Frame indices to millisecond timestamps with segment offset

### 🎛️ Key Alignment Parameters

BFA provides several unique control parameters not available in traditional aligners like MFA:

#### 🎯 `boost_targets` (Default: `True`)
Increases log-probabilities of expected phonemes by a fixed boost factor (typically +5.0) before Viterbi decoding. If the sentence is very long or contains every possible phoneme, then boosting them all equally doesn't have much effect—because no phoneme stands out more than the others.

**When it helps:**
- **Cross-lingual scenarios**: Using English models on other languages where some phonemes are underrepresented
- **Noisy audio**: When target phonemes have very low confidence but should be present
- **Domain mismatch**: When model training data differs significantly from your audio

**Important caveat:** For monolingual sentences, boosting affects ALL phonemes in the target sequence equally, making it equivalent to no boosting. The real benefit comes when using multilingual models or when certain phonemes are systematically underrepresented.

#### 🛡️ `enforce_minimum` (Default: `True`) 
Ensures every target phoneme has at least a minimum probability (default: 1e-8) at each frame, preventing complete elimination during alignment.

**Why this matters:**
- Prevents target phonemes from being "zeroed out" by the model
- Guarantees that even very quiet or unclear phonemes can be aligned
- Helps for highly noisy audio in which all phonemes, not just targets, have extremely low probabilities.

#### 🔒 `enforce_all_targets` (Default: `True`)
**This is BFA's key differentiator from MFA.** After Viterbi decoding, BFA applies post-processing to guarantee that every target phoneme is present in the final alignment—even those with low acoustic probability. However, **downstream tasks can filter out these "forced" phonemes using their confidence scores**. For practical use, consider setting a confidence threshold  e.g., `timestamps["phoneme_ts"][p]["confidence"] <0.05`) to exclude phonemes that were aligned with little to no acoustic evidence.

**Recovery mechanism:**
1. Identifies any missing target phonemes after initial alignment
2. Finds frames with highest probability for each missing phoneme
3. Strategically inserts missing phonemes by:
   - Replacing blank frames when possible
   - Searching nearby frames within a small radius
   - Force-replacing frames as last resort

**Use cases:**
- **Guaranteed coverage**: When you need every phoneme to be timestamped
- **Noisy environments**: Where some phonemes might be completely missed by standard Viterbi
- **Research applications**: When completeness is more important than probabilistic accuracy

#### ⚖️ Parameter Interaction Effects

| Scenario | Recommended Settings | Outcome |
|----------|---------------------|---------|
| **Clean monolingual audio** | All defaults | Standard high-quality alignment |
| **Cross-lingual/noisy** | `boost_targets=True` | Better phoneme recovery |
| **Research/completeness** | `enforce_all_targets=True` | 100% phoneme coverage |
| **Probabilistically strict** | `enforce_all_targets=False` | Only high-confidence alignments |

**Technical Details:**

- **Audio Processing**: 16kHz sampling, sliding window approach for long audio
- **Model Architecture**: Pre-trained CUPE-2i models from [HuggingFace](https://huggingface.co/Tabahi/CUPE-2i)  
- **Alignment Strategy**: CTC path construction with blank tokens between phonemes
- **Quality Assurance**: Post-processing ensures 100% target phoneme coverage (when enabled)

> **Performance Note**: CPU-optimized implementation. The iterative Viterbi algorithm and windowing operations are designed for single-threaded efficiency. Most operations are vectorized where possible, so batch processing should be faster on GPUs.

--- 




### 📊 Alignment Error Analysis

**Alignment error histogram on [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1):**

<div align="center">
    <img src="examples/samples/images/BFA_vs_MFA_errors_on_TIMIT.png" alt="Alignment Error Histogram - TIMIT Dataset" width="600"/>
</div>

- Most phoneme boundaries are aligned within **±30ms** of ground truth.
- Errors above **100ms** are rare and typically due to ambiguous or noisy segments.

**For comparison:**  
See [Montreal Forced Aligner](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.pdf) for benchmark results on similar datasets.





> ⚠️ **Best Performance**: For optimal results, use audio segments **under 30 seconds**. For longer audio, segment first using Whisper or VAD. Audio duration above 60 seconds creates too many possibilities for the Viterbi algorithm to handle properly.

---

## 🔬 Comparison with MFA

Our alignment quality compared to Montreal Forced Aligner (MFA) using [Praat](https://www.fon.hum.uva.nl/praat/) TextGrid visualization:

<div align="center">

| Metric | BFA | MFA |
|--------|-----|-----|
| **Speed** | 0.2s per 10s audio | 10s per 2s audio |
| **Real-time potential** | ✅ Yes (contextless) | ❌ No |
| **Stop consonants** | ✅ Better (t,d,p,k) | ⚠️ Extends too much |
| **Tail endings** | ⚠️ Sometimes missed | ❌ Onset only |
| **Breathy sounds** | ⚠️ Misses h# | ✅ Captures |
| **Punctuation** | ✅ Silence aware | ❌ No punctuation |

</div>

### 📊 Sample Visualizations in Praat

**"In being comparatively modern..."** - LJ Speech Dataset  
[🎵 Audio Sample](examples/samples/LJSpeech/LJ001-0002.wav)

![Praat Alignment Example](examples/samples/images/LJ02_praat.png)



---




### Citation

[Rehman, A., Cai, J., Zhang, J.-J., & Yang, X. (2025). BFA: Real-time multilingual text-to-speech forced alignment. *arXiv*. https://arxiv.org/abs/2509.23147](https://arxiv.org/pdf/2509.23147)

```bibtex
@misc{rehman2025bfa,
      title={BFA: Real-time Multilingual Text-to-speech Forced Alignment}, 
      author={Abdul Rehman and Jingyao Cai and Jian-Jun Zhang and Xiaosong Yang},
      year={2025},
      eprint={2509.23147},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2509.23147}, 
}
```
---

<div align="center">


[⭐ Star us on GitHub](https://github.com/tabahi/bournemouth-forced-aligner) • [🐛 Report Issues](https://github.com/tabahi/bournemouth-forced-aligner/issues)

</div>
