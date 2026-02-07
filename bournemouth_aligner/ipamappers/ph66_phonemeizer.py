# pip install requests phonemizer
# apt-get install espeak-ng

import sys
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from unicodedata import normalize
import re
import json
try:
    from .ph66_mapper import phoneme_mapper, compound_mapper, phoneme_mapped_index, phoneme_groups_index, phoneme_groups_mapper, get_compound_phoneme_mapping # maps IPA phonemesto a smaller set of phonemes#
except ImportError:
    import sys
    sys.path.append('.')
    try:
        from bournemouth_aligner.ipamappers.ph66_mapper import phoneme_mapper, compound_mapper, phoneme_mapped_index, phoneme_groups_index, phoneme_groups_mapper, get_compound_phoneme_mapping # maps IPA phonemesto a smaller set of phonemes#
    except ImportError:
        import sys
        sys.path.append('.')
        from bournemouth_aligner.ipamappers.ph66_mapper import phoneme_mapper, compound_mapper, phoneme_mapped_index, phoneme_groups_index, phoneme_groups_mapper, get_compound_phoneme_mapping # maps IPA phonemesto a smaller set of phonemes#
    
SIL_PHN="SIL"
assert(SIL_PHN in phoneme_mapped_index)
noise_PHN="noise"
assert(noise_PHN in phoneme_mapped_index)
noise_group = phoneme_groups_mapper[phoneme_mapped_index[noise_PHN]]
SIL_token = phoneme_mapped_index[SIL_PHN]


# Languages with non-Latin scripts that need special word breaking
# Based on espeak-ng supported languages (see espeak_languages.md)
specical_unicode_langs = [
    # CJK languages (no spaces between words)
    'zh', 'cmn', 'yue', 'hak',  # Chinese variants
    'ja',  # Japanese
    'ko',  # Korean
    # Indic languages (Devanagari and other Indic scripts)
    'hi',  # Hindi
    'mr',  # Marathi
    'ne',  # Nepali
    'kok', # Konkani
    'bn',  # Bengali
    'as',  # Assamese
    'gu',  # Gujarati
    'pa',  # Punjabi
    'or',  # Oriya
    'sd',  # Sindhi
    'si',  # Sinhala
    'bpy', # Bishnupriya Manipuri
    # Dravidian languages
    'ta',  # Tamil
    'te',  # Telugu
    'kn',  # Kannada
    'ml',  # Malayalam
    # Semitic/Arabic script languages
    'ar',  # Arabic
    'ur',  # Urdu
    'fa',  # Persian/Farsi
    'he',  # Hebrew
    # Other non-Latin scripts
    'am',  # Amharic (Ethiopic script)
    'my',  # Burmese
    'ka',  # Georgian
    'hy', 'hyw',  # Armenian (East and West)
    'el', 'grc',  # Greek (Modern and Ancient)
    'th',  # Thai
    'ug',  # Uyghur (Arabic script)
]

# Languages that typically don't separate words with spaces (treat characters as tokens)
no_space_langs = ['zh', 'cmn', 'yue', 'hak', 'ja', 'ko', 'th', 'my']


class Phonemizer:

    def __init__(self, language=None, remove_noise_phonemes=True, unicode_enable=False):
        '''
        Initialize the phonemizer with a specific language backend and options.
        
        Args:
        language (str): Language code for the phonemizer backend (e.g., 'en-us').
        remove_noise_phonemes (bool): If True, noise phonemes will be removed from the output.
        - if language=None, then set_backend(language='en') must be called before phonemize_sentence
        - remove_noise_phonemes: remove noise phonemes from the phoneme list that includes possibly valid but unknown phonemes
        - unicode_enable: if True, words won't be seperated by spaces, useful for languages like Chinese, Hindi, etc.
          `unicode_enable` is automatically set to True for languages like Chinese, Japanese, Korean, Hindi, etc.
        '''
        self.unicode_enable = unicode_enable
        self.word_counts = 0
        self.phn_counts = 0 
        self.phoneme_vocab_counts = {}
        self.phoneme_vocab_mapped_counts = {}
        
        self.unmapped_phonemes = {}
        self.phoneme_index = phoneme_mapped_index
        self.remove_noise_phonemes = remove_noise_phonemes
        self.noise_count = 0
        self.sil_count = 0
        self.ph_total_count = 0

        self.phonemes_ipa_key = "eipa"
        self.phonemes_key = "ph66"
        self.phoneme_groups_key = "pg16"
        
        for key, value in phoneme_mapper.items():
            self.phoneme_vocab_mapped_counts.update({value: 0})
            
        #print(f"Mapping: {len(phoneme_mapper)} phonemes onto {len(self.phoneme_vocab_mapped_counts)} phonemes")

        # set mapping dicts:
        self.index_to_plabel = {v: k for k, v in phoneme_mapped_index.items()}
        self.index_to_glabel = {v: k for k, v in phoneme_groups_index.items()}
        self.phoneme_id_to_group_id = phoneme_groups_mapper
        
        self.language = language
        if (language is not None):
            self.set_backend(language=language)

    def print_mapping_summary(self):
        """
        Print a summary of phonemizer statistics in a readable format.
        """
        print("=" * 40)
        print("Phonemizer Statistics Summary")
        print("=" * 40)
        print(f"Total words processed   : {self.word_counts}")
        print(f"Total phonemes counted  : {self.phn_counts}")
        print(f"Unique phonemes found   : {len(self.phoneme_vocab_counts)}")
        print(f"Mapped phonemes tracked : {len(self.phoneme_vocab_mapped_counts)}")
        print(f"Unmapped phonemes       : {len(self.unmapped_phonemes)}")
        print(f"Silence phoneme count   : {self.sil_count}")
        print(f"Noise phoneme count     : {self.noise_count}")
        print(f"Total phoneme count     : {self.ph_total_count}")
        ratio = self.noise_count / (self.ph_total_count + 1e-8)
        print(f"Noise/Phoneme ratio     : {ratio:.4f}")
        print("=" * 40)
        print("Phoneme Vocabulary Counts:")
        for ph, count in sorted(self.phoneme_vocab_counts.items(), key=lambda x: -x[1]):
            print(f"  {ph}: {count}")
        print("=" * 40)
        if self.unmapped_phonemes:
            print("Unmapped Phonemes:")
            for ph, count in sorted(self.unmapped_phonemes.items(), key=lambda x: -x[1]):
                print(f"  {ph}: {count}")
            print("=" * 40)
    
                                 
    def set_backend(self, language='en'):
        
        alt_codes = {'zh-CN': 'cmn', 'fr': 'fr-fr', 'en': 'en-us', 'uk': 'zle', 'sv-SE': 'sv', 'ga-IE': 'ga', 'cv': 'cu'}

        if (language in alt_codes):
            language = alt_codes[language]
        print(f"Setting espeak backend for language: {language}")
        self.backend = EspeakBackend(language=language, preserve_punctuation=False, tie=False, with_stress=False)
        self.language = language

        # unicode_enable for languages with non-Latin scripts
        # Check if the language code matches any in specical_unicode_langs
        if any(lang in language for lang in specical_unicode_langs):
            self.unicode_enable = True
            print(f"Unicode enabled for language: {language}")

    def break_words_alpha(self, single_sentence):
        """
        Process a sentence to extract words and insert <SIL> tokens for punctuation.
        
        Args:
            single_sentence (str): The input sentence to process
            
        Returns:
            list: Words with <SIL> tokens inserted for punctuation
        """
        
        # Split the sentence into tokens (words and punctuation)
        # This regex captures words, commas, periods, and other punctuation
        tokens = re.findall(r"\b\w+\b|[.,!?;:]", single_sentence)
        
        words = []
        
        for token in tokens:
            if re.match(r"\b\w+\b", token):  # It's a word
                words.append(token)
            elif token in [',', ';', ':', '<SIL>']:  # Comma - insert 1 SIL token
                words.append("<SIL>")
            elif token in ['.', '!', '?']:  # Full stop, exclamation, question - insert 2 SIL tokens
                words.append("<SIL>")
                words.append("<SIL>")
            # You can add more punctuation rules here as needed
        
        return words

    def break_words(self, single_sentence):
        # for backward compatibility, only latin based languages
        return self.break_words_alpha(single_sentence)
    
    def break_words_special(self, single_sentence):
        """
        Special word breaker for unicode-enabled languages (e.g., Hindi, Arabic, Chinese, Japanese).
        - For CJK (no-space) languages (zh, cmn, ja, ko) each non-punctuation character is treated as a token.
        - For space-separated unicode languages (hi, ar, etc.) split on whitespace but handle punctuation
          similarly to break_words_alpha by inserting <SIL> tokens.
        """
        if single_sentence is None:
            return []

        lang = (self.language or "").lower()

        # Punctuation sets covering multiple scripts
        # Single SIL punctuation (pauses like commas, semicolons)
        single_sil_punct = {
            ',', ';', ':',           # Latin
            '،', '٬', '؛',           # Arabic comma, thousands separator, semicolon
            '、',                    # CJK enumeration comma
            '，',                    # CJK comma
            '፣', '፤',                # Ethiopic comma, semicolon (Amharic)
            '·',                     # Georgian comma
            '၊',                     # Burmese section mark
            '๚', '๛',                # Thai abbreviation marks
        }

        # Double SIL punctuation (sentence enders)
        double_sil_punct = {
            '.', '!', '?',           # Latin
            '؟', '!',                # Arabic question mark, exclamation
            '।', '॥',                # Devanagari danda, double danda (Hindi, etc.)
            '។', '៕',                # Khmer sign, deprecation mark
            '。', '！', '？',         # CJK period, exclamation, question
            '፡', '።', '፨',           # Ethiopic word separator, full stop, paragraph (Amharic)
            '܀', '܁', '܂',           # Syriac end marks
            '။',                     # Burmese period
            ';',                     # Greek question mark (looks like semicolon in Greek)
        }

        all_punct = single_sil_punct.union(double_sil_punct)

        words = []

        if any(code in lang for code in no_space_langs):
            # For CJK languages: treat each character as a separate token
            i = 0
            while i < len(single_sentence):
                ch = single_sentence[i]

                # Skip whitespace
                if ch.isspace():
                    i += 1
                    continue

                # Check for explicit <SIL> tokens
                if single_sentence[i:i+5] == '<SIL>':
                    words.append("<SIL>")
                    i += 5
                    continue

                # Handle punctuation
                if ch in double_sil_punct:
                    words.append("<SIL>")
                    words.append("<SIL>")
                elif ch in single_sil_punct:
                    words.append("<SIL>")
                else:
                    # Regular character - treat as a word
                    words.append(ch)

                i += 1
            return words

        # For space-separated unicode languages (e.g., Hindi, Arabic, Urdu)
        # First, handle explicit <SIL> tokens by replacing them with a placeholder
        SIL_PLACEHOLDER = '\x00SIL\x00'
        sentence_processed = single_sentence.replace('<SIL>', SIL_PLACEHOLDER)

        # Split on whitespace
        tokens = re.findall(r"\S+", sentence_processed.strip())

        for token in tokens:
            # Check if entire token is the SIL placeholder
            if token == SIL_PLACEHOLDER:
                words.append("<SIL>")
                continue

            # Handle tokens that may contain the SIL placeholder mixed with other text
            if SIL_PLACEHOLDER in token:
                # Split by SIL placeholder and process each part
                parts = token.split(SIL_PLACEHOLDER)
                for idx, part in enumerate(parts):
                    if idx > 0:
                        words.append("<SIL>")
                    if part:
                        # Process this part for punctuation
                        self._process_token_with_punctuation(part, words, single_sil_punct, double_sil_punct, all_punct)
                continue

            # Normal token processing
            self._process_token_with_punctuation(token, words, single_sil_punct, double_sil_punct, all_punct)

        return words

    def _process_token_with_punctuation(self, token, words, single_sil_punct, double_sil_punct, all_punct):
        """
        Helper method to process a token and extract words and punctuation.
        Handles leading and trailing punctuation, converting them to <SIL> tokens.
        """
        t = token

        # Handle leading punctuation
        while t and (t[0] in all_punct):
            punct = t[0]
            if punct in double_sil_punct:
                words.append("<SIL>")
                words.append("<SIL>")
            else:
                words.append("<SIL>")
            t = t[1:]

        if not t:
            return

        # Handle trailing punctuation
        trailing = []
        while t and (t[-1] in all_punct):
            trailing.append(t[-1])
            t = t[:-1]

        # Add the core word (if any remains after stripping punctuation)
        if t:
            words.append(t)

        # Convert trailing punctuation to SIL tokens (reverse order to maintain sequence)
        for punct in reversed(trailing):
            if punct in double_sil_punct:
                words.append("<SIL>")
                words.append("<SIL>")
            else:
                words.append("<SIL>")
    
    
    def phonemize_sentence(self, sentence):
        '''
        Phonemize a single sentence and return a segment_out dict with phoneme indices and groups.

        Args:
        - sentence (str): The input sentence to phonemize.
        Returns:
        - segment_out dict with keys:
            - text: original sentence
            - ph66: list of phoneme class indices (mapped to phoneme_mapped_index)
            - pg16: list of phoneme group indices (mapped to phoneme_groups_mapper)
            - mipa: list of mapped phonemes in IPA format
            - self.phonemes_ipa_key ("eipa"): list of original espeak phonemes before mapping (one-to-one with mapped phonemes), IPA format, full e-speak dictionary output
            - words: list of words corresponding to the phonemes
            - word_num: list of word indices corresponding to the phonemes
        '''

        if self.language is None:
            raise ValueError("Phonemizer backend language is not set. Call set_backend(language) before phonemizing sentences.")
    
        if self.language in specical_unicode_langs or self.unicode_enable:
            # For languages like Chinese, Japanese, Korean, Hindi, etc., treat the entire sentence as one word
            words = self.break_words_special(sentence)
        else:
            words = self.break_words_alpha(sentence)

        # Phonemize the words in this segment
        phonemized_words = self.backend.phonemize(
            words, 
            separator=Separator(phone='|', word=None, syllable=None), 
            strip=True, 
            njobs=1
        )
        

        
        ph_sil_idx = self.phoneme_index[SIL_PHN]
        ph_sil_group = phoneme_groups_mapper[ph_sil_idx]

        segment_out = {self.phonemes_key:[], self.phoneme_groups_key:[], self.phonemes_ipa_key:[], "word_num":[], "words":[],  "mipa":[], "text": sentence}
        # Process each word in the segment
        for word_idx, (word, phonemized) in enumerate(zip(words, phonemized_words)):
            
            
            if word == "<SIL>":
                # Special case for <SIL> which is treated as silence
                segment_out[self.phonemes_key].append(ph_sil_idx)
                segment_out[self.phoneme_groups_key].append(ph_sil_group)
                segment_out["mipa"].append(SIL_PHN)
                segment_out[self.phonemes_ipa_key].append(SIL_PHN)
                segment_out["word_num"].append(word_idx)
                segment_out["words"].append("<sil>")  # Placeholder for silence
                self.sil_count += 1
                continue

            word_phonemes = phonemized.split('|')

            # Filter out empty phonemes
            word_phonemes = [ph for ph in word_phonemes if ph != ""]

            #print(f"Word: '{word}' -> Phonemes: {word_phonemes}")
            
            has_valid_phonemes = False
            
            # Process each phoneme
            for eph in word_phonemes:
                # Save the original espeak phoneme before any modifications
                original_eph = eph.strip()
                original_eph = normalize('NFC', original_eph)

                eph = original_eph

                # Update phoneme count statistics
                if eph not in self.phoneme_vocab_counts:
                    self.phoneme_vocab_counts[eph] = 1
                else:
                    self.phoneme_vocab_counts[eph] += 1

                # Handle unmapped phonemes
                if eph not in phoneme_mapper and eph not in compound_mapper:
                    if eph not in self.unmapped_phonemes:
                        print(f"Warning: phoneme not found in mapper: {eph}")
                        self.unmapped_phonemes.update({eph: 1})
                    else:
                        self.unmapped_phonemes[eph] += 1
                    eph = noise_PHN

                # Get compound phoneme mapping
                ph_comp = get_compound_phoneme_mapping(eph)
                assert isinstance(ph_comp, list), f"Compound mapping should return a list, got {type(ph_comp)} for phoneme {eph}"

                for ph in ph_comp:
                    if ph not in self.phoneme_index:
                        ph = noise_PHN

                    # Skip noise phonemes if configured to do so
                    if self.remove_noise_phonemes and ph == noise_PHN:
                        self.noise_count += 1
                        # Skip both the mapped phoneme AND the original eph
                        continue

                    # At this point, we have a valid phoneme that will be included
                    has_valid_phonemes = True

                    # Get phoneme index and group
                    ph_idx = self.phoneme_index[ph]
                    ph_group = phoneme_groups_mapper.get(ph_idx, noise_group)  # Default to noise_group if not found

                    # Add to segment output (including original espeak phoneme for one-to-one correspondence)
                    segment_out[self.phonemes_key].append(ph_idx)
                    segment_out[self.phoneme_groups_key].append(ph_group)
                    segment_out["mipa"].append(ph)
                    segment_out[self.phonemes_ipa_key].append(original_eph)
                    segment_out["word_num"].append(word_idx)

                    self.phn_counts += 1
                    self.ph_total_count += 1
            # end phoneme loop

            # Track which words actually have phonemes in the final output
            if has_valid_phonemes:
                segment_out["words"].append(word.strip())
            else:
                segment_out["words"].append("")  # Placeholder for words without valid phonemes
        # end words loop


        # After processing all words, create indexed word list and remap word_num
        if segment_out[self.phonemes_key]:  # Only process if there are phonemes
            # Find indices of words with valid phonemes
            words_with_phonemes = [i for i, w in enumerate(segment_out["words"]) if w]
            # Create a mapping from original word indices to new compressed indices
            word_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(words_with_phonemes)}
            # Build the new indexed word list
            indexed_words = [segment_out["words"][i] for i in words_with_phonemes]
            # Remap the word_num based on the new compressed indices
            remapped_word_num = [word_remap[idx] for idx in segment_out["word_num"]]
            segment_out["word_num"] = remapped_word_num
            segment_out["words"] = indexed_words
            
            
            # Perform final validation
            assert len(segment_out[self.phonemes_key]) == len(segment_out[self.phoneme_groups_key]) == len(segment_out["eipa"]) == len(segment_out["word_num"])
            assert max(segment_out["word_num"]) + 1 == len(segment_out["words"])
            # Check that all words with phonemes are accounted for
            

        print(segment_out)
        
        return segment_out

def simple_test_en():
    

    text = "Hello world, auditors. Persephone."
    text = [line.strip() for line in text.split('\n') if line]

    try:
        backend = EspeakBackend('en-us', preserve_punctuation=False, tie=False, with_stress=False)
        phn = [backend.phonemize([line], separator=Separator(phone='|', word=' ', syllable=None), strip=True, njobs=1) for line in text]
        print("Espeak Backend Phonemization Result:")
        print(phn)
    except Exception as e:
        raise RuntimeError(
            f"Failed to phonemize with EspeakBackend: {e}\n"
            "Make sure espeak-ng is installed and available in PATH.\n"
            "To install dependencies, run:\n"
            "  pip install requests phonemizer\n"
            "  sudo apt-get install espeak-ng"
        )
    
    segment_out =  Phonemizer('en').phonemize_sentence(text[0])
    print("Phonemizer Segment Output:")
    print(json.dumps(segment_out, indent=2, ensure_ascii=False))



def simple_test_2():
    """Test word breaking for Hindi, Arabic, and other special languages"""

    print("\n" + "=" * 70)
    print("Testing Special Languages Word Breaking")
    print("=" * 70)

    # Test Hindi
    print("\n" + "-" * 70)
    print("HINDI TESTS")
    print("-" * 70)

    phonemizer_hi = Phonemizer(language='hi', remove_noise_phonemes=False)
    # examples by claude
    hindi_tests = [
        ("नमस्ते दुनिया", "Hello world"),
        ("यह एक परीक्षण है।", "This is a test."),
        ("क्या हाल है?", "How are you?"),
        ("मैं ठीक हूं, धन्यवाद।", "I'm fine, thank you."),
        ("एक, दो, तीन।", "One, two, three."),
        ("नमस्ते <SIL> दुनिया", "With explicit SIL token"),
    ]

    for hi_text, description in hindi_tests:
        print(f"\n{description}:")
        print(f"  Input: {hi_text}")
        words = phonemizer_hi.break_words_special(hi_text)
        print(f"  Words: {words}")
        print(f"  Count: {len(words)} tokens")

    # Test Arabic
    print("\n" + "-" * 70)
    print("ARABIC TESTS")
    print("-" * 70)

    phonemizer_ar = Phonemizer(language='ar', remove_noise_phonemes=False)

    arabic_tests = [
        ("مرحبا بالعالم", "Hello world"),
        ("هذا اختبار.", "This is a test."),
        ("كيف حالك؟", "How are you?"),
        ("أنا بخير، شكرا.", "I'm fine, thank you."),
        ("واحد، اثنان، ثلاثة.", "One, two, three."),
        ("مرحبا <SIL> بالعالم", "With explicit SIL token"),
    ]

    for ar_text, description in arabic_tests:
        print(f"\n{description}:")
        print(f"  Input: {ar_text}")
        words = phonemizer_ar.break_words_special(ar_text)
        print(f"  Words: {words}")
        print(f"  Count: {len(words)} tokens")

    # Test Chinese (CJK - character-based)
    print("\n" + "-" * 70)
    print("CHINESE TESTS (Character-based tokenization)")
    print("-" * 70)

    phonemizer_zh = Phonemizer(language='cmn', remove_noise_phonemes=False)

    chinese_tests = [
        ("你好世界", "Hello world"),
        ("这是测试。", "This is a test."),
        ("一，二，三。", "One, two, three."),
        ("你好 <SIL> 世界", "With explicit SIL token"),
    ]

    for zh_text, description in chinese_tests:
        print(f"\n{description}:")
        print(f"  Input: {zh_text}")
        words = phonemizer_zh.break_words_special(zh_text)
        print(f"  Words: {words}")
        print(f"  Count: {len(words)} tokens")

    # Test Thai (no-space language, character-based)
    print("\n" + "-" * 70)
    print("THAI TESTS (Character-based tokenization)")
    print("-" * 70)

    try:
        phonemizer_th = Phonemizer(language='th', remove_noise_phonemes=False)

        thai_tests = [
            ("สวัสดี", "Hello"),
            ("สวัสดีครับ", "Hello (polite)"),
        ]

        for th_text, description in thai_tests:
            print(f"\n{description}:")
            print(f"  Input: {th_text}")
            words = phonemizer_th.break_words_special(th_text)
            print(f"  Words: {words}")
            print(f"  Count: {len(words)} tokens")
    except Exception as e:
        print(f"\nThai test skipped (espeak may not support Thai): {e}")

    # Test punctuation handling
    print("\n" + "-" * 70)
    print("PUNCTUATION HANDLING TESTS")
    print("-" * 70)

    print("\nHindi punctuation:")
    hi_punct_tests = [
        "शब्द।",  # Word with Devanagari full stop
        "शब्द, शब्द।",  # With comma and period
        "शब्द!",  # With exclamation
        "शब्द?",  # With question mark
    ]

    for text in hi_punct_tests:
        words = phonemizer_hi.break_words_special(text)
        print(f"  {text:20s} -> {words}")

    print("\nArabic punctuation:")
    ar_punct_tests = [
        "كلمة.",  # Word with period
        "كلمة، كلمة.",  # With comma and period
        "كلمة!",  # With exclamation
        "كلمة؟",  # With Arabic question mark
    ]

    for text in ar_punct_tests:
        words = phonemizer_ar.break_words_special(text)
        print(f"  {text:20s} -> {words}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
if __name__ == "__main__":

    simple_test_en()

    simple_test_2()
    exit()