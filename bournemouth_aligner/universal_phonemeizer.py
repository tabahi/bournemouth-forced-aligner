# pip install requests phonemizer
# apt-get install espeak-ng

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from unicodedata import normalize
import re
import json
from .mapper66 import phoneme_mapper, phoneme_mapped_index, phoneme_groups_mapper, get_compound_phoneme_mapping # maps IPA phonemesto a smaller set of phonemes#


SIL_PHN="SIL"
assert(SIL_PHN in phoneme_mapped_index)
noise_PHN="noise"
assert(noise_PHN in phoneme_mapped_index)
noise_group = phoneme_groups_mapper[phoneme_mapped_index[noise_PHN]]
SIL_token = phoneme_mapped_index[SIL_PHN]



class Phonemizer:

    def __init__(self, language=None, remove_noise_phonemes=True):
        '''
        Initialize the phonemizer with a specific language backend and options.
        
        Args:
        language (str): Language code for the phonemizer backend (e.g., 'en-us').
        remove_noise_phonemes (bool): If True, noise phonemes will be removed from the output.
        - if language=None, then set_backend(language='en') must be called before phonemize_sentence
        - remove_noise_phonemes: remove noise phonemes from the phoneme list that includes possibly valid but unknown phonemes
        '''
        
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

        self.phonemes_ipa_key = "ipa"
        self.phonemes_key = "ph66"
        self.phoneme_groups_key = "pg16"
        
        for key, value in phoneme_mapper.items():
            self.phoneme_vocab_mapped_counts.update({value: 0})
            
        #print(f"Mapping: {len(phoneme_mapper)} phonemes onto {len(self.phoneme_vocab_mapped_counts)} phonemes")
        
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
        print(f"Setting backend for language: {language}")
        self.backend = EspeakBackend(language=language, preserve_punctuation=False, tie=False, with_stress=False)

    def break_words(self, single_sentence):
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
    
    def phonemize_sentence(self, sentence):
        '''
        Phonemize a single sentence and return a segment_out dict with phoneme indices and groups.
        
        Args:
        - sentence (str): The input sentence to phonemize.
        Returns:
        - segment_out dict with keys:
            - text: original sentence
            - ipa: list of phonemes in IPA format
            - ph66: list of phoneme class indices (mapped to phoneme_mapped_index)
            - pg16: list of phoneme group indices (mapped to phoneme_groups_mapper)
            - words: list of words corresponding to the phonemes
            - word_num: list of word indices corresponding to the phonemes
        '''

        words = self.break_words(sentence)
        
        # Phonemize the words in this segment
        phonemized_words = self.backend.phonemize(
            words, 
            separator=Separator(phone='|', word=None, syllable=None), 
            strip=True, 
            njobs=1
        )
        

        
        ph_sil_idx = self.phoneme_index[SIL_PHN]
        ph_sil_group = phoneme_groups_mapper[ph_sil_idx]

        segment_out = {self.phonemes_key:[], self.phoneme_groups_key:[], self.phonemes_ipa_key:[], "word_num":[], "words":[], "text": sentence}
        # Process each word in the segment
        for word_idx, (word, phonemized) in enumerate(zip(words, phonemized_words)):
            
            
            if word == "<SIL>":
                # Special case for <SIL> which is treated as silence
                segment_out[self.phonemes_key].append(ph_sil_idx)
                segment_out[self.phoneme_groups_key].append(ph_sil_group)
                segment_out[self.phonemes_ipa_key].append(SIL_PHN)
                segment_out["word_num"].append(word_idx)
                segment_out["words"].append("<sil>")  # Placeholder for silence
                self.sil_count += 1
                continue

            word_phonemes = phonemized.split('|')
            
            # Filter out empty phonemes
            word_phonemes = [ph for ph in word_phonemes if ph != ""]
            
            
            
            has_valid_phonemes = False
            
            # Process each phoneme
            for ph in word_phonemes:
                ph = ph.strip()
                ph = normalize('NFC', ph)
                
                # Update phoneme count statistics
                if ph not in self.phoneme_vocab_counts:
                    self.phoneme_vocab_counts[ph] = 1
                else:
                    self.phoneme_vocab_counts[ph] += 1
                
                # Handle unmapped phonemes
                if ph not in phoneme_mapper:
                    if ph not in self.unmapped_phonemes:
                        print(f"Warning: phoneme not found in mapper: {ph}")
                        self.unmapped_phonemes.update({ph: 1})
                    else:
                        self.unmapped_phonemes[ph] += 1
                    ph = noise_PHN
                
                # Get compound phoneme mapping
                ph = get_compound_phoneme_mapping(ph)
                if ph not in self.phoneme_index:
                    ph = noise_PHN
                
                # Skip noise phonemes if configured to do so
                if self.remove_noise_phonemes and ph == noise_PHN:
                    self.noise_count += 1
                    continue
                
                # At this point, we have a valid phoneme that will be included
                has_valid_phonemes = True
                
                # Get phoneme index and group
                ph_idx = self.phoneme_index[ph]
                ph_group = phoneme_groups_mapper.get(ph_idx, noise_group)  # Default to noise_group if not found
                
                # Add to segment output
                segment_out[self.phonemes_key].append(ph_idx)
                segment_out[self.phoneme_groups_key].append(ph_group)
                segment_out[self.phonemes_ipa_key].append(ph)
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
            assert len(segment_out[self.phonemes_key]) == len(segment_out[self.phoneme_groups_key]) == len(segment_out["word_num"])
            assert max(segment_out["word_num"]) + 1 == len(segment_out["words"])
            # Check that all words with phonemes are accounted for
            

        
        return segment_out

def simple_test():
    

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


if __name__ == "__main__":

    simple_test()
    exit()