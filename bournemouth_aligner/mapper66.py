# this is a prepared index generated from the create_new_index() function

phoneme_mapped_index = {
    # Special token
    'SIL': 0,
    
    # High front vowels and commonly confused similar vowels
    'i': 1,        # High front unrounded
    'i:': 2,       # Long high front unrounded
    'ɨ': 3,        # High central (grouped here due to high confusion with 'i')
    'ɪ': 4,        # Near-high front unrounded
    
    # Mid front vowels
    'e': 5,        # Mid front unrounded
    'e:': 6,       # Long mid front unrounded
    'ɛ': 7,        # Open-mid front unrounded
    
    # Central vowels
    'ə': 8,        # Schwa (mid central)
    'ɚ': 9,        # R-colored schwa
    'ʌ': 10,       # Open-mid back unrounded
    
    # Back vowels
    'u': 11,       # High back rounded
    'u:': 12,      # Long high back rounded
    'ʊ': 13,       # Near-high back rounded
    'ɯ': 14,       # High back unrounded
    'o': 15,       # Mid back rounded
    'o:': 16,      # Long mid back rounded
    'ɔ': 17,       # Open-mid back rounded
    
    # Low vowels
    'a': 18,       # Open central/front unrounded
    'a:': 19,      # Long open central/front unrounded
    'æ': 20,       # Near-open front unrounded
    
    # Front rounded vowels
    'y': 21,       # High front rounded
    'ø': 22,       # Mid front rounded
    
    # Diphthongs
    'aɪ': 23,      # Open central to high front
    'eɪ': 24,      # Mid front to high front
    'aʊ': 25,      # Open central to high back
    'oʊ': 26,      # Mid back to high back
    'ɔɪ': 27,      # Open-mid back to high front
    
    # Stops (organized by place of articulation)
    'p': 28,       # Voiceless bilabial
    'b': 29,       # Voiced bilabial
    't': 30,       # Voiceless alveolar
    'd': 31,       # Voiced alveolar
    'k': 32,       # Voiceless velar
    'g': 33,       # Voiced velar
    'q': 34,       # Voiceless uvular
    
    # Affricates and related sibilant fricatives (grouped by similarity)
    'ts': 35,      # Voiceless alveolar affricate
    's': 36,       # Voiceless alveolar fricative
    'z': 37,       # Voiced alveolar fricative
    'tʃ': 38,      # Voiceless postalveolar affricate
    'dʒ': 39,      # Voiced postalveolar affricate
    'ʃ': 40,       # Voiceless postalveolar fricative
    'ʒ': 41,       # Voiced postalveolar fricative
    'ɕ': 42,       # Voiceless alveolo-palatal fricative
    
    # Other fricatives (organized by place)
    'f': 43,       # Voiceless labiodental
    'v': 44,       # Voiced labiodental
    'θ': 45,       # Voiceless dental
    'ð': 46,       # Voiced dental
    'ç': 47,       # Voiceless palatal
    'x': 48,       # Voiceless velar
    'ɣ': 49,       # Voiced velar
    'h': 50,       # Voiceless glottal
    'ʁ': 51,       # Voiced uvular
    
    # Nasals (organized by place)
    'm': 52,       # Bilabial
    'n': 53,       # Alveolar
    'ɲ': 54,       # Palatal
    'ŋ': 55,       # Velar
    
    # Liquids and approximants
    'l': 56,       # Alveolar lateral
    'ɭ': 57,       # Retroflex lateral
    'ɾ': 58,       # Alveolar tap
    'ɹ': 59,       # Alveolar approximant
    'j': 60,       # Palatal approximant
    'w': 61,       # Labial-velar approximant
    
    # Palatalized consonants
    'tʲ': 62,      # Palatalized t
    'nʲ': 63,      # Palatalized n
    'rʲ': 64,      # Palatalized r
    'ɭʲ': 65,      # Palatalized retroflex lateral
    
    # Special token
    'noise': 66
}


phoneme_groups_mapper = {0: 0, 1: 1, 2: 1, 3: 3, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 4, 19: 4, 20: 4, 21: 1, 22: 1, 23: 5, 24: 5, 25: 5, 26: 5, 27: 5, 28: 6, 29: 7, 30: 6, 31: 7, 32: 6, 33: 7, 34: 6, 35: 10, 36: 8, 37: 9, 38: 10, 39: 11, 40: 8, 41: 9, 42: 8, 43: 8, 44: 9, 45: 8, 46: 9, 47: 8, 48: 8, 49: 9, 50: 8, 51: 9, 52: 12, 53: 12, 54: 12, 55: 12, 56: 13, 57: 13, 58: 14, 59: 14, 60: 15, 61: 15, 62: 6, 63: 12, 64: 14, 65: 13, 66: 16}


phoneme_groups_index = {'SIL': 0, 'front_vowels': 1, 'central_vowels': 2, 'back_vowels': 3, 'low_vowels': 4, 'diphthongs': 5, 'voiceless_stops': 6, 'voiced_stops': 7, 'voiceless_fricatives': 8, 'voiced_fricatives': 9, 'voiceless_affricates': 10, 'voiced_affricates': 11, 'nasals': 12, 'laterals': 13, 'rhotics': 14, 'glides': 15, 'noise': 16}



# vocab counts from train_100h*8_langs, OpenSLR-MLS data 100hours each
complete_vocab = {'SIL': 4914031, 'a': 1604572, 'n': 1501496, 't': 1345451, 's': 1242856, 'i': 1207390, 'e': 985568, 'o': 850470, 'm': 840466, 'l': 840200, 'r': 825931, 'd': 821689, 'k': 814493, 'ɛ': 700232, 'p': 607786, 'ə': 492948, 'v': 432914, 'j': 430514, 'u': 422499, 'ɾ': 419723, 'b': 413539, 'ɑ': 399576, 'ɔ': 344276, 'ʌ': 334025, 'ɪ': 294767, 'f': 292469, 'z': 286215, 'ɡ': 267606, 'ʃ': 249935, 'ɐ': 247989, 'w': 222719, 'ʊ': 200737, 'h': 189391, 'ʁ': 161395, 'ð': 159285, 'ɨ': 157902, 'x': 151422, 'eː': 141335, 'y': 138328, 'iː': 135167, 'ŋ': 118468, 'aɪ': 104170, 'ts': 96727, 'ɹ': 96111, 'æ': 83801, 'tʃ': 83484, 'θ': 81846, 'ʒ': 81251, 'uː': 79788, 'aː': 69487, 'ɕ': 68197, 'β': 67991, 'oː': 67190, 'ɑː': 66515, 'ɣ': 64902, 'eɪ': 63049, 'tʲ': 60946, 'ø': 58520, 'ɭ': 58455, 'nʲ': 56439, 'dʒ': 54175, 'ɑ̃': 53516, 'aʊ': 53297, 'q': 49319, 'ɲ': 48479, 'rʲ': 46853, 'ɭʲ': 45521, 'ɔ̃': 44985, 'ɯ': 43339, 'sʲ': 38535, 'ɲʲ': 38014, 'ɒ': 37682, 'vʲ': 37231, 'ʎ': 37145, 'ç': 35610, 'ʋ': 32705, 'ɚ': 32019, 'tɕ': 32006, 'mʲ': 31189, 'dʲ': 29989, 'ɜ': 27714, 'ja': 27523, 'ʔ': 26931, 'oʊ': 26398, 'ɑɨ': 24604, 'tʃʲ': 24301, '1': 23766, 'dʑ': 22131, 'ɛɪ': 21834, 'tː': 21127, 'ᵻ': 20631, 'ɛ̃': 20508, 'uɨ': 20396, 'ɫ': 19763, 'ɬ': 19662, 'ʑ': 18169, 'œ': 17838, 'oɪ': 16897, 'ɔɨ': 16178, 'ɔː': 15555, 'ɐ̃': 15510, 'ɨː': 15065, 'ɜː': 14661, 'ju': 14500, 'pʲ': 14429, 'aɨ': 13971, 'əl': 13858, 'ɵ': 13512, 'kʲ': 13204, 'ss': 12685, 'ɐ̃ʊ̃': 12237, 't[': 12211, 'əɪ': 12209, 'ɑːɹ': 10053, 'bʲ': 9958, 'd[': 9275, 'yː': 9033, 'eʊ': 8931, 'ɨu': 7653, 'ɡʲ': 7633, 'ɔːɹ': 7072, '(en)': 6245, 'œy': 6018, 'kː': 5801, 'əɨ': 5629, 'ɔø': 5613, 'oːɹ': 5458, 'u"': 5356, 'fʲ': 5315, 'pː': 5307, 'ɛɹ': 5299, 'ɪː': 5068, '??': 5027, 'ɛː': 4894, 'øː': 4796, 'ɔɪ': 4603, 'dZ': 4345, 'ɪu': 4320, 'c': 4200, 'S': 4197, 'ʕ': 4050, '(fr)': 3885, 'ʌʊ': 3787, 'tS': 3731, 'oe': 3730, 'iə': 3654, 'dʒː': 3441, 'ɪɹ': 3101, 'r̝̊': 3099, 'bː': 3051, 'ɟ': 2761, 'uɪ': 2632, 'ʊɹ': 2589, 'tʃː': 2564, 'ħ': 2318, 'ũ': 2158, 's^': 2080, 't̪': 1778, 'r̝': 1702, 'ɪ^': 1651, 'tsː': 1558, 'dzː': 1479, 'r̩': 1272, 'u:': 1189, 'aɪɚ': 1180, '(de)': 1166, 's̪': 1070, 'dz': 1027, 'iʊ': 940, 'aɪə': 928, 'dˤ': 920, 'χ': 845, 'æi': 819, 'œ̃': 766, '(it)': 719, 'ɑ:': 715, 'o:': 623, 'n̩': 587, 'l̩': 536, 'æː': 534, 'dː': 532, 'õ': 491, 'N': 490, 'y:': 415, 'pf': 342, 'əʊ': 342, 'ʝ': 323, 't^': 288, 'oe:': 257, '(nl)': 237, 'ɛʊ': 237, '(ptpt)': 196, 'e:': 183, 'eə': 144, 'd^': 131, 'i.ː': 129, 'yʊ': 101, 't^ː': 79, 'nl': 76, '(fa)': 65, 'æiː': 64, 'yi': 63, '(es)': 61, 'dʒʲ': 47, 'qː': 43, '(ru)': 40, 'ɡː': 37, 'ɪuː': 37, 'ʊə': 34, 'X': 31, 'a.ː': 31, 'u.ː': 31, 'rr': 25, 'mb': 25, 'ɵː': 24, 'd̪w': 23, 'ʂ': 22, '(tt)': 22, 'dm': 20, 'daː': 17, 'əː': 17, 'it': 17, 'ɡd': 17, 'mj': 16, 'db': 16, 'wb': 16, 'iːː': 15, 'mt': 15, 'ɑk': 15, 'i:': 15, 'da': 14, 'nb': 14, 'eð': 13, 'mx': 13, 'maː': 13, 'tk': 13, 'niː': 13, 'rb': 13, 'mh': 13, 'dˤdˤ': 13, 'fm': 13, 'nm': 11, 'eːh': 11, 'mtʃ': 11, 'ma': 11, 'ʊːt': 11, 'aːn': 11, 'iːe': 11, 'im': 10, 'eːb': 10, 'np': 10, 'aɪaɪ': 9, 'ʃj': 9, 'eːt': 9, 'jː': 8, 'mv': 8, 'ae': 8, 'ed': 8, 'nn': 8, 'dtʃ': 8, 'ɑh': 8, 'tr': 7, 'sb': 7, 'tn': 7, 'lx': 7, 'eːa': 7, 'il': 7, 'ɑj': 7, 'ɑaː': 7, 'oːs': 7, 'eːq': 7, 'ah': 7, 'bb': 7, 'or': 7, 'ɑm': 7, 'kt': 7, 'as': 7, 'eβ': 6, 'eːd': 6, 'ob': 6, 'eːuː': 6, 'rtʃ': 6, 'ʃd': 6, 'md': 6, 'duː': 6, 'eːaː': 6, 'ɡz': 6, 'hx': 6, 'lk': 6, 'eːf': 6, 'ɑq': 6, 'nk': 6, 'nx': 6, 'nj': 6, 'dɔ': 6, 'is': 6, 'aɪp': 6, 'əm': 6, 'laː': 5, 'tb': 5, 'aa': 5, 'zm': 5, 'ntʃ': 5, 'ep': 5, 'bh': 5, 'lh': 5, 'do': 5, 'eːp': 5, 'miː': 5, 'tuː': 5, 'lm': 5, 'mr': 5, 'sz': 5, 'nv': 4, 'in': 4, 'ɑuː': 4, 'iaː': 4, 'oːb': 4, 'uːm': 4, 'naː': 4, 'eːs': 4, 'itʃ': 4, 'eːv': 4, 'az': 4, 'ha': 4, 'dɡ': 4, 'hb': 4, 'mf': 4, 'ɑn': 4, 'ia': 4, 'lb': 4, 'na': 4, 'ld': 4, 'd̪': 4, 'ndʒ': 3, 'uːk': 3, 'uːb': 3, 'mo': 3, 'ɡh': 3, 'mk': 3, 'dk': 3, 'eːtʃ': 3, 'dp': 3, 'tiː': 3, 'bv': 3, 'ta': 3, 'th': 3, 'ip': 3, 'eːj': 3, 'eːx': 3, 'mz': 3, 'eːz': 3, '(pl)': 2, 'diː': 2, 'eh': 2, 'biː': 2, 'mʃ': 2, 'mn': 2, 'dʒv': 2, 'hʃ': 2, 'liː': 2, 'mɡ': 2, 'jiː': 2, 'ms': 2, 'ʃz': 2, 'ne': 2, 'on': 2, 'raː': 2, 'nh': 2, 'eːʃ': 2, 'rm': 2, 'ʊː': 2, 'laɪ': 1, 'ɑa': 1, 'td': 1, 'eːe': 1, 'mp': 1, 'ɑt': 1, 'nr': 1, 'me': 1, 'ɑo': 1, 'ik': 1, 'iːv': 1, 'dh': 1, 'eːɑ': 1, 'dl': 1, 'ʃdʒ': 1, 'ns': 1}




base_phonemes = [
    # New:
    'x', 'ç', 'ɣ', 'ʁ', 'ts', 'tʃ', 'ʌ',  'ɭ',
    # Stops
    'p', 'b', 't', 'd', 'k', 'ɡ', 'ʔ',
    
    # Fricatives
    'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h', 'ʂ', 'ʐ', 'ɕ', 'ʑ', 'ʁ',
    
    # Affricates
    'ts', 'dz', 'tɕ', 'dʑ', 'tʃ', 'dʒ', 'pf',
    
    # Nasals
    'm', 'n', 'ŋ',
    
    # Liquids/Glides
    'l', 'ɫ', 'ɹ', 'j', 'w', 'ɾ',
    
    # Palatalized
    'nʲ', 'lʲ', 'rʲ',
    
    # Vowels
    'i', 'iː', 'ɪ', 'e', 'eː', 'ɛ', 'æ',
    'a', 'aː', 'ɑː',
    'o', 'ɔː', 'ʊ',
    'u', 'uː',
    'ə', 'ɚ', 'ʌ', 'ɨ', 'ɤ', 'ɯ',
    'œ', 'ø', 'øː', 'ʏ', 'yː',
    
    # Diphthongs
    'eɪ', 'aɪ', 'ɔɪ', 'oʊ', 'aʊ',
    
    # Other
    'ɒ', 'ɜː', 'ɲ',

    'noise', 'SIL'
]

phoneme_mapping = {
    
    # Core vowels - simplified based on confusion patterns
    'ə': 'ə', 
    #'ʌ': 'ə',  # Merge due to high confusion
    'ʌ': 'ʌ', # didn't work well before but still keep it
    'ɪ': 'ɪ', 'i': 'i', 
    'iː': 'i:',
    'ʊ': 'ʊ',
    'u': 'u',
    'uː': 'u:',
    'ɛ': 'ɛ', 'e': 'e', 'eː': 'e:',
    'ɔː': 'ɔ', 'ɔ': 'ɔ',
    #'ɒ': 'ɒ', # Merge to 'a' due to 100% wrong predictions in confusion matrix (23 Jan)
    'ɒ': 'a',
    'æ': 'æ', # DO NOT merge
    
    'ɑː': 'a:', 
    'ɑ': 'a', 
    'a': 'a',
    'ɜː': 'ʌ',
    'ɜ': 'ʌ',
    'ɚ': 'ɚ',
    'o': 'o',
    'ɨ': 'ɨ',
    
    # Common diphthongs - keep distinct ones
    'eɪ': 'eɪ', 'aɪ': 'aɪ', 'ɔɪ': 'ɔɪ',
    'aʊ': 'aʊ', 'oʊ': 'oʊ',
    
    # Less common diphthongs - map to similar common ones
    'ʌʊ': 'aʊ', 'eʊ': 'aʊ', 'ɛʊ': 'aʊ', 'əʊ': 'oʊ',
    'ɛɪ': 'eɪ', 'ʊɪ': 'aɪ', 'ea': 'eɪ',
    'aʊ̯': 'aʊ', 'aɪ̯': 'aɪ', 'ɔʏ̯': 'ɔɪ',
    
    # Core consonants
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd',
    'k': 'k', 'g': 'g', 'm': 'm', 'n': 'n',
    'ŋ': 'ŋ', 'f': 'f', 'v': 'v', 'θ': 'θ',
    'ð': 'ð', 's': 's', 'z': 'z', 'ʃ': 'ʃ',
    'ʒ': 'ʒ', 'h': 'h', 'l': 'l', 'ɹ': 'ɹ',
    'j': 'j', 'w': 'w', 'ɲ': 'ɲ', 'ɾ': 'ɾ',
    
    # Consonant mergers based on confusion
    # 'ɣ': 'g',      # Merge with closest stop
    'ɣ': 'ɣ',  # emprically confused but will keep it
    #'ʁ': 'ɹ',      # Map to rhotic
    'ʁ': 'ʁ',
    'r': 'ɹ',      # Map to rhotic
    #'x': 'h',      # Map to closest fricative
    'x': 'x',
    #'ç': 'ʃ',      # Map to closest fricative
    #'ç': 's',      # Based on empirical confusion
    'ç': 'ç',
    'ʂ': 'ʃ',      # Map to closest fricative
    'ʐ': 'ʒ',      # Map to closest fricative
    #'ɕ': 'ʃ',      # Map to closest fricative
    'ɕ': 'ɕ',       # keep it
    'ʑ': 'ʒ',      # Map to closest fricative
    
    # Simplify affricates to their primary component
    #'ts': 't',
    'ts': 'ts',
    'dz': 'dʒ',
    #'tʃ': 'ʃ',
    'tʃ': 'tʃ',
    #'dʒ': 'ʒ',
    'dʒ': 'dʒ',
    'tɕ': 'tʃ',
    'dʑ': 'dʒ',
    'pf': 'f',
    
    #'tʲ': 't', 
    'tʲ': 'tʲ',  # high freuqncy, keep it
    #'nʲ': 'n', 
    'nʲ': 'nʲ', # high freuqncy, keep it
    #'rʲ': 'ɹ',
    'rʲ': 'rʲ', # high freuqncy, keep it
    # Remove palatalization
    'lʲ': 'l',  
    'dʲ': 'd', 'sʲ': 's', 'vʲ': 'v',
    'fʲ': 'f', 'mʲ': 'm',
    'pʲ': 'p', 'kʲ': 'k', 'bʲ': 'b',
    'ɲʲ': 'ɲ', 'dʒʲ': 'dʒ',
    
    # Simplify geminate consonants
    'tː': 't', 'dː': 'd', 'kː': 'k',
    'gː': 'g', 'pː': 'p', 'bː': 'b',
    'fː': 'f', 'vː': 'v', 'sː': 's',
    'zː': 'z', 'ʃː': 'ʃ', 'ʒː': 'ʒ',
    'mː': 'm', 'nː': 'n', 'ŋː': 'ŋ',
    'lː': 'l', 'rː': 'ɹ', 'jː': 'j',
    
    # Nasal vowels to oral counterparts
    'ɑ̃': 'a', 'ɛ̃': 'ɛ', 'ɔ̃': 'ɔ',
    'ũ': 'u', 'õ': 'oʊ', 'ɐ̃': 'ʌ',
    
    # R-colored vowels
    'ɑːɹ': 'ɚ', 'ɔːɹ': 'ɚ',
    'ʊɹ': 'ɚ', 'ɪɹ': 'ɚ', 'ɛɹ': 'ɚ',
    'oːɹ': 'ɚ',
    
    # Vowel sequences
    'ia': 'i:', 'ua': 'u:',
    'ɔø': 'ɔ', 'iːɛ': 'i:',
    'ʊə': 'ʊ', 'iə': 'i:',
    'eə': 'ɛ',
    
    # Common sequences
    # 'əl': 'əl',  # Keep this distinct sequence
    #'əl': 'o', # based on empirical confusion. theoretically, this should be merged with 'l' or 'e' but it's most confused with 'o'
    'əl': 'l',
    'n̩': 'n',
    'ʃf': 'ʃ',
    'eð': 'ð',
    'ns': 'n',
    'nd': 'n',
    'ʃts': 'ts',
    
    # Special symbols
    'SIL': 'SIL',
    'noise': 'noise',   # noise will be ignored by the model. CTC will take it as blank token.
    '': 'SIL',
    'ʔ': 'noise',
    
    # Language markers to silence
    '(en)': 'SIL', '(es)': 'SIL', '(fr)': 'SIL',
    '(de)': 'SIL', '(it)': 'SIL', '(nl)': 'SIL',
    '(pl)': 'SIL', '(ru)': 'SIL', '(ptpt)': 'SIL',
    
    # Error cases to noise
    '??': 'noise', 'uk': 'noise', 'it': 'noise',
    'ɡd': 'noise', 'rd': 'noise', 'as': 'noise',
    'up': 'noise', 'os': 'noise', 'kf': 'noise',
    '1': 'noise', 'ʃd': 'noise', 'ʃz': 'noise',
    'ʃn': 'noise',



    # Vowels
    'y': 'y',        # Map to existing long form
    'yː': 'y',       # Keep distinct high front rounded vowel
    'œ': 'ø',         # Map to closest unrounded vowel
    'ø': 'ø',        # Map to long version
    'øː': 'ø',       # Keep distinct mid front rounded vowel
    'ɐ': 'ʌ',         # Map to schwa
    'aː': 'a:',       # Keep long a
    #'oː': 'ɔ',       # Map to similar long vowel
    'oː': 'o:',       # Keep distinct long o
    'ɛː': 'ɛ',        # Map to base form
    'ɪː': 'i:',       # Map to similar long vowel
    'ɵ': 'ʊ',         # Map to closest vowel
    'ᵻ': 'ɪ',         # Map to similar vowel
    
    # Double vowels (map to their long counterparts)
    'aa': 'a',
    'ɐɐ': 'a',
    'ææ': 'æ',
    
    # Diphthongs
    'yʊ': 'u',       # Map to similar monophthong
    'œy': 'ɔɪ',       # Map to similar diphthong
    'uɪ': 'aɪ',       # Map to existing diphthong
    'oɪ': 'ɔɪ',       # Map to similar diphthong
    'iʊ': 'u',       # Map to similar monophthong
    'aɪə': 'aɪ',      # Map to base diphthong
    'aɪɚ': 'aɪ',      # Map to base diphthong
    
    # Nasal vowels
    'ɐ̃ʊ̃': 'aʊ',      # Map to oral diphthong
    'œ̃': 'ɛ',         # Map to oral vowel
    
    # Consonants
    'ʝ': 'j',         # Map to similar approximant
    'ɟ': 'ʒ',        # Map to similar affricate
    'ʋ': 'v',         # Map to similar fricative
    'd̪': 'd',         # Map dental to alveolar
    't̪': 't',         # Map dental to alveolar
    'ɬ': 'l',         # Map to plain lateral
    'ʎ': 'l',         # Map to plain lateral
    'β': 'v',         # Map to similar fricative
    'ɡ': 'g',         # Standardize to 'g'
    
    # Geminate consonants
    'ɡː': 'g',        # Map to single consonant
    'tsː': 'ts',      # Map to single affricate
    'dzː': 'd',      # Map to single affricate
    #'tʃː': 'ʃ',      # Map to single affricate
    'tʃː': 'tʃ',      # Map to single affricate
    'dʒː': 'dʒ',      # Map to single affricate
    'ss': 's',        # Map to single consonant
    
    # Palatalized consonants
    'ɡʲ': 'g',        # Map to plain consonant
    
    # Sequences
    'dɔ': 'noise',     # Map unusual sequence to noise


    # These are found (with counts) in Google MSWC data, but not in the OpenSLR-MLS data
    # Complex sequences with frequency counts
    'ja': 'j',      # Common sequence (36,809) -> simplify to first component
    'ju': 'j',      # Common sequence (19,620) -> simplify to first component  
    'tʃʲ': 'tʃ',     # Common palatalized affricate (32,707) -> map to fricative
    #'ɭ': 'l',       # Very common retroflex lateral (78,504) -> map to alveolar
    'ɭ': 'ɭ',
    'ɭʲ': 'ɭʲ',      # Common palatalized retroflex (61,298) -> map to plain lateral
    'u"': 'u',     # Quote variant (7,265) -> normalize to standard long u
    'ɪ^': 'ɪ',      # Rare diacritic variant (2,222) -> remove diacritic
    'sz': 's',      # Rare sequence (5) -> simplify to first component
    #'q': 'k',       # Common uvular stop (75,838) -> map to velar
    'q': 'q',     # keep it EVEN though it's relively rare (45k)
    #'qː': 'k',      # Rare long uvular (103) -> map to velar
    'qː': 'q',
    'r̝̊': 'ɹ',      # Rare trilled/fricative r (3,099) -> map to approximant
    'r̝': 'ɹ',      # Rare variant (1,702) -> map to approximant
    'r̩': 'ɹ',      # Rare syllabic (1,272) -> map to approximant
    'l̩': 'l',      # Rare syllabic (536) -> map to standard lateral
    'c': 'k',       # Uncommon palatal stop (4,195) -> map to velar

    # Vowel sequences
    'uɨ': 'ɨ',     # Common sequence (20,396) -> map to monophthong
    'aɨ': 'aɪ',     # Common sequence (13,971) -> map to similar diphthong
    'ɨu': 'u:',     # Less common (7,653) -> map to monophthong
    'ɪu': 'u:',     # Uncommon (4,320) -> map to monophthong
    'ɨː': 'ɨ',      # Common variant (15,065) -> remove length marker
    'ɑɨ': 'aɪ',     # Common sequence (24,604) -> map to diphthong
    'əɪ': 'eɪ',     # Common sequence (12,209) -> map to similar diphthong
    'əɨ': 'ɨ',      # Less common (5,629) -> simplify to first component
    'ɔɨ': 'ɔɪ',     # Common sequence (16,178) -> map to similar diphthong
    'ɪuː': 'u:',    # Rare sequence (37) -> map to monophthong

    # Rare sequences: 1-5 occurrences ---------------------------------------
    # Some of the extremely rare consonant-consonant and vowel-consonant sequences map to 'noise' (i.e., ignored), most don't.
    
    # More nasal sequences
    'nm': 'n',    # was 'noise', map to alveolar nasal
    'nn': 'n',    # was 'noise', map to single nasal
    'mn': 'm',    # was 'noise', map to bilabial nasal
    'mm': 'm',    # was 'noise', map to single nasal
    'na': 'n',    # was 'noise', preserve nasal
    'maː': 'm',   # was 'noise', preserve nasal
    'mz': 'm',    # was 'noise', preserve nasal
    'ms': 'm',    # was 'noise', preserve nasal
    'mf': 'm',    # was 'noise', preserve nasal
    'mɡ': 'm',    # was 'noise', preserve nasal
    'mx': 'm',    # was 'noise', preserve nasal
    'mv': 'm',    # was 'noise', preserve nasal
    'mʃ': 'm',    # current mapping is good
    
    # Stop sequences
    'dk': 'd',    # was 'noise', preserve first stop
    'dp': 'd',    # was 'noise', preserve first stop
    'db': 'd',    # was 'noise', preserve first stop
    'td': 't',    # was 'noise', preserve first stop
    'tb': 't',    # was 'noise', preserve first stop
    'tn': 't',    # was 'noise', preserve stop
    
    # Long vowel sequences
    'eːs': 'e:',  # was 'noise', preserve long vowel
    'eːt': 'e:',  # was 'noise', preserve long vowel
    'eːp': 'e:',  # was 'noise', preserve long vowel
    'eːf': 'e:',  # current mapping is good
    'eːz': 'e:',  # current mapping is good
    'eːj': 'e:',  # current mapping is good
    'eːx': 'e:',  # current mapping is good
    'eːʃ': 'e:',  # current mapping is good
    'oːs': 'o:',  # current mapping is good
    'oːb': 'o:',  # current mapping is good
    
    # Vowel sequences
    'ɑj': 'aɪ',   # was 'noise', map to diphthong
    'ɑh': 'a',    # was 'noise', preserve vowel
    'ɑm': 'a',    # was 'noise', preserve vowel
    'ɑk': 'a',    # was 'noise', preserve vowel
    'ɑn': 'a',    # was 'noise', preserve vowel
    'ɑq': 'a',    # was 'noise', preserve vowel
    'ɑt': 'a',    # was 'noise', preserve vowel
    'ɑo': 'a',    # was 'noise', preserve first vowel
    'ɑa': 'a',    # was 'noise', preserve first vowel
    'ɑaː': 'a:',  # was 'noise', map to long vowel
    'ɑuː': 'aʊ',  # was 'noise', map to diphthong
    
    # Other sequences
    'dʒv': 'dʒ',  # current mapping is good
    'bv': 'b',    # was 'noise', preserve stop
    'bh': 'b',    # was 'noise', preserve stop
    'ɡh': 'g',    # was 'noise', preserve stop
    'ɡz': 'g',    # was 'noise', preserve stop
    'hx': 'x',    # current mapping is good
    'ʃj': 'ʃ',    # was 'noise', preserve fricative
    
    # Special cases
    '(fa)': 'SIL',  # current mapping is good
    'bb': 'b',      # current mapping is good

    'uːb': 'u:',
    'uːk': 'u:',
    'laɪ': 'noise',
    # ---------------------------------------  End of rare sequences

    # Vowels and length variants
    'əː': 'ə',         # Long schwa maps to schwa (index 18)
    'æː': 'æ',         # Long ash maps to ash (index 32)
    'æi': 'eɪ',        # Map to similar diphthong (index 23)
    'æiː': 'eɪ',       # Map to similar diphthong (index 23)
    'ɵː': 'ʊ',         # Long rounded vowel maps to nearest equivalent (index 22)
    #'ɯ': 'ʊ',         # Unrounded high back vowel maps to nearest equivalent (index 22)
    'ɯ': 'ɯ',

    # Alternative transcription formats
    'e:': 'e:',        # Long e
    'eː': 'e:',        # Normalize colon to IPA length mark (index 43)
    #'e:': 'e',          # NOT merged due to high confusion
    'o:': 'o:',
    'y:': 'y',        # Normalize colon to IPA length mark (index 39)
    'u:': 'u:',        # Normalize colon to IPA length mark (index 5)
    'i:': 'i:',        # Normalize colon to IPA length mark (index 12)
    'ɑ:': 'a',        # Normalize colon to IPA length mark (index 13)
    'oe:': 'ø',       # Normalize colon to IPA length mark (index 40)
    'oe': 'ø',        # Map to equivalent (index 40)
    
    # ASCII-based transcription variants
    'S': 's',          # ASCII variant of 's' (index 21)
    'N': 'n',          # ASCII variant of 'n' (index 11)
    'X': 'k',          # ASCII variant, typically representing 'k' (index 27)
    'tS': 'tʃ',         # ASCII variant of 'tʃ' (index 1)
    'dZ': 'dʒ',         # ASCII variant of 'dʒ' (index 2)
    
    # Special characters and diacritics
    't^': 't',         # Remove diacritic (index 4)
    's^': 's',         # Remove diacritic (index 21)
    'd^': 'd',         # Remove diacritic (index 9)
    't^ː': 't',        # Remove diacritic and length (index 4)
    't[': 't',         # Remove bracket notation (index 4)
    'd[': 'd',         # Remove bracket notation (index 9)
    
    # Arabic phonemes
    'ʕ': 'h',          # Voiced pharyngeal fricative maps to nearest fricative (index 37)
    'ħ': 'h',          # Voiceless pharyngeal fricative maps to 'h' (index 37)
    'dˤ': 'd',         # Pharyngealized 'd' maps to plain 'd' (index 9)
    's̪': 's',          # Dental 's' maps to plain 's' (index 21)
    'χ': 'x',          # Voiceless uvular fricative maps to 'h' (index 37)
    'dˤdˤ': 'd',       # Doubled pharyngealized 'd' maps to 'd' (index 9)
    'dd': 'd',         # ASCII variant of doubled/pharyngealized 'd' (index 9)
    
    # Dot notation variants
    'i.ː': 'i:',       # Normalize dot notation (index 12)
    'a.ː': 'a:',       # Normalize dot notation (index 13)
    'u.ː': 'u:',       # Normalize dot notation (index 5)
    
    # Lateral approximant variant
    'ɫ': 'l',          # Velarized lateral maps to plain 'l' (index 16)
    
    # Consonant sequences (map to noise)
    'kt': 'noise',     # Consonant sequence (index 50)
    'd̪w': 'noise',     # Consonant sequence (index 50)
    'wb': 'noise',     # Consonant sequence (index 50)
    'fm': 'noise',     # Consonant sequence (index 50)
    
    # Vowel-consonant sequences (map to noise)
    'ʊːt': 'noise',    # Vowel-consonant sequence (index 50)
    'aɪp': 'noise',    # Vowel-consonant sequence (index 50)
    'əm': 'noise',     # Vowel-consonant sequence (index 50)
    'aːn': 'a:',    # Vowel-consonant sequence (index 50)
    'iːe': 'i:',    # Vowel-vowel sequence (index 50)
    'yi': 'i:',     # Vowel-vowel sequence (index 50)
    
    # Language markers (map to SIL)
    '(tt)': 'SIL',      # Language marker (index 0)

    # Double long vowel - map to standard long vowel
    'iːː': 'i:',       # Excessive length mark, normalize to standard long i (index 12)
    
    # Doubled diphthong - map to single diphthong
    'aɪaɪ': 'aɪ',      # Repeated diphthong, map to single instance (index 7)
    
    # Consonant sequences - map to noise like other sequences
    'ndʒ': 'dʒ',    # Consonant cluster (index 50)
    'tr': 'noise',     # Consonant cluster (index 50)
    'eβ': 'noise',     # Vowel-consonant sequence (index 50)


    # Double palatalization - map to single palatalized form then apply existing mappings
    'ʂʲ': 'ʃ',         # Map palatalized retroflex to palato-alveolar (index 1)
    'nʲʲ': 'nʲ',        # Double palatalized nasal to plain nasal (index 11)
    'tsʲ': 'ts',        # Palatalized affricate follows affricate mapping (index 4)
    'xʲ': 'h',         # Palatalized velar fricative to h (index 37)
    'dʑʲ': 'dʒ',        # Palatalized voiced affricate to voiced palato-alveolar (index 2)
    'ɕʲ': 'ɕ',         # Palatalized alveolo-palatal to palato-alveolar (index 1)
    'tɕʲ': 'ʃ',        # Palatalized affricate to palato-alveolar (index 1)
    'tʲʲ': 'tʲ',        # Double palatalized stop to plain stop (index 4)
    'ʒʲ': 'ʒ',        # Palatalized palato-alveolar remains (index 2)
    'ʃʲʲ': 'ʃ',        # Double palatalized palato-alveolar remains (index 1)
    'tsʲʲ': 'ts',       # Double palatalized affricate to stop (index 4)
    'ɾʲʲ': 'ɾ',        # Double palatalized tap remains (index 48)
    'zʲʲ': 'z',        # Double palatalized fricative remains (index 36)
    'ɾʲ': 'rʲ',         # Palatalized tap remains (index 48)
    'ʃʲ': 'ʃ',         # Palatalized palato-alveolar remains (index 1)
    'mʲʲ': 'm',        # Double palatalized nasal to plain (index 28)
    'ʲ': 'noise',      # Isolated palatalization mark to noise (index 50)

    # Vowel sequences - map to nearest phoneme or diphthong
    'uo': 'oʊ',        # Map to nearest diphthong (index 24)
    'ee': 'i:',        # Map to long vowel (index 12)
    'ie': 'i:',        # Map to long vowel (index 12)
    'ai': 'aɪ',        # Map to standard diphthong (index 7)
    'ui': 'u:',        # Map to long vowel (index 5)
    'au': 'aʊ',        # Map to standard diphthong (index 8)
    'eɑ': 'ɛ',         # Map to nearest monophthong (index 6)
    'iu': 'u:',        # Map to long vowel (index 5)
    'auː': 'aʊ',       # Map to standard diphthong (index 8)
    'ei': 'eɪ',        # Map to standard diphthong (index 23)
    'eu': 'oʊ',        # Map to nearest diphthong (index 24)
    'aiː': 'aɪ',       # Map to standard diphthong (index 7)
    'iuː': 'u:',       # Map to long vowel (index 5)
    'eiː': 'eɪ',       # Map to standard diphthong (index 23)
    'euː': 'oʊ',       # Map to nearest diphthong (index 24)
    'ɔa': 'ɔ',        # Map to long vowel (index 3)
    'yɪ': 'y',        # Map to long vowel (index 39)
    'iɪ': 'i:',        # Map to long vowel (index 12)
    'eo': 'oʊ',        # Map to nearest diphthong (index 24)

    # Special notations
    'cː': 'k',         # Long palatal stop to velar (index 27)

    # All Chinese tonal patterns (with numbers) and complex sequences map to 'noise'
    # Examples:
    'iɜk': 'noise', 'onɡ5': 'noise', 'ts.': 'ts', 'ə5': 'noise',
    'ŋf': 'noise', 'u2': 'noise', 'oɜɕ': 'noise', 'iɜ': 'noise',

    # MLS-fr
    # Consonant sequences to noise
    'ls': 'noise',     # Lateral + fricative sequence maps to noise (50)
    'll': 'noise',     # Double lateral sequence maps to noise (50)
    
    # Vowel-consonant sequences to noise
    'øːl': 'noise',    # Long oe + lateral sequence maps to noise (50)
    'øːs': 'noise',    # Long oe + fricative sequence maps to noise (50)


    # from UCLA Phonetics Dataset

    # Syllabic consonants - map to their non-syllabic counterparts
    'h̩': 'h',      # Syllabic h to h (37)
    'ɹ̩': 'ɹ',      # Syllabic r to r (17)
    'ŋ̩': 'ŋ',      # Syllabic ng to ng (34)
    'ɫ̩': 'l',      # Syllabic dark l to l (16)
    'v̩': 'v',      # Syllabic v to v (15)
    'm̩': 'm',      # Syllabic m to m (28)

    # Aspirated consonants - map to unaspirated counterparts
    'pʰ': 'p',     # Aspirated p to p (25)
    'tʰ': 't',     # Aspirated t to t (4)
    'kʰ': 'k',     # Aspirated k to k (27)
    'sʰ': 's',     # Aspirated s to s (21)
    'ʃʰ': 'ʃ',     # Aspirated sh to sh (1)
    'cʰ': 'k',     # Aspirated c to k (27)
    't͡sʰ': 'ts',    # Aspirated ts to t (4)
    't͡ʃʰ': 'tʃ',    # Aspirated tsh to sh (1)
    'ɕʰ': 'ɕ',     # Aspirated alveolo-palatal to sh (1)

    # Labialized consonants - map to base consonants
    'tʷ': 't',     # Labialized t to t (4)
    'kʷ': 'k',     # Labialized k to k (27)
    'pʷ': 'p',     # Labialized p to p (25)
    'ʒʷ': 'ʒ',     # Labialized zh to zh (2)
    'xʷ': 'h',     # Labialized x to h (37)
    'dʷ': 'd',     # Labialized d to d (9)
    'bʷ': 'b',     # Labialized b to b (26)
    'mʷ': 'm',     # Labialized m to m (28)
    'ŋʷ': 'ŋ',     # Labialized ng to ng (34)
    
    # Retroflexes - map to closest non-retroflex
    'ʈ': 't',      # Retroflex t to t (4)
    'ɖ': 'd',      # Retroflex d to d (9)
    'ɳ': 'n',      # Retroflex n to n (11)
    'ɻ': 'ɹ',      # Retroflex r to r (17)
    'ɽ': 'ɾ',      # Retroflex flap to tap (48)

    # Breathy voiced - map to regular voiced
    'n̤': 'n',      # Breathy n to n (11)
    'b̤': 'b',      # Breathy b to b (26)
    'j̤': 'j',      # Breathy j to j (29)
    'a̤': 'a',     # Breathy a to long a (30)
    'i̤ː': 'i:',    # Breathy long i to long i (12)
    'o̤': 'o',      # Breathy o to o (44)
    'o̤ː': 'o:',     # Breathy long o to o (44)
    
    # Nasalized vowels - map to oral counterparts
    'ãː': 'a:',    # Nasalized long a to long a (30)
    'ẽ': 'e',      # Nasalized e to e (42)
    'ɪ̃': 'ɪ',      # Nasalized short i to short i (31)
    'ỹ': 'y',     # Nasalized y to long y (39)
    'õː': 'o:',     # Nasalized long o to o (44)
    'æ̃': 'æ',      # Nasalized ae to ae (32)
    'ʌ̃': 'ʌ',      # Nasalized wedge to schwa (18)
    'ə̃': 'ə',      # Nasalized schwa to schwa (18)
    'ã': 'a',     # Nasalized a to long a (30)
    'ĩ': 'i:',     # Nasalized i to long i (12)
    'ĩː': 'i:',    # Nasalized long i to long i (12)
    'ũː': 'u:',    # Nasalized long u to long u (5)
    
    # Affricates - map to primary component
    't͡s': 'ts',     # ts to t (4)
    't͡ʃ': 'tʃ',     # tsh to sh (1)
    'd͡ʒ': 'dʒ',     # dzh to zh (2)
    't͡ɬ': 't',     # tl to t (4)
    
    # Ejectives - map to non-ejective counterparts
    'tʼ': 't',     # Ejective t to t (4)
    'kʼ': 'k',     # Ejective k to k (27)
    'qʼ': 'q',     # Ejective q to k (27)
    'pʼ': 'p',     # Ejective p to p (25)
    'sʼ': 's',     # Ejective s to s (21)
    
    # Additional vowels
    'ʏ': 'ɪ',      # Near-close near-front rounded to short i (31)
    'ʏː': 'y',    # Long near-close near-front rounded to long y (39)
    'ʊː': 'ʊ',     # Long near-close near-back rounded to short u (22)
    'ɤ': 'ə',      # Close-mid back unrounded to schwa (18)
    'ɤː': 'ə',     # Long close-mid back unrounded to schwa (18)
    'œː': 'ø',    # Long open-mid front rounded to long oe (40)
    'ɯː': 'u:',    # Long close back unrounded to long u (5)
    'ɛ̤': 'ɛ',      # Breathy open-mid front unrounded to epsilon (6)
    
    # Short/reduced vowels
    'ĕ': 'e',      # Short e to e (42)
    'ă': 'a',     # Short a to long a (30)
    'ĭ': 'ɪ',      # Short i to short i (31)
    'ŏ': 'o',      # Short o to o (44)
    'ŭ': 'ʊ',      # Short u to short u (22)
    
    # Laryngealized/creaky vowels - map to regular vowels
    'ḛ': 'e',      # Creaky e to e (42)
    'ḭ': 'i',      # Creaky i to i (41)
    'o̰': 'o',      # Creaky o to o (44)
    'ɛ̰': 'ɛ',      # Creaky epsilon to epsilon (6)
    'a̰': 'a',     # Creaky a to long a (30)
    'ʊ̰': 'ʊ',      # Creaky upsilon to upsilon (22)
    
    # Additional consonants
    'ɦ': 'h',      # Voiced h to h (37)
    'ʍ': 'w',      # Voiceless w to w (47)
    'ɢ': 'g',      # Uvular g to g (10)
    'ɱ': 'm',      # Labiodental nasal to m (28)
    'ʔ': 'noise',  # Glottal stop to noise (50)
    'ɮ': 'z',      # Voiced lateral fricative to z (36)
    'ɸ': 'f',      # Bilabial fricative to f (20)
    
    # Co-articulated stops 
    'k͡p': 'k',    # was 'noise', map to velar stop as it's typically more salient
    'ɡ͡b': 'g',    # was 'noise', map to velar stop (voiced counterpart)
    'p͡t': 'p',    # was 'noise', map to first stop in sequence
    'b͡d': 'b',    # was 'noise', map to first stop in sequence
    
    # Lengthened consonants
    'ʔː': 'q',    # was 'noise', map to closest glottal/uvular stop in inventory
    'hː': 'h',    # was 'noise', map to plain glottal fricative

    'æ̆': 'æ',      # Short ae to ae (32)
    'ɜ̆': 'ə',     # Short epsilon to long epsilon (33)
    'ɔ̆': 'ʌ',     # Short open-o to long open-o (3)
    'ə̠': 'ʌ',       # Retracted schwa (when it appears in stressed positions)
    'ə̆': 'ə',      # Short schwa to schwa (18)
    'ɒː': 'a:',    # Long open-o to long open-o (3)
    
    # Aspirated and modified affricates
    'd͡ʒʰ': 'dʒ',    # Aspirated dzh to zh (2)
    't͡sʼ': 'ts',    # Ejective ts to t (4)
    't͡ʃʼ': 'tʃ',    # Ejective tsh to sh (1)
    't͡ɬʼ': 't',    # Ejective tl to t (4)
    't͡ʃʲ': 'tʃ',    # Palatalized tsh to sh (1)
    'd͡ʒʲ': 'dʒ',    # Palatalized dzh to zh (2)
    
    # Voiceless sonorants
    'e̥': 'e',      # Voiceless e to e (42)
    'ɲ̥': 'ɲ',      # Voiceless ny to ny (38)
    'm̥': 'm',      # Voiceless m to m (28)
    'n̥': 'n',      # Voiceless n to n (11)
    'l̥': 'l',      # Voiceless l to l (16)
    'r̥': 'ɹ',      # Voiceless r to r (17)
    'ŋ̥': 'ŋ',      # Voiceless ng to ng (34)
    'i̥': 'i',      # Voiceless i to i (41)
    'u̥': 'u:',     # Voiceless u to long u (5)
    'ʎ̥': 'l',      # Voiceless palatal l to l (16)

    # Long consonants
    'tʰː': 't',    # Long aspirated t to t (4)
    'çː': 'ç',     # Long palatal fricative to h (37)
    'xː': 'h',     # Long x to h (37)
    'ɟː': 'ʒ',     # Long palatal stop to zh (2)
    'l̪ː': 'l',     # Long dental l to l (16)
    'pʰː': 'p',    # Long aspirated p to p (25)
    'θː': 'θ',     # Long th to th (46)
    'ɲː': 'ɲ',     # Long ny to ny (38)
    'wː': 'w',     # Long w to w (47)

    # Modified velars
    'kʰʲ': 'k',    # Palatalized aspirated k to k (27)
    'kʼʲ': 'k',    # Palatalized ejective k to k (27)
    'qʰʷ': 'q',    # Labialized aspirated q to k (27)
    'kʰʷ': 'k',    # Labialized aspirated k to k (27)
    'kʷʰ': 'k',    # Labialized aspirated k to k (27)
    'kʷʼ': 'k',    # Labialized ejective k to k (27)
    'qʷ': 'q',     # Labialized q to k (27)
    'qʷʼ': 'q',    # Labialized ejective q to k (27)
    'qʰ': 'q',     # Aspirated q to k (27)
    'q̠': 'q',      # Retracted q to k (27)
    'ɢʲ': 'g',     # Palatalized uvular g to g (10)
    'ɡʷ': 'g',     # Labialized g to g (10)

    # Rhotic vowels
    'e˞': 'ɚ',     # Rhotacized e to schwar (14)
    'a˞': 'ɚ',     # Rhotacized a to schwar (14)
    'o˞': 'ɚ',     # Rhotacized o to schwar (14)
    'u˞': 'ɚ',     # Rhotacized u to schwar (14)
    'i˞': 'ɚ',     # Rhotacized i to schwar (14)

    # Nasalized variants
    'ɛ̃ː': 'ɛ',     # Long nasalized epsilon to epsilon (6)
    'ʊ̃': 'ʊ',      # Nasalized upsilon to upsilon (22)
    'z̃': 'z',      # Nasalized z to z (36)
    'j̃': 'j',      # Nasalized j to j (29)
    'w̃': 'w',      # Nasalized w to w (47)
    'ʊ̰̃': 'ʊ',      # Creaky nasalized upsilon to upsilon (22)
    'æ̃ː': 'æ',     # Long nasalized ae to ae (32)
    'ɔ̃ː': 'ɔ',    # Long nasalized open-o to long open-o (3)
    'ɛ̰̃': 'ɛ',      # Creaky nasalized epsilon to epsilon (6)

    # Modified dentals/alveolars
    'd̪ʰ': 'd',     # Aspirated dental d to d (9)
    't̪ʰ': 't',     # Aspirated dental t to t (4)
    't̪ʲ': 'tʲ',     # Palatalized dental t to t (4)
    'tʲʰ': 'tʲ',     # Palatalized aspirated t to t (4)
    'dʰ': 'd',     # Aspirated d to d (9)
    'ðʲ': 'ð',     # Palatalized eth to eth (35)
    'zʲ': 'z',     # Palatalized z to z (36)
    'zʷ': 'z',     # Labialized z to z (36)
    
    # Complex modifications
    'ʃʷ': 'ʃ',     # Labialized sh to sh (1)
    'ɕʷ': 'ɕ',     # Labialized alveolo-palatal to sh (1)
    'ʑʷ': 'ʒ',     # Labialized voiced alveolo-palatal to zh (2)
    'ʕʷ': 'h',     # Labialized pharyngeal to h (37)
    'ħʷ': 'h',     # Labialized voiceless pharyngeal to h (37)
    'ʁʷ': 'ɹ',     # Labialized uvular to r (17)
    'χʲ': 'h',     # Palatalized x to h (37)
    'hʲ': 'h',     # Palatalized h to h (37)

    # Retracted/advanced variants
    'ɨ̠': 'ɨ',      # Retracted barred-i to barred-i (45)
    'ʊ̠': 'ʊ',      # Retracted upsilon to upsilon (22)
    'ʊ̟': 'ʊ',      # Advanced upsilon to upsilon (22)
    'æ̟': 'æ',      # Advanced ae to ae (32)
    'ə̟': 'ə',      # Advanced schwa to schwa (18)
    
    # Dental variants
    'n̪': 'n',      # Dental n to n (11)
    'l̪': 'l',      # Dental l to l (16)
    
    # Special vowels
    'ö': 'ø',     # O-umlaut to long oe (40)
    'ü': 'y',     # U-umlaut to long y (39)
    'ʉ': 'ɨ',     # Central u to long u (5)
    'ɞ': 'ə',      # Open-mid central rounded to schwa (18)
    'ɤ̈': 'ə',      # Advanced close-mid back unrounded to schwa (18)
    'ɯ̈': 'ɨ',      # Advanced high back unrounded
    
    # Implosives/ejectives/glottalized
    'ɗ': 'd',      # Implosive d to d (9)
    'ɓ': 'b',      # Implosive b to b (26)
    'ʄ': 'ʒ',      # Implosive palatal to zh (2)
    'dˀ': 'd',     # Glottalized d to d (9)
    'bˀ': 'b',     # Glottalized b to b (26)
    'ˀa': 'a',    # Preglottalized a to long a (30)

    # Modified retroflexes
    'ʈʰ': 't',     # Aspirated retroflex t to t (4)
    'ɖʰ': 'd',     # Aspirated retroflex d to d (9)

    # Remaining special cases
    'ɥ': 'j',      # Labial-palatal approximant to j (29)
    'ʀ': 'ɹ',      # Uvular trill to r (17)
    'ɹ̝': 'ɹ',      # Raised r to r (17)
    'ṽ': 'v',      # Nasalized v to v (15)
    'ə̥': 'ə',      # Voiceless schwa to schwa (18)
    'ə̯': 'ə',      # Non-syllabic schwa to schwa (18)
    'i̯': 'i',      # Non-syllabic i to i (41)
    'l̴': 'l',      # Velarized l to l (16)
    'dⁿ': 'd',     # Prenasalized d to d (9)
    'tⁿ': 't',     # Prenasalized t to t (4)
    
    # Breathy/creaky variants
    'd̪̤': 'd',     # Breathy dental d to d (9)
    'ɑ̤': 'a',     # Breathy long a to long a (13)
    'ṳː': 'u:',     # Breathy long u to long u (5)
    'ṳ': 'u:',      # Breathy u to long u (5)
    'ɯ̤': 'u:',     # Breathy unrounded u to long u (5)
    'ɪ̰': 'ɪ',      # Creaky short i to short i (31)
    'ɔ̰': 'ɔ',     # Creaky open-o to long open-o (3)
    'ɔ̤': 'ɔ',     # Breathy open-o to long open-o (3)
    
    # Height/backness variants
    'ɑ̝': 'a',     # Raised long a to long a (13)
    'ɛ̞': 'ɛ',      # Lowered epsilon to epsilon (6)
    'ɛ̝': 'ɛ',      # Raised epsilon to epsilon (6)
    'e̝': 'e',      # Raised e to e (42)
    'o̝': 'o',      # Raised o to o (44)
    'u̝': 'u:',     # Raised u to long u (5)
    'ɑ̞': 'a',     # Lowered long a to long a (13)
    'a̘': 'a',     # Advanced tongue root a to long a (30)
    'ä': 'a',     # Centralized a to long a (30)
    
    # Modified vowel quality
    'ɛ̈': 'ɛ',      # Centralized epsilon to epsilon (6)
    'œ̈': 'ø',     # Centralized oe to long oe (40)
    'ʌ̈': 'ʌ',      # Centralized wedge to schwa (18)
    'ɛ̠': 'ɛ',      # Retracted epsilon to epsilon (6)
    'a̠': 'a',     # Retracted a to long a (30)
    'o̠': 'o',      # Retracted o to o (44)
    'i̠': 'i',      # Retracted i to i (41)
    
    # Remaining consonant variants
    't̠': 't',      # Retracted t to t (4)
    'd̠': 'd',      # Retracted d to d (9)
    'n̠': 'n',      # Retracted n to n (11)
    't̟': 't',      # Advanced t to t (4)
    'r̟': 'ɹ',      # Advanced r to r (17)
    'r̠': 'ɹ',      # Retracted r to r (17)
    'rˠ': 'ɹ',     # Velarized r to r (17)
    'ɪ̥': 'ɪ',      # Voiceless short i to short i (31)
    'ʔʷ': 'noise', # Labialized glottal stop to noise (50)
    'ɕʼ': 'ɕ',     # Ejective alveolo-palatal to sh (1)
    'cʼ': 'k',     # Ejective c to k (27)
    'cʷʰ': 'k',    # Labialized aspirated c to k (27)
    'w̝': 'w',      # Raised w to w (47)

    'ʃ̠': 'ʃ',      # Retracted sh to sh (1)
    'ɪ̰̃': 'ɪ',      # Creaky nasalized short i to short i (31)
    'tʷʼ': 't',    # Labialized ejective t to t (4)
    'ŋʲ': 'ŋ',     # Palatalized ng to ng (34)
    'bʰ': 'b',     # Aspirated b to b (26)
    'æ̈': 'æ',      # Centralized ae to ae (32)
    'ɘ': 'ə',       # Close-mid central unrounded vowel to schwa (18)
    'tsʰ': 'ts',    # Aspirated ts to ts (4)
    'r̩ː': 'ɚ',     # Long rhotic schwa to schwar (14)


    'nθ': 'n',      # Dental nasal to n (11)
    'df': 'd',      # Dental fricative to d (9)
    'ds': 'd',      # Dental sibilant to d (9)
    'in': 'i:',     # Dental i to long i (12)
    'deɪ': 'eɪ',    # Dental eɪ to diphthong (23)
    'dt': 'd',      # Dental t to d (9)
    'dn': 'd',      # Dental n to d (9)
    'is': 'i:',     # Dental s to long i (12)
    'dɪ': 'd',      # Dental i to d (9)
    'if': 'i:',     # Dental f to long i (12)
    'dθ': 'd',      # Dental th to d (9)
    'uːθ': 'u:',    # Long u with th to long u (5)
    'ieɪ': 'eɪ',    # Dental eɪ to diphthong (23)
    'oʊz': 'oʊ',    # Diphthong oʊ with z to oʊ (24)
    'vθ': 'v',      # Dental v to v (15)
    'sθ': 's',      # Dental s to s (21)
    'nt': 'n',      # Dental t to n (11)
    'iw': 'i:',     # Dental w to long i (12)
    'iθ': 'i:',     # Dental th to long i (12)

    # new mappings for {'kb': 2, 'ək': 6, 'kd': 14, 'tθ': 2, 'iːf': 3, 'tʃɹ': 26, 'kw': 9, 'nf': 7, 'iːt': 11, 'oːɹθ': 5, 'iːθ': 5, 'iːeɪ': 2, 'neɪ': 1}

    'kb': 'k',      # Dental k to k (27)
    'ək': 'ə',      # Dental ə to schwa (18)
    'kd': 'k',      # Dental d to k (27)
    'tθ': 't',      # Dental th to t (4)
    'iːf': 'i:',    # Long i with f to long i (12)
    'tʃɹ': 'tʃ',    # Dental tsh with r to tsh (1)
    'kw': 'k',      # Dental w to k (27)
    'nf': 'f',      # Dental f to n (11)
    'iːt': 'i:',    # Long i with t to long i (12)
    'oːɹθ': 'oʊ',   # Long o with r and th to oʊ (24)
    'iːθ': 'i:',    # Long i with th to long i (12)
    'iːeɪ': 'eɪ',   # Long i with eɪ to diphthong (23)
    'neɪ': 'eɪ',    # Dental eɪ to eɪ (1)
}



def get_compound_phoneme_mapping(phoneme):
    # First try direct mapping
    if phoneme in phoneme_mapping:
        return phoneme_mapping[phoneme]
    
    # For compound phonemes, map components and combine
    mapped = ""
    remaining = phoneme
    while remaining:
        found = False
        # Try to match longest possible substring first
        for i in range(len(remaining), 0, -1):
            subset = remaining[:i]
            if subset in phoneme_mapping:
                mapped += phoneme_mapping[subset]
                remaining = remaining[i:]
                found = True
                break
        if not found:
            # If no mapping found for current character, treat as noise
            remaining = remaining[1:]
    
    return mapped if mapped else "noise"



def create_normalized_mapping(mapping_dict):
        
    # Create normalized version of the mapping
    from unicodedata import normalize
    """Create a mapping dictionary with normalized Unicode characters."""
    return {
        normalize('NFC', key): normalize('NFC', value)
        for key, value in mapping_dict.items()
    }


phoneme_mapper = create_normalized_mapping(phoneme_mapping) #Both the 'key' and value ar normalized

#print(phoneme_mapper)



def analyze_phoneme_merger(phoneme_mapper):
    # Check for circular references
    def check_circular_refs(mapper):
        issues = []
        for phoneme, target in mapper.items():
            if target in mapper and mapper[target] != target:
                issues.append(f"Potential circular reference: {phoneme} -> {target} -> {mapper[target]}")
        return issues

    # Check for consistency in vowel merging
    def check_vowel_consistency(mapper):
        issues = []
        # Common vowel pairs that should merge consistently
        vowel_pairs = [
            ('ɑː', 'ɑːɹ'),  # Long a with/without r
            ('ɔː', 'ɔːɹ'),  # Long o with/without r
            ('iː', 'iə'),   # Long i and i-schwa
            ('ʊ', 'ʊɹ'),    # Short u with/without r
        ]
        
        for v1, v2 in vowel_pairs:
            if v1 in mapper and v2 in mapper:
                if mapper[v1] != mapper[v2]:
                    issues.append(f"Inconsistent vowel mapping: {v1} -> {mapper[v1]} but {v2} -> {mapper[v2]}")
        return issues

    # Check for r-colored vowel consistency
    def check_r_colored_consistency(mapper):
        issues = []
        r_colored = ['ɪɹ', 'ɛɹ', 'ʊɹ']
        target = 'ɚ'  # All should map to schwa-r
        
        for phoneme in r_colored:
            if phoneme in mapper and mapper[phoneme] != target:
                issues.append(f"Inconsistent r-colored vowel: {phoneme} -> {mapper[phoneme]}, expected -> {target}")
        return issues

    # Check compound phoneme handling
    def check_compound_handling(mapper):
        issues = []
        for phoneme in mapper:
            if len(phoneme) > 1 and phoneme not in ['tʃ', 'dʒ', 'aɪ', 'eɪ', 'oʊ', 'aʊ', 'ɔɪ', 'iə', 'uː', 'iː', 'ɑː', 'ɔː', 'ɜː', 'əl']:
                if not phoneme.startswith(mapper[phoneme][0]):
                    issues.append(f"Potentially incorrect compound mapping: {phoneme} -> {mapper[phoneme]}")
        return issues

    # Collect all issues
    all_issues = []
    all_issues.extend(check_circular_refs(phoneme_mapper))
    all_issues.extend(check_vowel_consistency(phoneme_mapper))
    all_issues.extend(check_r_colored_consistency(phoneme_mapper))
    #all_issues.extend(check_compound_handling(phoneme_mapper))

    print("Testing complete vocab:")
    for kv in list(complete_vocab.keys()):
        mapped = get_compound_phoneme_mapping(kv)
        if (mapped != kv):
            if (mapped == 'noise') or (complete_vocab[kv] > 5000):
                print(f"{kv} -> {mapped} \tcount: {complete_vocab[kv]}")

    # Verify coverage
    missing_phonemes = set(complete_vocab.keys()) - set(phoneme_mapping.keys())
    
    print(f"Missing phonemes: {missing_phonemes}")
    for phoneme in missing_phonemes:
        print(f"{phoneme} -> {complete_vocab[phoneme]}")
    
    return all_issues



def create_new_index():

    # First, count the frequencies mapping to the new phonemes (count merged branches)
    phoneme_vocab_mapped_counts = {}
    for key, value in phoneme_mapper.items():
        if value not in phoneme_vocab_mapped_counts:
            phoneme_vocab_mapped_counts[value] = 0
        phoneme_vocab_mapped_counts[value] += 1
    
    print(f"Mapped: {len(phoneme_mapper)} phonemes onto {len(phoneme_vocab_mapped_counts)} phonemes")
    #print(phoneme_mapper)
    # Sort phonemes by frequency in descending order, excluding SIL
    sorted_phonemes = sorted(
        [p for p in phoneme_vocab_mapped_counts.keys() if p not in ['SIL', 'noise']],
        key=lambda x: phoneme_vocab_mapped_counts[x],
        reverse=True
    )
    
    # Create the index mapping
    phoneme_mapped_index = {}
    
    # Put SIL at index 0
    phoneme_mapped_index['SIL'] = 0
    
    # Add the rest of the phonemes with indices starting from 1
    for i, phoneme in enumerate(sorted_phonemes):
        phoneme_mapped_index[phoneme] = i + 1
        
    # Put noise at the last index
    phoneme_mapped_index['noise'] = len(sorted_phonemes) + 1
    
    print("New index created:")
    print(phoneme_mapped_index)


    print("Unique phonemes in the new index:")
    print(list(phoneme_mapped_index.keys()))
    # Run the analysis
    issues = analyze_phoneme_merger(phoneme_mapper)


    # Print findings
    print("Found the following potential issues:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

    # Additional validation of the phoneme_mapped_index
    mapped_phonemes = set(phoneme_mapped_index.keys())
    merger_outputs = set(p for p in phoneme_mapper.values() if not p.endswith('*'))
    missing_indices = merger_outputs - mapped_phonemes
    extra_indices = mapped_phonemes - merger_outputs

    print("\nIndex validation:")
    if missing_indices:
        print(f"Merged phonemes missing from index: {missing_indices}")
    if extra_indices:
        print(f"Extra phonemes in index: {extra_indices}")
    print("Done")

def check_missing_phonemes():


    test_phonemes = ['a', 'd͡ʒ', 'ʃʲ', 'm', 'ɜ', 'ɘ', 'ʃ', 't͡ʃʰ', 'r', 'ä', 't͡ʃ', 'ə̆', 'pʰ', 'ɜ̆', 'ʌ̈', 't', 'ʃʰ', 'kʼ', 'ʒʲ', 'ə', 'ă', 'b', 'ɨ', 'æ̈', 'j', 'ɛ̈', 'p', 'd', 'n', 'ɥ', 'ɡ', 't͡ʃʼ', 'χ', 'ˀa', 'ʒ', 'ħʷ', 'ɹ', 'ħ', 'œ̈', 'ɾ', 'ʁ', 'ɤ̈', 'z', 'i', 'χʲ', 'tʰ', 's', 'ʁʷ', 'h', 'ɛ', 'k', 'ɑ', 'x', 'ɔ', 'o', 'u', 'e', 'ɑ̃', 'ŋ', 'l', 'ʊ', 'ã', 'q̠', 'õ', 'w', 'β', 'f', 'v', 'ʎ', 'oː', 'eː', 'kʰ', 'ð', 'œ', 'ɹ̩', 'ɛ̝', 'ʔ', 'l̥', 'e̝', 'aː', 'uː', 'iː', 'ʌ̃', 'æ', 'ẽ', 'y', 'yː', 'ɪː', 'ɛː', 'øː', 'œː', 'ɑː', 'o̝', 'ʌ', 'ø', 'ɯ', 'sː', 'ɛ̃', 'c', 'ɪ', 'ɟ', 'ɲ', 'æː', 'æ̃ː', 'ʉ', 'ɫ̩', 'ʋ', 'ɫ', 'kʲ', 'ɣ', 'ɦ', 'n̩', 'ɸ', 'dʰ', 'm̩', 'h̩', 'ç', 'bʰ', 't̪', 'd̪', 'd̪̤', 'b̤', 'n̪', 'ĩ', 'ũː', 'ũ', 'j̤', 'l̪', 'pː', 'kː', 'rː', 'nː', 'l̪ː', 'bː', 'mː', 'ɞ', 't̪ʲ', 'hː', 'ʔː', 'tː', 'dː', 'ʈ', 'ɖ', 'ʂ', 'ʐ', 'r̥', 'ɔː', 'ʏː', 'ʏ', 'θ', 'n̥', 'cː', 'ɟː', 'fː', 'lː', 'ŋ̥', 'ə̯', 'ə̟', 'i̯', 'ʊ̟', 'ɛ̞', 'ʊ̠', 'r̟', 'r̠', 'ɕ', 'pʲ', 'bʲ', 'ŭ', 'tʲ', 'ĕ', 'dʲ', 'ɡʲ', 'nʲ', 'fʲ', 'zʲ', 'vʲ', 'lʲ', 'sʲ', 'xʲ', 'hʲ', 'ŏ', 'mʲ', 't͡ʃʲ', 'd͡ʒʲ', 'æ̆', 'ŋʲ', 'rʲ', 'ɾʲ', 'ĭ', 'ɔ̆', 's̪', 'ɱ', 'ɽ', 'ɳ', 'ʈʰ', 'ɖʰ', 'ɵ', 't̪ʰ', 'd͡ʒʰ', 'ɭ', 'ʊ̃', 'sʰ', 'ḭ', 'cʰ', 'ʊ̰', 'ɛ̰', 'ɪ̰', 'a̰', 'ḛ', 'o̰', 'ɛ̰̃', 'ɪ̃', 'ʊ̰̃', 'ɲ̥', 'æ̃', 'm̥', 'ɪ̰̃', 'ɔ̰', 'wː', 'ɔ̃ː', 'ɗ', 'ɔ̃', 'õː', 'ɯː', 'ə̃', 'tʰː', 'pʰː', 'vː', 'zː', 'ʃː', 'jː', 'ɲː', 'xː', 'çː', 'ɓ', 'ãː', 't͡sʼ', 'ɻ', 'ʀ', 't͡s', 'a', 'b', 'w', 'e', 'ɔ', 'p', 'ɛ', 't', 'o', 't͡ʃ', 'u', 'd', 'k', 'ɔ̃', 'kʷ', 'ɡ', 'k͡p', 'm', 'n', 'n̠', 'j', 'f', 's', 'ç', 'ɹ', 'l', 'i', 'ʍ', 'd̠', 'ʐ', 'ŋ', 'ɥ', 't̠', 'ɕʷ', 'ɕ', 'pʰ', 'tʰ', 'sʰ', 'kʰ', 'z', 'ä', 'h', 'v', 'ʃ', 'ʒ', 'r', 'ü', 'y', 'ʔ', 'ɪ', 'æ', 'ə', 'q̠', 'ɞ', 't͡ʃʰ', 'ĩ', 'ã', 'õ', 'ʋ', 'x', 'ɾ', 'ɓ', 'ɗ', 'c', 'ɟ', 'ʄ', 'aː', 'ɲ', 'ɔː', 'tʲ', 'oː', 'ɤː', 'uː', 'ʊː', 'ɳ', 'ɯː', 'ðʲ', 'tʲʰ', 'ɛ̃', 'ɣ', 'kʲ', 'ũ', 'ĩː', 'rˠ', 'ɛ̃ː', 'ãː', 'ɔ̃ː', 'ũː', 't̪', 'ʑʷ', 'ʑ', 'ɡʷ', 'ŋʷ', 'ɽ', 'o̠', 'w̃', 'ɯ', 'ö', 'ɡ͡b', 'd͡ʒ', 'ʁ', 'q', 'i̠', 'ɛ̠', 'v̩', 'l̥', 'ɤ', 'r̥', 'ɢ', 'ɢʲ', 'χ', 'kʰʲ', 'm̥', 'n̥', 'nː', 'pː', 'lː', 'rː', 'æː', 'eː', 'o˞', 'e˞', 'a˞', 'i˞', 'iː', 'u˞', 'ʕʷ', 'ʕ', 'xʷ', 'ɬ', 'qʷ', 'ɑ', 'ɪ̃', 'ẽ', 'ʊ', 'd̪', 'd͡ʒʰ', 'ɦ', 't̪ʰ', 'd̪ʰ', 'dʰ', 'bʰ', 'ʌ', 'pʼ', 'ʊ̃', 'kʼ', 'β', 'kʼʲ', 'ħ', 'qʼ', 'cʼ', 'kʰʷ', 'qʰʷ', 'ɨ', 'ð', 'ɖ', 'ɸ', 'ʏ', 'ø', 'l̩', 'dʷ', 'pʷ', 'bʷ', 'tʷ', 'ṽ', 'z̃', 'ʃʷ', 'ʒʷ', 'a̘', 't͡s', 'n̤', 'ŋ̩', 'h̩', 'ɹ̝', 'ɑː', 'ɑ̞', 'ɑ̝', 'ɛː', 'ɪː', 'u̝', 'sʲ', 'ɜ', 'ɨː', 'θ', 'l̴', 'n̩', 'j̃', 't͡ɬ', 'sʼ', 'kʷʼ', 'cʰ', 'qʷʼ', 'zʷ', 'qʰ', 'kʷʰ', 't͡ɬʼ', 'cʷʰ', 'ʁʷ', 'tʷʼ', 'a̤', 'ɔ̤', 'o̤ː', 'i̤ː', 'ṳ', 'o̤', 'ṳː', 'ɯ̤', 'tʼ', 'ɑ̃', 'ɫ', 'ɑ̤', 'ʌ̃', 'ɛ̤', 'p͡t', 'b͡d', 'mʷ', 'w̝', 'ʎ̥', 'ɮ', 'ʃ̠', 'fː', 'i̥', 'u̥', 'ɪ̥', 'zː', 'sː', 'ʎ', 'ə̥', 'ʃː', 'e̥', 'ỹ', 'ɯ̈', 'ʉ', 'ɒ', 'xː', 'l̪', 'n̪', 'θː', 'ɒː', 'dˀ', 'bˀ', 't̟', 'æ̟', 'dⁿ', 'ɨ̠', 'tⁿ', 'a̠', 't͡sʰ', 'ɕʰ', 'm̩', 'ɭ', 'ə̃', 'ɕʼ', 't͡ʃʼ', 'ʔʷ', 'tsʰ'] # from UCLA phonetics, some repeated
    missing_phonemes = set(test_phonemes) - set(phoneme_mapper.keys())
    print(f"Missing phonemes: {missing_phonemes}")
    print(len(missing_phonemes))

    # list of phonemes that map to noise:
    noise_phonemes = [k for k, v in phoneme_mapper.items() if v == 'noise']
    noise_phonemes_in_test_set = set(noise_phonemes) & set(test_phonemes)
    print(f"Noise phonemes in test set: {noise_phonemes_in_test_set}")
    # only  {'ʔ', 'ʔʷ'} are mapped to noise from ucla dataset

def check_duplicates():
    from collections import defaultdict

    # Create a dictionary to store the key-value pairs
    key_value_pairs = defaultdict(set)

    # Populate the key-value pairs
    for key, value in phoneme_mapper.items():
        key_value_pairs[key].add(value)

    # Find and print keys with multiple different values
    duplicates = {key: values for key, values in key_value_pairs.items() if len(values) > 1}

    print("Duplicate keys with different values:", len(duplicates))
    for key, values in duplicates.items():
        print(f"Key '{key}' has different values: {values}")



def make_phoneme_groups():
    
        
    phoneme_groups_19 = {
        # Vowels - Separated by height and frontness
        "high_front_vowels": ["i", "i:", "ɪ", "y", "ʏ", "iː"],
        "high_back_vowels": ["u", "u:", "ʊ", "ɯ", "ʉ", "ɨ", "uː"],
        "mid_front_vowels": ["e", "e:", "ɛ", "ø", "œ", "eː"],
        "mid_central_vowels": ["ə", "ɜ", "ɜ:", "ɚ", "ʌ", "ɘ", "ɵ"],
        "mid_back_vowels": ["o", "o:", "ɔ", "ɔ:", "ɤ", "oː"],
        "low_vowels": ["a", "a:", "æ", "ɐ", "ɑ", "ɑ:", "ɒ", "aː"],
        "diphthongs": ["aɪ", "eɪ", "ɔɪ", "aʊ", "oʊ", "ɛə", "ɪə", "ʊə"],
        
        # Consonants - Organized by manner and voicing
        "voiceless_stops": ["p", "t", "k", "q", "ʔ", "ʈ", "c"],
        "voiced_stops": ["b", "d", "g", "ɢ", "ɖ", "ɟ"],
        "voiceless_fricatives": ["f", "θ", "s", "ʃ", "ç", "x", "h", "ħ", "ʂ", "ɕ", "χ"],
        "voiced_fricatives": ["v", "ð", "z", "ʒ", "ʝ", "ɣ", "ʕ", "ʐ", "ʑ", "ʁ"],
        "voiceless_affricates": ["ts", "tʃ", "tɕ", "ʈʂ"],
        "voiced_affricates": ["dz", "dʒ", "dʑ", "ɖʐ"],
        "nasals": ["m", "n", "ɲ", "ŋ", "ɴ", "ɱ", "ɳ"],
        
        # Liquids, glides, and palatalized sounds
        "laterals": ["l", "ɭ", "ʎ", "ʟ"],
        "rhotics": ["r", "ɾ", "ɹ", "ʀ", "ɽ", "ɻ"],
        "glides": ["j", "w", "ɥ", "ɰ"],
        "palatalized": ["ɭʲ", "rʲ", "tʲ", "nʲ"],
        
        "SIL": ["SIL"],
        "noise": ["noise"],
    }

    phoneme_groups = {
        # Vowels - Adjusted based on confusion patterns
        "front_vowels": ["i", "i:", "ɪ", "y", "ʏ", "iː", "e", "e:", "ɛ", "ø", "œ", "eː"],  # Merged high/mid front
        "central_vowels": ["ə", "ɜ", "ɜ:", "ɚ", "ʌ", "ɘ", "ɵ"],  # Keep central vowels separate
        "back_vowels": ["u", "u:", "ʊ", "ɯ", "ʉ", "ɨ", "uː", "o", "o:", "ɔ", "ɔ:", "ɤ", "oː"],  # Merged high/mid back
        "low_vowels": ["a", "a:", "æ", "ɐ", "ɑ", "ɑ:", "ɒ", "aː"],  # Keep low vowels separate
        "diphthongs": ["aɪ", "eɪ", "ɔɪ", "aʊ", "oʊ", "ɛə", "ɪə", "ʊə"],  # Keep diphthongs separate
        
        # Consonants - Maintain voicing distinction for stops and fricatives
        "voiceless_stops": ["p", "t", "k", "q", "ʔ", "ʈ", "c", "tʲ"],  # Add palatalized t
        "voiced_stops": ["b", "d", "g", "ɢ", "ɖ", "ɟ"],
        "voiceless_fricatives": ["f", "θ", "s", "ʃ", "ç", "x", "h", "ħ", "ʂ", "ɕ", "χ"],
        "voiced_fricatives": ["v", "ð", "z", "ʒ", "ʝ", "ɣ", "ʕ", "ʐ", "ʑ", "ʁ"],
        
        # Keep affricates distinction by voicing
        "voiceless_affricates": ["ts", "tʃ", "tɕ", "ʈʂ"],
        "voiced_affricates": ["dz", "dʒ", "dʑ", "ɖʐ"],
        
        # Merge palatalized nasals with base nasals
        "nasals": ["m", "n", "nʲ", "ɲ", "ŋ", "ɴ", "ɱ", "ɳ"],
        
        # Merge palatalized laterals with base laterals
        "laterals": ["l", "ɭ", "ɭʲ", "ʎ", "ʟ"],
        
        # Merge palatalized rhotics with base rhotics
        "rhotics": ["r", "rʲ", "ɾ", "ɹ", "ʀ", "ɽ", "ɻ"],
        
        # Keep glides separate
        "glides": ["j", "w", "ɥ", "ɰ"],
        
        # Special tokens
        "SIL": ["SIL"],
        "noise": ["noise"],
    }
    
    # verify groups cover all phonemes
    phoneme_groups_flat = [p for g in phoneme_groups for p in phoneme_groups[g]]
    extra_phonemes = set(phoneme_groups_flat)- set(phoneme_mapped_index.keys())
    print(f"extra phonemes: {extra_phonemes}")
    missing_phonemes = set(phoneme_mapped_index.keys()) - set(phoneme_groups_flat)
    print(f"missing phonemes: {missing_phonemes}")
    assert len(missing_phonemes) == 0, "Phoneme groups do not cover all phonemes"

    # remove extra phonemes:
    for p in extra_phonemes:
        for g in phoneme_groups:
            if p in phoneme_groups[g]:
                phoneme_groups[g].remove(p)


    # covert groups to index
    phoneme_groups_based = {}
    for g in phoneme_groups:
        phoneme_groups_based[g] = [phoneme_mapped_index[p] for p in phoneme_groups[g]]

    # verify groups are correctly mapped
    for g in phoneme_groups:
        for p in phoneme_groups[g]:
            assert phoneme_mapped_index[p] in phoneme_groups_based[g], f"{p} not in {g}"


    global phoneme_groups_index
    # clear
    phoneme_groups_index = {}
    phoneme_groups_index = { "SIL": 0,}
    for i, g in enumerate(phoneme_groups):
        if (g != "SIL") and (g != "noise"):
            phoneme_groups_index[g] = i+1
    phoneme_groups_index["noise"] = len(phoneme_groups_index)
    print("phoneme_groups_index:", phoneme_groups_index)
    print("total groups (excluding noise)", len(phoneme_groups_index)-1)
    

    # base phonemes index to group index
    base66_to_groups = {}
    for p in phoneme_mapped_index:
        for g in phoneme_groups:
            if p in phoneme_groups[g]:
                base66_to_groups[phoneme_mapped_index[p]] = phoneme_groups_index[g]


    # verify all phonemes are mapped to a group
    assert len(base66_to_groups) == len(phoneme_mapped_index), "Not all phonemes are mapped to a group"
    print("base66_to_groups:", base66_to_groups)


#main

if __name__ == "__main__":
    # Create the new index
    #create_new_index()
    
    #check_missing_phonemes()
    #check_duplicates()
    make_phoneme_groups()