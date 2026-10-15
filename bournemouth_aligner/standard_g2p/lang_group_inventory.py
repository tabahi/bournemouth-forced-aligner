"""Per-language-group phoneme inventories: gold indices -> local indices.

The local tokens of a language group are the final output of this repo;
downstream tasks are expected to use them. Each language group's token list is
a SUBSET of the 270 gold tokens (see mappings/lang_group_inventories.json,
built by scripts/create_phoneme_inventories.py). The .gs.json files keep gold
indices only; the conversion happens here, at load time, so the lists can be
regenerated -- a different coverage target, a language added to EXCLUDED_ISO --
without rewriting any .gs.json.

"Language group" (lang_group) is never shortened to "group" in this repo:
"phoneme group" (phg, phoneme_features.GROUPS) is the other meaning.

    from standard_g2p import lang_group_inventory as LGI

    LGI.list_all_lang_groups                      # ['latin', 'vietnamese', ..., 'burmese']
    LGI.member_langs['latin']                     # ['af', 'ang', ..., 'uz']
    LGI.lang_group_of('pt-BR')                    # 'latin'
    t = LGI.to_local(seg['gold_ph'], 'pt-BR', seg['gold_ph_count'])
    t['local_ph']                                 # indices into the language group's token list
    LGI.expand(seg['tone'], t['local_ph_count'])  # tone per local token, same length
    t['n_local_ph']                               # its token count (208)
    LGI.decode(t['local_ph'], 'pt-BR')            # ['p', 'a', 'ʁ', ...]

`lang` is a single BCP 47 code, as everywhere else. Other spellings lang_codes
accepts ('en-US', 'cmn', and the bare ISO 639-3 codes in .gs.json files written
before the BCP 47 switch) are resolved through the table's alias list.

Like lang_codes, this imports nothing heavy, so a data loader can use it
without pulling in transformers.
"""
import os
import json

try:                                  # imported as part of the package
    from . import phoneme_inventory_gold as PI
except ImportError:                   # imported with standard_g2p/ itself on sys.path
    import phoneme_inventory_gold as PI

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'mappings', 'lang_group_inventories.json')

_TABLE = None


class StaleInventoryError(RuntimeError):
    """lang_group_inventories.json was built against a different gold inventory."""


def gold_fingerprint():
    """Same fingerprint create_phoneme_inventories.py records in the table."""
    import hashlib
    return hashlib.sha1('\n'.join(PI.TOKENS).encode('utf-8')).hexdigest()[:12]


def load(path=None):
    """The table, read once and checked against the current gold inventory.

    A language group's list indexes into gold TOKENS, so a list built against a
    different TOKENS maps every index to the wrong phoneme without any error.
    Refuse it here instead; the fix is to rerun
    scripts/create_phoneme_inventories.py.
    """
    global _TABLE
    if _TABLE is not None and path is None:
        return _TABLE
    with open(path or DEFAULT_PATH, encoding='utf-8') as f:
        table = json.load(f)
    want, got = gold_fingerprint(), table['gold']['fingerprint']
    if got != want:
        raise StaleInventoryError(
            f'lang_group_inventories.json was built against gold inventory {got}, '
            f'but phoneme_inventory_gold is now {want}. Rerun '
            f'scripts/create_phoneme_inventories.py.')
    if path is None:
        _TABLE = table
    return table


def canonical_lang(lang):
    """Any accepted spelling -> the canonical BCP 47 code the table is keyed by
    ('en-US' -> 'en', 'cmn' -> 'zh'). Raises KeyError on an unknown code."""
    t = load()
    code = t['aliases'].get(lang, lang)
    if code not in t['lang_to_lang_group']:
        raise KeyError(f'{lang!r} is not a language the G2P can emit; known: '
                       f'{sorted(t["lang_to_lang_group"])}')
    return code


# `list_all_lang_groups` and `member_langs` are module attributes, not
# functions, but they are built on first access (PEP 562 module __getattr__):
# a plain constant would read the table at import time, and importing this
# module must not raise StaleInventoryError before it is used. Each access
# returns a fresh copy, so editing the result cannot corrupt the table.

def _lang_groups():
    """Language group names, in table order."""
    return list(load()['lang_groups'])


def _member_langs():
    """Language group -> the sorted BCP 47 codes routed to it. Includes the
    excluded languages (lang_codes.EXCLUDED_ISO), exactly as `lang_group_of`
    routes them; `inventory(g)['excluded_langs']` names those."""
    out = {g: [] for g in load()['lang_groups']}
    for code, g in load()['lang_to_lang_group'].items():
        out[g].append(code)
    return {g: sorted(codes) for g, codes in out.items()}


_LAZY = {'list_all_lang_groups': _lang_groups, 'member_langs': _member_langs}


def __getattr__(name):
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


def lang_group_of(lang):
    """BCP 47 code -> language group name.

    Excluded languages (lang_codes.EXCLUDED_ISO: 'km', 'my', 'nan', 'tts') still get
    their language group, but did not shape its list, so their coverage is not
    guaranteed -- 'burmese' has no usable language and holds only the specials.
    """
    return load()['lang_to_lang_group'][canonical_lang(lang)]


def inventory(lang_group):
    """The full record for one language group: tokens, gold_index,
    gold_to_local, coverage, selected_by, langs."""
    lang_groups = load()['lang_groups']
    try:
        return lang_groups[lang_group]
    except KeyError:
        raise KeyError(f'{lang_group!r} is not a language group; '
                       f'known: {sorted(lang_groups)}') from None


def n_tokens(lang_group):
    """Number of local tokens in a language group, specials included."""
    return inventory(lang_group)['n_tokens']


def tokens(lang_group):
    """Local index -> phoneme string. The 4 gold special tokens come first, at
    the same indices as in gold (0 = CTC blank)."""
    return inventory(lang_group)['tokens']


def to_local(gold_ph, lang, gold_ph_count=None):
    """Gold indices (a .gs.json `gold_ph`) -> the language group's local
    indices, with the scope needed to interpret them.

    `gold_ph_count` is the segment's per-phoneme count of gold tokens (a
    .gs.json `gold_ph_count`). Pass it to get `local_ph_count` back, which is
    what lines the per-phoneme layers (tone, stress, length, word_num) up with
    `local_ph`: see `expand`. For files written before `gold_ph_count`
    existed, `gold_ph_count_of(seg['phonemes'])` recomputes it.

    Returns a dict:
        lang        canonical BCP 47 code the table is keyed by
        lang_group  the language group whose token list `local_ph` indexes
        local_ph    local index per gold index, same length as `gold_ph`
        local_ph_count  (only when `gold_ph_count` is given) per phoneme, how
                    many `local_ph` entries it became. One local token per
                    gold token, so this equals `gold_ph_count`; it is checked
                    to sum to len(local_ph)
        n_local_ph  the language group's token count, specials included
        n_gold_ph   size of the gold space `gold_ph` indexed
        unk         the language group's <unk> index
        n_folded    how many entries of `gold_ph` were real phonemes outside
                    the language group's list and so became <unk> (a gold
                    <unk> that stays <unk> is not counted)
        gold_fingerprint  the gold inventory the table was built from

    A gold phoneme outside the language group's list becomes its <unk>. By
    construction that is under 0.1% of any member language's tokens, and
    phoneme_counts.md §5 has the exact figure per language.
    """
    code = canonical_lang(lang)
    lang_group = load()['lang_to_lang_group'][code]
    rec = inventory(lang_group)
    table = rec['gold_to_local']
    unk = rec['tokens'].index(PI.UNK)
    gold_unk = PI.TOKEN_INDEX[PI.UNK]
    local = [table[i] for i in gold_ph]
    out = {
        'lang': code,
        'lang_group': lang_group,
        'local_ph': local,
        'n_local_ph': rec['n_tokens'],
        'n_gold_ph': PI.N_TOKENS,
        'unk': unk,
        'n_folded': sum(1 for g, l in zip(gold_ph, local) if l == unk and g != gold_unk),
        'gold_fingerprint': load()['gold']['fingerprint'],
    }
    if gold_ph_count is not None:
        if sum(gold_ph_count) != len(local):
            raise ValueError(f'gold_ph_count sums to {sum(gold_ph_count)} but '
                             f'gold_ph has {len(local)} entries')
        out['local_ph_count'] = list(gold_ph_count)
    return out


def expand(per_phoneme, local_ph_count):
    """Repeat a per-phoneme layer so it lines up with `local_ph`.

        t = to_local(seg['gold_ph'], lang, seg['gold_ph_count'])
        tone = expand(seg['tone'], t['local_ph_count'])     # len == len(t['local_ph'])

    A phoneme that became two tokens (ʈ͡ʂ -> ʈ ʂ) gives both of them its tone,
    stress, length and word_num; one that became none drops out.
    """
    if len(per_phoneme) != len(local_ph_count):
        raise ValueError(f'{len(per_phoneme)} values for {len(local_ph_count)} '
                         f'phonemes')
    return [v for v, n in zip(per_phoneme, local_ph_count) for _ in range(n)]


def gold_ph_count_of(phonemes):
    """`gold_ph_count` recomputed from a segment's `phonemes`, for .gs.json
    files written before it was stored. Valid while the gold inventory is the
    one the file was written with (`load()` checks the fingerprint)."""
    return [len(PI.map_phoneme(p)) for p in phonemes]


def decode(local, lang):
    """Local indices -> phoneme strings, for the language group `lang` belongs to."""
    toks = tokens(lang_group_of(lang))
    return [toks[i] for i in local]
