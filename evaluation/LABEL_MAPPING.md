# How TIMIT and Buckeye are mapped onto the std phoneme set

TIMIT and Buckeye are hand-labelled in ARPAbet-style symbols. BFA's `mbfa-latin` models emit
the std phoneme set (latin group, 208 tokens). This page says how one becomes the other, what
that does to a boundary score, and what was measured when the mapping was tuned on
2026-10-11.

Code: [dataset/std_labels.py](dataset/std_labels.py) (the tables),
[dataset/timit_parser.py](dataset/timit_parser.py),
[dataset/buckeye_parser.py](dataset/buckeye_parser.py),
[compute_metrics.py](compute_metrics.py) (the scorer).

## The one thing to know first

**The scorer never reads a phoneme label.** For each segment it reads two things from the
SRT the parser wrote:

| field | used for |
|---|---|
| `text` | what the aligner is asked to place (text → G2P → tokens → Viterbi) |
| `boundaries` | the ground truth: times in ms from the segment start |

It then takes the onset of every predicted token except `SIL`, matches the two lists
one-to-one in time order, and counts a hit when a pair is within the tolerance (f1@25ms).

So whether `ah` is written `ə` or `ʌ` cannot move the score by a hair. The label table
decides the `local_ph` ids stored next to each phone (useful for inspection and for the
training probe), and it decides one thing that does matter: **whether an interval counts as a
speech phone**, because only speech phones contribute a boundary.

Three things move the score:

1. **Which times are in `boundaries`** (both corpora).
2. **Where a segment starts and stops** (Buckeye).
3. **What the `text` is**: every phone that is in the audio but not in the text, or in the
   text but not in the audio, costs boundaries on both sides of it, not just its own.

## The label table

One table for the ARPAbet symbols both corpora share, written the way the std G2P writes
English, so every id is one `mbfa-latin` was trained to emit on English speech.

| corpus symbol | std tokens | note |
|---|---|---|
| `iy ih eh ae aa ao uh uw` | `i ɪ ɛ æ ɑ ɔ ʊ u` | |
| `ah`, `ax`, `ax-h` | `ə` | the G2P has no `ʌ` in English |
| `er`, `axr` | `ɝ` | |
| `ey ay oy aw ow` | `e ɪ`, `a ɪ`, `ɔ ɪ`, `a ʊ`, `o ʊ` | **two tokens** for one hand-labelled phone |
| `ch jh` | `t ʃ`, `d ʒ` | **two tokens** |
| `p b t d k g` | `p b t d k ɡ` | |
| `f v th dh s z sh zh hh` | `f v θ ð s z ʃ ʒ h` | |
| `m n ng l r w y` | `m n ŋ ɫ ɹ w j` | |
| `dx` | `ɾ` | the flap; the G2P itself would write `t` or `d` |
| `q`, `tq` | `ʔ` | glottal stop; the G2P never writes it |
| `el em en eng` | `ɫ m n ŋ` | syllabic; the G2P writes `ə ɫ`, `ə n` (two tokens) |
| `nx` | `n` | nasal flap |

TIMIT only:

| symbol | std | note |
|---|---|---|
| `ix`, `ux`, `hv` | `ɪ`, `u`, `h` | allophones folded onto their phoneme |
| `pcl bcl tcl dcl kcl gcl` | `p b t d k ɡ` | a closure is its stop, so `tcl t` is `t t` |
| `h#`, `pau`, `epi` | `SIL` | not speech: no boundary |

Buckeye only:

| symbol | std | note |
|---|---|---|
| `aan aen ahn … uwn` (15) | the oral vowel | nasalised vowels |
| `h`, `hhn`, `ix` | `h`, `h`, `ɪ` | stray TIMIT-style symbols |
| `SIL` | `SIL` | not speech |
| `VOCNOISE NOISE LAUGH IVER IVER-LAUGH UNKNOWN EXCLUDE` | `noise` | not speech |
| `i e a an x id no j` | `ɪ ɛ ɑ ɑ ə ɪ n`, `d ʒ` | typos, see below |

Before a Buckeye label is looked up, `clean_buckeye_label` drops the transcriber's doubt mark
(`iy; *` → `iy`), the stress mark (`ih+1` → `ih`), and folds every spelling of a redacted name
onto `EXCLUDE`.

### No more `<unk>`

The eight typo labels (25 intervals in the whole corpus) used to be `<unk>`. Each is now read
from the word it sits in, using the `.words` file, which gives the word and its dictionary
phones next to what was transcribed:

| label | count | seen in | read as |
|---|---|---|---|
| `i` | 8 | `w i th` (with), `b i k ah z` (because) | `ih` → `ɪ` |
| `a` | 6 | `a n` (on), `a` (I) | `aa` → `ɑ` |
| `x` | 4 | `t eh x n` (ten), `y ih x n ow` (yknow) | a stray reduced vowel → `ə` |
| `e` | 2 | `dh e r` (there) | `eh` → `ɛ` |
| `id` | 1 | `r ah dx id d` (rotted) | `ih` → `ɪ` |
| `no` | 1 | `no ow n` (don't) | `n` |
| `an` | 1 | `t w an nx iy` (twenty) | nasalised `a` → `ɑ` |
| `j` | 1 | `<CUTOFF-jih=just>` | `jh` → `d ʒ` |

This changes no score (labels are not scored), but each of these intervals is now a speech
phone and so contributes its onset, where an `<unk>` interval did not.

## TIMIT

One clip is one sentence and one segment: `start` 0, `end` the clip length, `text` the
prompt from the `.TXT` file.

A phone belongs to the word whose span contains it. The silences around and between words
(`h#`, `pau`) belong to no word.

**`boundaries` = where each in-word interval starts.** Changed on 2026-10-11 from "both edges
of every in-word interval". The difference is the *end* of a word's last phone when a pause
follows (and the end of the utterance): that edge is the start of nothing, the scorer has no
predicted onset for it, and it was a guaranteed miss. An `epi` (the short silent gap inside a
word) starts no boundary either.

A stop that TIMIT labels as closure + release (`tcl t`) keeps **both** onsets. See the
measurement below for why.

## Buckeye

A recording is a whole interview (~7 min), so the parser cuts it into segments using the
`.words` file, then collects the boundaries of each segment from the `.phones` file. Both
files give the time where a row **ends**.

### Segments

The `.words` file is words interleaved with tags. What each tag is, over the whole corpus:

| tag | rows | what the `.phones` file has under it |
|---|---|---|
| `<SIL>` | 27 038 | silence |
| `<VOCNOISE>`, `<VOCNOISE-word>` | 22 659 | noise (and 2 492 phones under the worded ones) |
| `<IVER>` | 10 209 | the interviewer |
| `<LAUGH>`, `<LAUGH-yeah>` | 2 950 | laughter, **and 5 933 phones** of laughed speech |
| `<CUTOFF-th=the>` | 2 607 | 4 709 phones of a word that was started and dropped |
| `<NOISE>`, `<NOISE-word>` | 1 623 | noise, and 2 741 phones |
| `<EXT-I've>` | 756 | 2 222 phones of a drawn-out word |
| `<HES-uh>` | 694 | 1 397 phones of a hesitation |
| `<UNKNOWN>`, `<UNKNOWN-or?>` | 534 | unintelligible, 603 phones |
| `<ERROR-is=it's>` | 310 | 1 253 phones of a slip of the tongue |
| `<EXCLUDE-name>` | 182 | a redacted name |

**A segment now ends at every tag except `<SIL>`** (`is_break_token`). Before 2026-10-11 it
ended only at `<IVER>`, and every other tag was deleted from the text while its audio stayed
inside the segment. That left ~20 000 hand-labelled phones in the audio with no word in the
text to place on them, plus all the laughter and noise. This is the "missing phoneme in a
row" problem: the aligner has to stretch the neighbouring phones over the gap, and the damage
spreads along the segment.

So a laugh is not relabelled, it is cut out: the segment before it ends where the last plain
word ends, the next one starts where the next plain word starts. Leading and trailing
`<SIL>` are trimmed the same way. A pause *inside* a run of plain words stays inside the
segment.

A run longer than 30 s is still split, at the first pause after 10 s.

Effect on the corpus (255 interviews):

| | before | after |
|---|---|---|
| segments | 12 182 | 33 693 |
| audio inside segments | 25.5 h | 21.5 h |
| mean / median segment | 7.6 s / 6.1 s | 2.3 s / 1.4 s |
| segments under 1 s | 3 055 | 13 162 |
| ground-truth boundaries | 904 427 | 840 492 |

### Text

The segment text is its plain words joined by spaces. Seven of Buckeye's own spellings are
respelt (`SPELLINGS`), because the std G2P reads them as words or as letter names:

| Buckeye | corpus count | std G2P reads it as | respelt | then reads |
|---|---|---|---|---|
| `yknow` | 3 589 | `ɪ k n o ʊ` | `you know` | `j u n o ʊ` |
| `hm`, `hmm` | 32 | `e ɪ t ʃ m i` ("aitch em ee") | `hum` | `h ə m` |
| `mm` | 23 | `m ɪ s t ə m` ("mister m") | `um` | `ə m` |
| `mm-hmm`, `mm-hm`, `mhm` | 54 | `m ɪ m ə h ɑ m ə m` | `um hum` | `ə m h ə m` |

A pause inside a segment leaves no mark in the text. Writing it as a comma was tried and
gives the same alignment, onset for onset.

### Boundaries

**`boundaries` = where each speech phone starts, inside the segment.** Changed on 2026-10-11
from "every row time of the `.phones` file inside the segment", which also counted the edge
where a pause or a noise begins and the edge where the segment ends. Those are ends, not
onsets, and the scorer has no prediction for them.

`.words` and `.phones` times were typed separately and differ by a fraction of a millisecond,
so a phone is taken as inside a segment with 2 ms of tolerance.

## What was measured

`mbfa-latin`, published decoder settings (boundary bonus 30) unless a column says bonus 0,
frame-grid onsets. Scored with `compute_metrics.check_method_alignment`, aligned with
`process_alignments.task_bfa_align`, all in a scratch directory so the running sweeps and
`tmp/data` were not touched.

### Buckeye: 4 random interviews (s0402b, s1201a, s1902a, s3901b), ~15 000 boundaries

| segments end at | boundaries | segments | f1@25, bonus 30 | f1@25, bonus 0 |
|---|---|---|---|---|
| `<IVER>` only (before) | every edge (before) | 220 | **64.62** | 63.93 |
| `<IVER>` only | speech onsets | 220 | 63.96 | 64.10 |
| `<IVER>` and laughs | every edge | 235 | 65.07 | 64.46 |
| `<IVER>` and laughs | speech onsets | 235 | 64.54 | 64.75 |
| every tag but `<SIL>` | every edge | 725 | 69.21 | 68.80 |
| every tag but `<SIL>` | speech onsets | 725 | 70.91 | 71.01 |
| + `<EXT-…>` / `<HES-…>` words kept as text | speech onsets | 686 | 70.64 | 70.47 |
| + `yknow` / `hm` / `mm` respelt (**now**) | speech onsets | 725 | **70.93** | n/a |
| + pauses written as commas | speech onsets | 725 | 70.93 | n/a |

Reading it:

- **Cutting at every tag is the big one: +4.6.** Cutting at laughs alone is +0.5, because
  laughs are a small share of what was hidden in the segments.
- **Speech onsets add +1.7 once the segments are clean.** On the old segments the same change
  is slightly negative (-0.7 here, -0.25 on the 20 interviews of bf101): there, the edges of
  the noise intervals were collecting chance hits from an alignment that was already off.
- Keeping `<EXT-…>` and `<HES-…>` words as text is slightly worse than cutting them out, so
  they are cut.
- Respelling is worth +0.02 on these four interviews, which is nothing. It is kept because
  it is right (no speaker said the `k` in `yknow`), not because it scores.
- Net: **64.62 → 70.93** at the published settings, on the same four interviews.

### TIMIT: 10 random clips, then the 100 clips of bf107 / bf108

Ten clips are 320 boundaries, which is too few to rank anything: one clip moves f1 by a
point. The same definitions were therefore re-scored on the 100 clips already aligned as
bf108 (bonus 30) and bf107 (bonus 0), which costs no alignment.

| boundaries | 10 clips, bonus 30 | 100 clips, bonus 30 | 100 clips, bonus 0 |
|---|---|---|---|
| both edges of every interval (before) | 69.73 | **73.07** | 67.16 |
| onsets of every interval (**now**) | 72.20 | 73.07 | 67.88 |
| onsets, closure + release as one stop starting at the closure | 72.38 | 70.98 | 70.96 |
| onsets, closure + release as one stop starting at the release | n/a | 69.10 | 65.84 |

- **Onsets** is free on TIMIT: the same or slightly better, and the same rule as Buckeye.
- **Merging a stop's closure and release was rejected.** It is the more principled reading
  (the aligner emits one `t`), and it helps by +3.8 at bonus 0, but it costs -2.1 at the
  published bonus. With the bonus on, the model's onsets snap to acoustic change points, and
  for more than half of the stops that is the release, not the start of the closure. Keeping
  both edges gives the model credit for either. If a future model places stops at the
  closure, this is the first thing to revisit.

### The two-token phones (`ay` → `a ɪ`, `ch` → `t ʃ`)

A diphthong or an affricate is one hand-labelled phone and two std tokens, so `mbfa-latin`
predicts one more onset than the ground truth has, for about 9% of its tokens. ph66 and MFA
write these as one symbol and do not.

The scorer can count the pair as one onset: `merge_two_token_phones=True` in
`check_method_alignment` (same word only, so `that she` stays `t` + `ʃ`). It is **off by
default**, because the effect depends on the bonus and is negative at the published one:

| f1@25, tokens → merged | bonus 30 | bonus 0 |
|---|---|---|
| TIMIT, 100 clips, onsets | 73.07 → 71.89 | 67.88 → 68.65 |
| Buckeye, 4 interviews, new segments and onsets | 70.91 → 70.35 | 71.01 → 72.32 |
| Buckeye, 20 interviews (bf101 / bf100), old segments | 64.26 → 63.38 | 61.83 → 62.73 |

Merging always raises precision and lowers recall. At bonus 30 the second token's onset
lands on *some* true boundary often enough (roughly 4 times in 10 on Buckeye) that removing
it loses more hits than false alarms. That is partly luck at a 25 ms tolerance, so read a std
row's precision with this in mind: about 9% of its predictions have no ground truth of their
own.

### What this does to MFA

The ground-truth change is not specific to BFA. MFA's existing alignments, on the files and
segments they still match:

| f1@25 | every edge | onsets |
|---|---|---|
| MFA, Buckeye (20 interviews, 505 of 991 segments still match) | 79.75 | 82.16 |
| MFA, TIMIT (100 clips) | 79.21 | 80.62 |

So onsets-only is a fairer ground truth for everyone and helps MFA at least as much as BFA.
The Buckeye re-segmentation is different: it removes audio that neither aligner could have
got right from the text it was given.

## What has to be re-run

Nothing under `tmp/data` was regenerated, because sweeps were reading it. When they finish:

```bash
python evaluation/dataset/timit_parser.py
python evaluation/dataset/buckeye_parser.py
```

After that:

- **Every earlier row in `results/*.csv` is on the old ground truth** and is not comparable
  with a new one. That includes the MFA reference (78.51 on full Buckeye).
- **MFA has to be re-aligned on Buckeye** (`align_with_mfa.py`): its `.mfa.json` files are
  matched to segments by start time, and the segments moved.
- **A Buckeye sweep costs about 2.7 times as much.** There are 2.8 times as many segments,
  and each is padded to `DURATION_MAX_SEGMENTS` (30 s) however short it is: 420 s per
  interview against 153 s on this (loaded) machine. No segment is longer than 30 s, so the
  padding is the thing to lower if that is too slow.

## Things seen along the way that are not mapping

These are G2P outputs on ordinary English words, found while comparing the G2P's phones with
Buckeye's dictionary. They were left alone: respelling them in the evaluation text would hide
a library error from the benchmark.

| word | std G2P | expected |
|---|---|---|
| `yeah` | `j i ə` | `j æ` |
| `buy`, `guy` | `b w i`, `ɡ w i` | `b a ɪ`, `ɡ a ɪ` |
| `maybe` | `m e ɪ b` | `m e ɪ b i` |
| `once` | `ɑ n s` | `w ə n s` |
| `upon` | `j u p ɑ n` | `ə p ɑ n` |
| `when`, `which`, `whether` | `h w ɛ n`, … | `w ɛ n` (an extra `h` on every wh-word) |

`yeah` alone is about 1% of Buckeye's words.

The remaining gap is mostly not mapping at all: Buckeye is transcribed as spoken
(`you're from` is `y er f ah m`, 5 phones) and the G2P writes the dictionary form
(`j u ɹ f ɹ ɑ m`, 7 tokens), so `mbfa-latin` predicts about 12% more onsets than Buckeye has
even with the diphthongs merged.
