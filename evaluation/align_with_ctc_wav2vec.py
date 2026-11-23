"""
A phoneme CTC model with torchaudio's forced alignment, as one more model of the sweep.

The standard CTC recipe, with no training of ours: a wav2vec2 model fine-tuned with CTC to
recognise eSpeak phonemes (by default `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, XLSR-53
pretraining + Common Voice phoneme transcripts, 20 ms frames), its per-frame log-posteriors,
and `torchaudio.functional.forced_align` (Viterbi over the known phoneme sequence with the
CTC blank) to place each phoneme. A phoneme's onset is the first frame of its span on the
forced path, its end the frame after its last; nothing is snapped or shifted.

Not run directly: `process_alignments.py` calls `run_ctc` for every experiment whose model is
a `method: ctc` entry of `evaluate.yaml`. Like MFA and MAPS it aligns the segment audio, so
its times are segment-relative; the manifest's `method: CTC` is what tells the scorer.

The phoneme sequence is BFA's own ph66 one: the segment's text through the same eSpeak
phonemizer CUPE-3-en is given (`BFA(phoneme_set="ph66").phonemize_sentence`), without its
SIL tokens (the CTC blank covers pauses). So the run places exactly the onsets a ph66 BFA run
places, and its scores differ from CUPE-3-en's only by the model and the decoder. Each ph66
phoneme is written to the model with the eSpeak symbol it came from, which is in the model's
vocabulary for every English phoneme. The exception is eSpeak's compound symbols (`ɑːɹ`,
`ɔːɹ`, `əl`, `iə`, `aɪɚ`, ...): the model has a token for each, but ph66 counts two phonemes
and two onsets there. The model entry's `compounds` decides what happens to them:
    split    the compound becomes its two parts in the model's tokens (`ɑːɹ` -> `ɑː` `ɹ`),
             each with its own span: every ph66 onset comes from the model
    native   the compound stays one model token (what the model was trained to emit) and its
             span, up to the next phoneme's onset, is divided evenly between its phonemes

Output: one `<clip>.json` per clip, in BFA's ph66 form (`phoneme_id`, `phoneme_label`,
`ipa_label` = the model token, `confidence` = the mean posterior over the span), stamped
`phoneme_set: ph66`, so the scorer's per-group columns work as for a CUPE run.
"""

import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# ph66 symbols whose spelling differs from eSpeak's (and so from the model's vocabulary).
# Every other ph66 symbol is spelt as the model spells it.
PH66_TO_ESPEAK = {"i:": "iː", "u:": "uː", "a:": "ɑː", "o:": "oː", "e:": "eː", "g": "ɡ"}
COMPOUND_MODES = ("split", "native")
SAMPLE_RATE = 16000


def load_ctc_model(hf_model, revision, device):
    """
    The model, its feature extractor, its vocabulary and blank id, its frame step, and the
    fewest samples that give one output frame (its receptive field: 400 for wav2vec2).
    """
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    from huggingface_hub import hf_hub_download

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_model, revision=revision)
    model = Wav2Vec2ForCTC.from_pretrained(hf_model, revision=revision).to(device).eval()
    with open(hf_hub_download(hf_model, "vocab.json", revision=revision), encoding="utf-8") as f:
        vocab = json.load(f)
    blank = model.config.pad_token_id                   # CTC blank = <pad> in HF wav2vec2
    step = int(np.prod(model.config.conv_stride))       # samples per output frame (320)
    min_samples, jump = 1, 1
    for kernel, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        min_samples += (kernel - 1) * jump
        jump *= stride
    if extractor.sampling_rate != SAMPLE_RATE:
        raise ValueError(f"{hf_model} expects {extractor.sampling_rate} Hz audio")
    return model, extractor, vocab, blank, step, min_samples


def _greedy_split(symbol, n_parts, vocab):
    """`symbol` as exactly `n_parts` vocabulary tokens, longest prefix first; None if it is not."""
    parts, rest = [], symbol
    while rest:
        for end in range(len(rest), 0, -1):
            piece = rest[:end]
            if piece in vocab and piece != symbol:
                parts.append(piece)
                rest = rest[end:]
                break
        else:
            return None
    return parts if len(parts) == n_parts else None


def _ph66_token(label, vocab):
    token = PH66_TO_ESPEAK.get(label, label)
    if token not in vocab:
        raise KeyError(f"ph66 {label!r} has no token in the model's vocabulary")
    return token


def build_targets(phonemized, vocab, compounds):
    """
    The model's target tokens for one phonemized segment.

    Returns `(units, positions)`: `units` is a list of `(token, [ph66 positions])`, one per
    target token in order, where a position indexes `phonemized["mipa"]` (one phoneme each).
    In `split` mode every unit has one position; in `native` mode a compound token has
    several. `positions` are the speech phonemes (SIL left out) in order.
    """
    mipa, eipa = phonemized["mipa"], phonemized["eipa"]
    positions = [i for i, label in enumerate(mipa) if label != "SIL"]

    # Group the positions into the eSpeak symbols they came from: a symbol's first phoneme
    # carries it in `eipa`, the rest of a compound carry "-".
    groups = []
    for i in positions:
        if eipa[i] == "-" and groups and groups[-1][1][-1] == i - 1:
            groups[-1][1].append(i)
        else:
            groups.append((eipa[i], [i]))

    units = []
    for symbol, idx in groups:
        if len(idx) == 1:
            token = symbol if symbol in vocab else _ph66_token(mipa[idx[0]], vocab)
            units.append((token, idx))
        elif compounds == "native" and symbol in vocab:
            units.append((symbol, idx))
        else:
            parts = _greedy_split(symbol, len(idx), vocab) \
                or [_ph66_token(mipa[i], vocab) for i in idx]
            units.extend((token, [i]) for token, i in zip(parts, idx))
    return units, positions


def align_segment(model, extractor, blank, step, audio, units, vocab, device):
    """
    Forced-align one segment. Returns `(spans, n_frames)` with one `(start_frame, end_frame,
    mean_log_prob)` per unit, or `(None, n_frames)` when the segment is too short for its
    tokens (CTC needs a frame per token plus one between each repeated pair).
    """
    import torchaudio.functional as AF

    inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    n_frames = log_probs.shape[1]

    ids = [vocab[token] for token, _ in units]
    repeats = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    if n_frames < len(ids) + repeats:
        return None, n_frames

    targets = torch.tensor([ids], dtype=torch.int32, device=device)
    path, scores = AF.forced_align(log_probs, targets, blank=blank)
    path, scores = path[0].cpu(), scores[0].cpu()

    # Walk the path: a new span starts wherever the label changes to a non-blank, or a blank
    # separates two emissions of the same label.
    spans, current, prev = [], None, blank
    for t, label in enumerate(path.tolist()):
        if label != blank and label != prev:
            current = [t, t + 1, [float(scores[t])]]
            spans.append(current)
        elif label != blank:
            current[1] = t + 1
            current[2].append(float(scores[t]))
        prev = label
    if len(spans) != len(ids):
        raise RuntimeError(f"forced path has {len(spans)} spans for {len(ids)} tokens")
    return [(s, e, float(np.mean(lp))) for s, e, lp in spans], n_frames


def segment_phoneme_ts(phonemized, units, spans, frame_ms, seg_ms):
    """
    The `phoneme_ts` of one segment, segment-relative ms. A `native` compound's span runs
    from its onset to the next unit's onset (the segment end after the last) and is divided
    evenly between its phonemes.
    """
    from bournemouth_aligner.ipamappers.ph66_mapper import phoneme_mapped_index

    out = []
    for k, ((token, idx), (s, e, lp)) in enumerate(zip(units, spans)):
        start, end = s * frame_ms, e * frame_ms
        if len(idx) > 1:
            stop = spans[k + 1][0] * frame_ms if k + 1 < len(spans) else min(seg_ms, end)
            stop = max(stop, end)
            bounds = np.linspace(start, stop, len(idx) + 1)
        else:
            bounds = [start, end]
        for j, i in enumerate(idx):
            label = phonemized["mipa"][i]
            out.append({
                "phoneme_id": phoneme_mapped_index[label],
                "phoneme_label": label,
                "ipa_label": token,
                "start_ms": round(float(bounds[j]), 3),
                "end_ms": round(float(bounds[j + 1]), 3),
                "confidence": float(np.exp(lp)),
                "target_seq_idx": i,
            })
    return out


def run_ctc(exp, jobs, ctc_cfg, log):
    """
    Align every job's segments and write one `<clip>.json` each.

    Args:
        exp: the experiment; `hf_model`, `revision`, `compounds` and `lang` are read
        jobs: `(audio_path, srt_path, out_path)` per clip. A clip whose `out_path` exists is
            left alone, which is what resuming an interrupted run means here.
        ctc_cfg: the `ctc:` section of evaluate.yaml (`device`)
        log: called with each line worth keeping
    Returns the run's statistics for the manifest.
    """
    from bournemouth_aligner import BFA

    if exp["compounds"] not in COMPOUND_MODES:
        raise ValueError(f"compounds {exp['compounds']!r}: one of {COMPOUND_MODES}")
    device = ctc_cfg["device"]
    log(f"CTC model {exp['hf_model']}@{exp['revision']} on {device}, compounds={exp['compounds']}")
    model, extractor, vocab, blank, step, min_samples = load_ctc_model(
        exp["hf_model"], exp["revision"], device)
    frame_ms = step / SAMPLE_RATE * 1000.0
    # The phonemizer BFA's ph66 runs use, and nothing else of BFA (no model is loaded).
    phonemizer = BFA(preset=None, model_name=None, ckpt_path=None, lang=exp["lang"],
                     phoneme_set="ph66").phonemizer
    phonemizer.verbose = 0

    stats = {"total_segments": 0, "failed_segments": 0, "too_short_segments": 0,
             "empty_segments": 0, "error_segments": 0, "failed_files": 0, "files_aligned_now": 0,
             "phonemes_aligned": 0, "compound_tokens": 0, "seconds": None}
    t0 = time.time()
    for audio_path, srt_path, out_path in tqdm(jobs, desc="CTC alignment"):
        if os.path.exists(out_path):
            continue
        try:
            with open(srt_path, encoding="utf-8") as f:
                srt_segments = json.load(f)["segments"]
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            audio = audio[:, 0]
            if sample_rate != SAMPLE_RATE:
                import torchaudio.functional as AF
                audio = AF.resample(torch.from_numpy(audio), sample_rate, SAMPLE_RATE).numpy()
                sample_rate = SAMPLE_RATE

            segments = []
            for seg in srt_segments:
                stats["total_segments"] += 1
                text = seg.get("text", "").strip()
                piece = audio[int(seg["start"] * sample_rate):int(seg["end"] * sample_rate)]
                phonemized = phonemizer.phonemize_sentence(text) if text else None
                units, positions = build_targets(phonemized, vocab, exp["compounds"]) \
                    if phonemized and phonemized["mipa"] else ([], [])
                if not units:
                    stats["empty_segments"] += 1
                    continue
                if len(piece) < min_samples:        # shorter than one frame's receptive field
                    stats["too_short_segments"] += 1
                    stats["failed_segments"] += 1
                    continue
                try:
                    spans, _ = align_segment(model, extractor, blank, step, piece, units, vocab,
                                             device)
                except Exception as ex:             # one bad segment must not cost the clip
                    stats["error_segments"] += 1
                    stats["failed_segments"] += 1
                    log(f"ERROR segment @ {seg['start']} s of {srt_path}: "
                        f"{type(ex).__name__}: {ex}")
                    continue
                if spans is None:
                    stats["too_short_segments"] += 1
                    stats["failed_segments"] += 1
                    continue
                phoneme_ts = segment_phoneme_ts(phonemized, units, spans, frame_ms,
                                                (seg["end"] - seg["start"]) * 1000.0)
                stats["phonemes_aligned"] += len(phoneme_ts)
                stats["compound_tokens"] += sum(1 for _, idx in units if len(idx) > 1)
                segments.append({"start": seg["start"], "end": seg["end"], "text": text,
                                 "word_num": phonemized["word_num"],
                                 "words": phonemized["words"], "phoneme_ts": phoneme_ts})
            if not segments:
                stats["failed_files"] += 1
                log(f"no segment aligned: {srt_path}")
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"phoneme_set": "ph66", "lang": exp["lang"], "segments": segments},
                          f, ensure_ascii=False)
            stats["files_aligned_now"] += 1
        except Exception as ex:                 # one bad clip must not stop the run
            stats["failed_files"] += 1
            log(f"ERROR {srt_path}: {type(ex).__name__}: {ex}")
    stats["seconds"] = time.time() - t0
    log(f"CTC alignment finished in {stats['seconds']:.1f} s: {stats}")
    return stats
