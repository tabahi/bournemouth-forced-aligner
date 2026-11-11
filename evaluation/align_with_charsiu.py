"""
Charsiu's text-dependent W2V2-FS aligner, as one more model of the sweep.

Charsiu (Zhu, Zhang & Jurgens, "Phone-to-audio alignment without text: A semi-supervised
approach", ICASSP 2022; https://github.com/lingjzhu/charsiu, MIT licence) released two kinds of
English aligner. This module runs only W2V2-FS (by default `charsiu/en_w2v2_fs_10ms`), the one
that never saw a boundary label: wav2vec2-base plus a BERT phone encoder and one cross-attention
layer between audio frames and the transcript's phones, trained with wav2vec2's contrastive loss
and a forward-sum (CTC-over-phone-positions) loss on Common Voice, LibriSpeech and, in its last
round, the TIMIT train set (text only). The released checkpoint has no model card, so whether it
is that last round is unknown; score TIMIT on `split: test` as well. W2V2-FC, a frame classifier
trained on MFA alignments, is not run: its boundaries come from another aligner.

Nothing here is trained. The run is Charsiu's own inference:
    * text -> phones by g2p_en, stress digits removed, the 39 CMU ARPAbet phones of Charsiu's
      tokenizer (`charsiu/tokenizer_en_cmu`), `[SIL]` added at both ends; punctuation and
      unknown words' non-phone tokens dropped, as `CharsiuPreprocessor_en` does
    * audio normalised to zero mean, unit variance (`Wav2Vec2FeatureExtractor`, as Charsiu)
    * the attention energies, frames x phones, 10 ms per frame
then one of two decoders (the model entry's `decoder`):
    viterbi  the best monotonic path through log-softmax(energy) over the phone positions: every
             frame belongs to one phone, the phones keep their order and each gets >= 1 frame.
             This is the forward-sum loss's own path with the blank left out (the "DTW" forced
             alignment of the paper), and the path a forward-sum TTS aligner decodes with.
    argmax   Charsiu's released `charsiu_attention_aligner.align`: each frame takes its most
             attended phone, runs of one label merge. Phones can repeat or vanish, so the
             onsets need not follow the transcript one-to-one. Kept as a check.
A phone's onset is the first frame of its span times 10 ms, as Charsiu's `seq2duration` writes
it; nothing is shifted.

Not run directly: `process_alignments.py` calls `run_charsiu` for every experiment whose model is
a `method: charsiu` entry of `evaluate.yaml`. Its times are segment-relative (`method: CHARSIU`).

Output: one `<clip>.json` per clip in MFA's form - `phone` is the ARPAbet label, `[SIL]` written
`sil` - so the scorer reads it through its ARPAbet -> ph66 table, as it reads MFA.
"""

import json
import os
import re
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SAMPLE_RATE = 16000
DECODERS = ("viterbi", "argmax")
TOKENIZER = "charsiu/tokenizer_en_cmu"
SIL = "[SIL]"


# --------------------------------------------------------------------------------------
# The model: the inference half of Charsiu's `Wav2Vec2ForAttentionAlignment` (src/models.py),
# with its module names, so the released weights load as they are. One difference: Charsiu's
# ConvBank calls `F.dropout` without `training=`, so its released inference drops 10% of the
# frame features at random on every call. Here dropout is off in eval mode, which makes the run
# deterministic; with Charsiu's dropout set to 0 the two give the same energies (checked to
# 2e-5 on a TIMIT clip, 2026-11-10).
# --------------------------------------------------------------------------------------

class _ConvBank(nn.Module):
    def __init__(self, input_dim, output_dim, kernels, cnn_size, hidden_size, dropout):
        super().__init__()
        self.drop_p = dropout
        self.in_linear = nn.Linear(input_dim, hidden_size)
        self.cnns = nn.ModuleList(nn.Conv1d(hidden_size, cnn_size, k, padding=k // 2)
                                  for k in kernels)
        self.out_linear = nn.Linear(cnn_size * len(kernels), output_dim)

    def forward(self, features):
        hidden = F.dropout(F.relu(self.in_linear(features)), p=self.drop_p,
                           training=self.training).transpose(1, 2)
        hidden = torch.cat([cnn(hidden) for cnn in self.cnns], dim=1).transpose(1, 2)
        hidden = F.dropout(F.relu(hidden), p=self.drop_p, training=self.training)
        return self.out_linear(hidden)


class _Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)       # in the checkpoint, unused by Charsiu

    def energy(self, frame_hidden, phone_hidden):
        return torch.bmm(self.q(frame_hidden), self.k(phone_hidden).transpose(2, 1))


def _build_model_class():
    from transformers import BertConfig, BertForMaskedLM, Wav2Vec2ForPreTraining

    class BertForMaskedPhoneLM(BertForMaskedLM):
        def __init__(self, config):
            super().__init__(config)
            self.cnn = _ConvBank(config.hidden_size, config.hidden_size, config.convbank,
                                 config.hidden_size, config.hidden_size,
                                 config.hidden_dropout_prob)

    class CharsiuFS(Wav2Vec2ForPreTraining):
        def __init__(self, config):
            super().__init__(config)
            bert_config = BertConfig(**{
                key: getattr(config, f"bert_{key}") for key in (
                    "attention_probs_dropout_prob", "hidden_act", "hidden_dropout_prob",
                    "hidden_size", "initializer_range", "intermediate_size", "layer_norm_eps",
                    "max_position_embeddings", "num_attention_heads", "num_hidden_layers",
                    "pad_token_id", "position_embedding_type", "type_vocab_size", "vocab_size")})
            bert_config.convbank = config.bert_convbank
            self.bert = BertForMaskedPhoneLM(bert_config)
            self.cnn = _ConvBank(config.hidden_size, config.bert_hidden_size,
                                 config.bert_convbank, config.bert_hidden_size,
                                 config.bert_hidden_size, config.hidden_dropout)
            self.attention = _Attention(config.bert_hidden_size)

        def energy(self, input_values, phone_ids):
            """Attention energies, (1, frames, phones), for one unpadded segment."""
            frame_hidden = self.cnn(self.wav2vec2(input_values)[0])
            phone_hidden = self.bert.bert(input_ids=phone_ids,
                                          attention_mask=torch.ones_like(phone_ids))[0]
            return self.attention.energy(frame_hidden, phone_hidden)

    return CharsiuFS


def load_charsiu(hf_model, revision, device):
    """The model, the audio normaliser, the phone vocabulary, the frame step and the fewest
    samples that give one output frame."""
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor

    model = _build_model_class().from_pretrained(hf_model, revision=revision).to(device).eval()
    # Charsiu builds the same extractor in code (CharsiuPreprocessor_en); the model repo has none.
    extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE,
                                         padding_value=0.0, do_normalize=True,
                                         return_attention_mask=False)
    with open(hf_hub_download(TOKENIZER, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    step = int(np.prod(model.config.conv_stride))       # samples per output frame (160)
    min_samples, jump = 1, 1
    for kernel, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        min_samples += (kernel - 1) * jump
        jump *= stride
    return model, extractor, vocab, step, min_samples


# --------------------------------------------------------------------------------------
# Text -> phone ids (Charsiu's CharsiuPreprocessor_en.get_phones_and_words + get_phone_ids)
# --------------------------------------------------------------------------------------

_WORD = re.compile(r"\w+\d?")
_STRESS = re.compile(r"\d")


def text_to_phones(g2p, text, vocab):
    """ARPAbet phones (stress removed) of `text` with `[SIL]` at both ends, as Charsiu does.
    Returns `[]` when the text has no phone."""
    from itertools import groupby
    words = [tuple(g) for keep, g in groupby(g2p(text), key=lambda x: x != " ") if keep]
    phones = [_STRESS.sub("", p) for word in words if _WORD.search(word[0]) for p in word]
    if not phones:
        return []
    unknown = [p for p in phones if p not in vocab]
    if unknown:
        raise KeyError(f"phones outside Charsiu's vocabulary: {unknown}")
    return [SIL] + phones + [SIL]


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------

def viterbi_spans(log_att):
    """
    Best monotonic path through `log_att` (frames x phones, log-softmax over phones): each frame
    takes one phone, phones in order, each >= 1 frame. Returns `(start, end)` frames per phone,
    or None when there are fewer frames than phones.

    torchaudio's CTC forced alignment with targets 1..N (all distinct, so no blank is ever
    needed between them) and a blank column it can never afford is exactly this path.
    """
    import torchaudio.functional as AF
    n_frames, n_phones = log_att.shape
    if n_frames < n_phones:
        return None
    emissions = torch.cat([torch.full((n_frames, 1), -1e4, dtype=log_att.dtype,
                                      device=log_att.device), log_att], dim=1)[None]
    targets = torch.arange(1, n_phones + 1, dtype=torch.int32, device=log_att.device)[None]
    path, _ = AF.forced_align(emissions, targets, blank=0)
    path = path[0].cpu().numpy()
    if (path == 0).any():
        raise RuntimeError("monotonic path used the blank")
    starts = np.searchsorted(path, np.arange(1, n_phones + 1), side="left")
    ends = np.searchsorted(path, np.arange(1, n_phones + 1), side="right")
    if (ends <= starts).any():
        raise RuntimeError("monotonic path skipped a phone")
    return list(zip(starts.tolist(), ends.tolist()))


def argmax_runs(log_att):
    """Charsiu's own decoding: per-frame argmax over phone positions, runs merged by phone
    LABEL is done by the caller; here runs of the same position. Returns `(position, start,
    end)` per run."""
    best = log_att.argmax(dim=-1).cpu().numpy()
    runs, start = [], 0
    for t in range(1, len(best) + 1):
        if t == len(best) or best[t] != best[start]:
            runs.append((int(best[start]), start, t))
            start = t
    return runs


def segment_phoneme_ts(phones, log_att, decoder, frame_ms):
    """`phoneme_ts` (segment-relative ms) of one segment, or None when it is too short."""
    out = []
    if decoder == "viterbi":
        spans = viterbi_spans(log_att)
        if spans is None:
            return None
        for i, (s, e) in enumerate(spans):
            out.append((phones[i], s, e, float(log_att[s:e, i].mean())))
    else:
        # seq2duration merges consecutive frames with the same phone LABEL, not position.
        for pos, s, e in argmax_runs(log_att):
            if out and out[-1][0] == phones[pos]:
                label, s0, _, lp = out[-1]
                out[-1] = (label, s0, e, lp)
            else:
                out.append((phones[pos], s, e, float(log_att[s:e, pos].mean())))
    return [{"phone": "sil" if label == SIL else label,
             "start_ms": round(s * frame_ms, 3), "end_ms": round(e * frame_ms, 3),
             "confidence": float(np.exp(lp))} for label, s, e, lp in out]


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------

def run_charsiu(exp, jobs, charsiu_cfg, log):
    """
    Align every job's segments and write one `<clip>.json` each.

    Args:
        exp: the experiment; `hf_model`, `revision` and `decoder` are read
        jobs: `(audio_path, srt_path, out_path)` per clip. A clip whose `out_path` exists is
            left alone, which is what resuming an interrupted run means here.
        charsiu_cfg: the `charsiu:` section of evaluate.yaml (`device`)
        log: called with each line worth keeping
    Returns the run's statistics for the manifest.
    """
    from g2p_en import G2p

    if exp["decoder"] not in DECODERS:
        raise ValueError(f"decoder {exp['decoder']!r}: one of {DECODERS}")
    device = charsiu_cfg["device"]
    log(f"Charsiu model {exp['hf_model']}@{exp['revision']} on {device}, "
        f"decoder={exp['decoder']}")
    model, extractor, vocab, step, min_samples = load_charsiu(exp["hf_model"], exp["revision"],
                                                              device)
    frame_ms = step / SAMPLE_RATE * 1000.0
    g2p = G2p()

    stats = {"total_segments": 0, "failed_segments": 0, "too_short_segments": 0,
             "empty_segments": 0, "error_segments": 0, "failed_files": 0, "files_aligned_now": 0,
             "phones_aligned": 0, "phones_in_transcripts": 0, "seconds": None}
    t0 = time.time()
    for audio_path, srt_path, out_path in tqdm(jobs, desc="Charsiu alignment"):
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
                phones = text_to_phones(g2p, text, vocab) if text else []
                if not phones:
                    stats["empty_segments"] += 1
                    continue
                piece = audio[int(seg["start"] * sample_rate):int(seg["end"] * sample_rate)]
                if len(piece) < min_samples:        # shorter than one frame's receptive field
                    stats["too_short_segments"] += 1
                    stats["failed_segments"] += 1
                    continue
                try:
                    inputs = extractor(piece, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                    ids = torch.tensor([[vocab[p] for p in phones]], device=device)
                    with torch.inference_mode():
                        energy = model.energy(inputs.input_values.to(device), ids)
                    log_att = torch.log_softmax(energy[0].float(), dim=-1)
                    phoneme_ts = segment_phoneme_ts(phones, log_att, exp["decoder"], frame_ms)
                except Exception as ex:             # one bad segment must not cost the clip
                    stats["error_segments"] += 1
                    stats["failed_segments"] += 1
                    log(f"ERROR segment @ {seg['start']} s of {srt_path}: "
                        f"{type(ex).__name__}: {ex}")
                    continue
                if phoneme_ts is None:
                    stats["too_short_segments"] += 1
                    stats["failed_segments"] += 1
                    continue
                stats["phones_in_transcripts"] += len(phones) - 2
                stats["phones_aligned"] += sum(1 for p in phoneme_ts if p["phone"] != "sil")
                segments.append({"start": seg["start"], "end": seg["end"], "text": text,
                                 "phoneme_ts": phoneme_ts})
            if not segments:
                stats["failed_files"] += 1
                log(f"no segment aligned: {srt_path}")
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"segments": segments}, f)
            stats["files_aligned_now"] += 1
        except Exception as ex:                 # one bad clip must not stop the run
            stats["failed_files"] += 1
            log(f"ERROR {srt_path}: {type(ex).__name__}: {ex}")
    stats["seconds"] = time.time() - t0
    log(f"Charsiu alignment finished in {stats['seconds']:.1f} s: {stats}")
    return stats
