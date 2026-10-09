'''
Loading a checkpoint and running it over a whole clip - the part both model families share.

The trunk only ever sees one 120 ms window. A clip is sliced into overlapping windows, every
window is encoded on its own, and the per-window outputs are stitched back onto one clip-long
5 ms frame grid. That is the whole mechanism behind the multilingual claim: nothing in the
model can learn a language's phonotactics, because nothing in the model sees far enough to have
any.

The window stride advances a whole number of output frames (80 ms = 16 frames of 5 ms), so the
stitched grid IS the native conv grid and

    time_ms(frame f) = f * 1000 * frame_stride_samples / sample_rate = f * 5.0

with no correction term anywhere. `windowing.frames_geometry` raises rather than warns if a
geometry breaks that.

p3cupe and p4mbfa have the same trunk and differ in their heads, so a subclass says only how to
build its model from a checkpoint and what its heads are called (`p3cupe/extractor.py`,
`p4mbfa/extractor.py`).
'''

import json
import os

import torch
import torch.nn.functional as F

from . import windowing

# Windows per forward pass. One window per 80 ms of audio, so 512 windows is ~41 s. Capping
# this bounds peak memory on a long clip without changing a single output value, because each
# window is classified independently of every other.
WINDOW_BATCH = 512

# Aligner settings that belong to the checkpoint, not to the caller: the training targets were
# built by this same decoder, so decoding with a different topology than they were built with
# is decoding a different model. `core.py` lets each one be overridden, but the default is
# always what the checkpoint was trained with.
DECODER_KEYS = ('min_phone_frames', 'soft_phone_frames', 'soft_frame_cost', 'sil_skippable',
                'optional_sil', 'class_prior_norm', 'boundary_bonus_weight',
                'boundary_contrast_frames')


# A released model is a directory (or Hub repository) of these two files, written by
# `export_hf.py` in the models repository. `config.json` names its format; BFA reads version 1
# of both.
RELEASE_FILES = ('config.json', 'model.safetensors')
RELEASE_FORMATS = {'p3cupe': 1, 'p4mbfa': 1}


def load_checkpoint(ckpt_path):
    '''
    Read a model from disk, refusing anything that is not one of BFA's two families.

    `ckpt_path` is a RELEASE DIRECTORY (config.json + model.safetensors - what the Hub
    repositories hold) or a TRAINING CHECKPOINT file (.ckpt / .pt). Both come back in the
    training checkpoint's layout - `model_config`, `hyper_parameters`, `state_dict` - so one
    code path builds either.
    '''
    if os.path.isdir(ckpt_path):
        return _load_release(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_config' not in checkpoint:
        raise ValueError(
            f"{ckpt_path} is not a p3cupe or p4mbfa checkpoint (no 'model_config'). BFA >= 2.0 "
            f"only loads those two; the model2i/model3a checkpoints of earlier versions are "
            f"not compatible.")
    return checkpoint


def _load_release(directory):
    '''
    A release directory as a checkpoint dict.

    A release carries more than a training checkpoint does, and all of it is kept:
      - `decoder` is the decoder the model was SELECTED with, which for CUPE-3-en is not the
        one it was trained with (min_phone_frames 6, not 3). It is what `hyper_parameters`
        is filled from, so the model decodes as it was scored.
      - `labels` is the label space as strings. For p4mbfa it becomes the `inventory` stamp
        `MbfaExtractor.check_inventory` compares against this installation's tables.
    '''
    missing = [n for n in RELEASE_FILES if not os.path.isfile(os.path.join(directory, n))]
    if missing:
        raise ValueError(f"{directory} is not a released BFA model: it has no "
                         f"{' or '.join(missing)}")
    with open(os.path.join(directory, 'config.json'), encoding='utf-8') as f:
        config = json.load(f)
    fmt, version = config.get('format'), config.get('format_version')
    if fmt not in RELEASE_FORMATS:
        raise ValueError(f"{directory}: config.json is format {fmt!r}, not one of "
                         f"{sorted(RELEASE_FORMATS)}")
    if version != RELEASE_FORMATS[fmt]:
        raise ValueError(f"{directory}: {fmt} format version {version}, but this BFA reads "
                         f"version {RELEASE_FORMATS[fmt]}; upgrade bournemouth-forced-aligner")
    absent = [k for k in ('model_config', 'geometry', 'decoder', 'labels') if k not in config]
    if absent:
        raise ValueError(f"{directory}: config.json has no {absent}")

    from safetensors.torch import load_file
    state = load_file(os.path.join(directory, 'model.safetensors'))
    checkpoint = {
        'model_config': dict(config['model_config']),
        'hyper_parameters': {**config['geometry'], **config['decoder']},
        # Under the `model.` prefix a training checkpoint's tensors carry.
        'state_dict': {f'model.{k}': v for k, v in state.items()},
        'release': config,
    }
    if checkpoint_family(checkpoint) != fmt:
        raise ValueError(f"{directory}: config.json says format {fmt!r} but its model_config "
                         f"is a {checkpoint_family(checkpoint)} one")
    labels = config['labels']
    if fmt == 'p4mbfa':
        checkpoint['inventory'] = {
            'lang_group': labels['lang_group'], 'tokens': list(labels['tokens']),
            'n_tokens': len(labels['tokens']),
            'gold_fingerprint': labels.get('gold_fingerprint')}
    return checkpoint


def checkpoint_family(checkpoint):
    '''
    'p3cupe' or 'p4mbfa', read off `model_config`: the two name their heads differently, and
    that is what decides which phoneme set a checkpoint can decode.
    '''
    config = checkpoint['model_config']
    if 'n_tokens' in config and 'lang_group' in config:
        return 'p4mbfa'
    if 'phoneme_classes' in config:
        return 'p3cupe'
    raise ValueError(f"unrecognised model_config (keys {sorted(config)}): neither a p3cupe "
                     f"nor a p4mbfa checkpoint")


class BaseExtractor:
    '''
    A loaded checkpoint, plus the clip-level forward pass.

    The checkpoint is the source of truth for both halves of the configuration: `model_config`
    gives the tensor shapes, and `hyper_parameters` gives the windowing geometry and the
    decoder settings the model was trained against. Reading them from the file rather than
    from defaults is what stops a checkpoint being decoded with a topology it never saw.

    Attributes:
        family: 'p3cupe' or 'p4mbfa'.
        release: a released model's config.json as a dict; None for a training checkpoint.
        hp: the checkpoint's training hyperparameters, as a plain dict. For a released model,
            its geometry and the decoder it was published with.
        decoder_hp: the subset of them the aligner needs (`DECODER_KEYS`).
        classes_dim, groups_dim: logits of the phoneme and the group head.
        tones_dim: logits of the tone head, or None for a family without one.
        frames_per_window, frame_stride_samples, stride_frames, ms_per_frame: the geometry.
    '''

    family = None
    # Fallbacks for hyperparameters an OLD checkpoint may not carry. None = the family's
    # checkpoints always carry every key, so a missing one is an error and not a default.
    DECODER_DEFAULTS = None
    tones_dim = None

    def __init__(self, ckpt_path, device='cpu', sample_rate=16000, verbose=1, checkpoint=None):
        if checkpoint is None:
            checkpoint = load_checkpoint(ckpt_path)
        self.ckpt_path = ckpt_path
        # The whole config.json of a released model (labels, training provenance, selection
        # metrics), or None for a training checkpoint.
        self.release = checkpoint.get('release')

        hp = checkpoint.get('hyper_parameters', {}) or {}
        if hasattr(hp, '__dict__') and not isinstance(hp, dict):
            hp = vars(hp)                              # argparse.Namespace
        self.hp = dict(hp)
        self.decoder_hp = self._read_decoder_hp()
        # 'mel' reads the onset bonus off the log-mel spectrum, which has no parameters;
        # 'tap' reads it off the model's own first conv. Only matters when a bonus weight is
        # set, but which signal the checkpoint was trained with is not the caller's choice.
        self.flux_source = str(self.hp.get('boundary_flux_source', 'mel'))

        model = self._build_model(checkpoint, verbose)
        self.device = device
        self.model = model.to(device).eval()

        # Geometry, from the checkpoint. A model decoded at a different window stride than it
        # was trained at stitches frames describing different moments together.
        self.sample_rate = int(self.hp.get('sample_rate', sample_rate))
        self.window_size_ms = float(self.hp.get('window_size_ms', 120.0))
        self.stride_ms = float(self.hp.get('window_stride_ms', 80.0))
        self.frames_per_window = int(model.frames_per_window)
        self.frame_stride_samples = int(model.frame_stride_samples)
        self.stride_frames, self.stride_size_wav, self.window_size_wav = \
            windowing.frames_geometry(self.window_size_ms, self.stride_ms, self.sample_rate,
                                      self.frames_per_window, self.frame_stride_samples)
        self.ms_per_frame = 1000.0 * self.frame_stride_samples / self.sample_rate
        self._mel_cache = None

        if verbose >= 2:
            print(f"{self.family} loaded:", model.describe(self.sample_rate))
            print(f"{self.family} geometry: window {self.window_size_ms:g} ms / stride "
                  f"{self.stride_ms:g} ms = {self.stride_frames} frames, "
                  f"{self.ms_per_frame:g} ms/frame, "
                  f"receptive field "
                  f"{1000.0 * model.receptive_field_samples / self.sample_rate:.1f} ms")

    # ------------------------------------------------------------------ subclass hooks

    def _build_model(self, checkpoint, verbose):
        '''Rebuild the model from `checkpoint`, load its weights, set the head sizes.'''
        raise NotImplementedError

    def _heads(self, emb):
        '''[N, T, D] embeddings -> `{'ph': logits, 'group': logits[, 'tone': logits]}`.'''
        raise NotImplementedError

    def check_inventory(self, inventory):
        '''
        Raise unless `inventory` is the label space this checkpoint's heads were trained on.
        Aligning against any other one runs without error and returns nonsense.
        '''
        if self.classes_dim != inventory.n_classes:
            raise ValueError(
                f"checkpoint emits {self.classes_dim} phoneme classes but the "
                f"'{inventory.name}' inventory has {inventory.n_classes}")

    # ------------------------------------------------------------------ plumbing

    def _read_decoder_hp(self):
        if self.DECODER_DEFAULTS is not None:
            return {k: self.hp.get(k, v) for k, v in self.DECODER_DEFAULTS.items()}
        missing = [k for k in DECODER_KEYS if k not in self.hp]
        if missing:
            raise ValueError(f"{self.ckpt_path}: hyper_parameters is missing {missing}; not a "
                             f"{self.family} checkpoint BFA can decode")
        return {k: self.hp[k] for k in DECODER_KEYS}

    def to(self, device):
        self.device = device
        self.model.to(device)
        self._mel_cache = None
        return self

    def eval(self):
        self.model.eval()
        return self

    def spectral_lengths(self, wav_lens):
        '''Valid stitched frames per clip, from the real (unpadded) sample lengths.'''
        if not isinstance(wav_lens, torch.Tensor):
            wav_lens = torch.tensor(wav_lens, dtype=torch.long)
        return windowing.spectral_lengths(wav_lens, self.window_size_wav, self.stride_size_wav,
                                          self.frames_per_window, self.stride_frames)

    @torch.no_grad()
    def encode_clip(self, audio_batch, return_embeddings=False, return_flux=False):
        '''
        Windowed forward over whole clips.

        Args:
            audio_batch: [B, L] or [B, 1, L] padded waveforms at `sample_rate`, RMS-normalised
                by the caller (the models were trained on RMS-normalised audio).
            return_embeddings: also return the [B, T, D] pre-classifier embeddings.
            return_flux: also return [B, T] acoustic change entering each frame, for the
                aligner's onset bonus.
        Returns:
            A dict of [B, T, ...] tensors on the model's device, where T is what the padded
            clip length implies:
                'ph'     phoneme logits      'group'  phoneme-group logits
                'tone'   tone logits, or None for a family with no tone head
                'emb'    embeddings, or None  'flux'   acoustic change, or None
        '''
        if audio_batch.dim() == 3:
            audio_batch = audio_batch.squeeze(1)
        elif audio_batch.dim() == 1:
            audio_batch = audio_batch.unsqueeze(0)
        audio_batch = audio_batch.to(self.device)

        windows = windowing.slice_windows(audio_batch, self.sample_rate,
                                          self.window_size_ms, self.stride_ms)
        b, n_win, win_len = windows.shape
        flat = windows.reshape(-1, win_len)

        want_tap = return_flux and self.flux_source == 'tap'
        chunks, chunks_e, chunks_tap = {}, [], []
        for i in range(0, flat.shape[0], WINDOW_BATCH):
            part = flat[i:i + WINDOW_BATCH]
            if want_tap:
                emb, tap = self.model.encode(part, return_tap=True)
                chunks_tap.append(tap)
            else:
                emb = self.model.encode(part)
            for name, logits in self._heads(emb).items():
                chunks.setdefault(name, []).append(logits)
            if return_embeddings:
                chunks_e.append(emb)

        def _stitch(parts):
            if not parts:
                return None
            x = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            return windowing.stitch_windows(
                x.reshape(b, n_win, self.frames_per_window, -1),
                num_windows_total=n_win, frames_per_window=self.frames_per_window,
                stride_frames=self.stride_frames)

        out = {'tone': None}
        out.update({name: _stitch(parts) for name, parts in chunks.items()})
        out['emb'] = _stitch(chunks_e) if return_embeddings else None

        out['flux'] = None
        if return_flux:
            t_full = out['ph'].shape[1]
            if self.flux_source == 'mel':
                out['flux'] = self._mel_flux(audio_batch, t_full)
            elif self.flux_source == 'tap':
                out['flux'] = self._tap_flux(chunks_tap, b, n_win, _stitch)
            else:
                raise ValueError(f"boundary_flux_source must be 'mel' or 'tap', got "
                                 f"{self.flux_source!r}")
        return out

    @torch.no_grad()
    def _mel_flux(self, audio_batch, t_full):
        '''
        L2 change of the log-mel spectrum ENTERING each frame, on the model's own frame grid:
        25 ms window, hop = one frame, `center=True`, so mel frame t is centred on sample
        t * frame_stride exactly like conv frame t. [B, t_full], entry 0 = 0.

        Log-mel rather than the model's own tap because it has no parameters, so training
        cannot erode it - the learned tap's boundary score fell as the front end specialised
        for phone identity (`model3p/memory.md`, pa01c).
        '''
        import torchaudio
        wav = audio_batch.reshape(audio_batch.shape[0], -1).float()
        cache = self._mel_cache
        if cache is None or cache[0] != wav.device:
            transform = torchaudio.transforms.MelSpectrogram(
                self.sample_rate, n_fft=512, win_length=400,
                hop_length=self.frame_stride_samples, n_mels=80, center=True).to(wav.device)
            self._mel_cache = cache = (wav.device, transform)
        log_mel = (cache[1](wav) + 1e-6).log().transpose(1, 2)        # [B, frames, 80]
        n = min(log_mel.shape[1], t_full)
        flux = torch.zeros(wav.shape[0], t_full, device=wav.device)
        flux[:, 1:n] = (log_mel[:, 1:n] - log_mel[:, :n - 1]).norm(dim=-1)
        return flux

    @torch.no_grad()
    def _tap_flux(self, chunks_tap, b, n_win, stitch):
        '''
        Cosine change of the `feature_extractor` tap entering each frame, for a checkpoint
        trained with `boundary_flux_source: tap`.

        Computed inside each window as a [N, frames] signal and then stitched, rather than
        stitching the 512-d tap and differencing it: a window's first frame has no predecessor
        inside the window, so it is masked out of the stitch and takes its value from the
        overlapping window instead.
        '''
        tap = torch.cat(chunks_tap, dim=0) if len(chunks_tap) > 1 else chunks_tap[0]
        e = F.normalize(tap.float(), dim=-1)
        flux = torch.zeros(e.shape[:2], device=e.device)
        flux[:, 1:] = 1.0 - (e[:, 1:] * e[:, :-1]).sum(-1)
        valid = torch.ones_like(flux)
        valid[:, 0] = 0.0
        stitched = stitch([torch.stack([flux * valid, valid], dim=-1)])
        out = stitched[..., 0] / stitched[..., 1].clamp(min=1e-6)
        out[:, 0] = 0.0
        return out
