'''
Loading a p3cupe checkpoint and running it over a whole clip.

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
'''

import torch
import torch.nn.functional as F

from . import windowing
from .model import CupeCNN

# Windows per forward pass. One window per 80 ms of audio, so 512 windows is ~41 s. Capping
# this bounds peak memory on a long clip without changing a single output value, because each
# window is classified independently of every other.
WINDOW_BATCH = 512

# The state_dict prefixes a p3cupe checkpoint carries. Anything else in it (the loss module's
# buffers, optimiser state) belongs to training and is dropped.
_TRUNK_PREFIXES = ('feature_extractor.', 'freq_attention.', 'context.', 'blocks.', 'norm.',
                   'classifier.', 'group_classifier.')


class CupeExtractor:
    '''
    A loaded p3cupe checkpoint, plus the clip-level forward pass.

    The checkpoint is the source of truth for both halves of the configuration: `model_config`
    gives the tensor shapes, and `hyper_parameters` gives the windowing geometry and the
    decoder settings the model was trained against. Reading them from the file rather than
    from defaults is what stops a checkpoint being decoded with a topology it never saw.

    Attributes:
        hp: the checkpoint's training hyperparameters, as a plain dict.
        decoder_hp: the subset of them the aligner needs (see `DECODER_KEYS`).
        frames_per_window, frame_stride_samples, stride_frames, ms_per_frame: the geometry.
    '''

    # Aligner settings that belong to the checkpoint, not to the caller: the training targets
    # were built by this same decoder, so decoding with a different topology than they were
    # built with is decoding a different model. `core.py` lets each one be overridden, but the
    # default is always what the checkpoint was trained with.
    DECODER_DEFAULTS = {'min_phone_frames': 6, 'soft_phone_frames': 4, 'soft_frame_cost': 1.0,
                        'sil_skippable': True, 'optional_sil': True, 'class_prior_norm': True,
                        'boundary_bonus_weight': 0.0, 'boundary_contrast_frames': 17}

    def __init__(self, ckpt_path, device='cpu', sample_rate=16000, verbose=1):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict) or 'model_config' not in checkpoint:
            raise ValueError(
                f"{ckpt_path} is not a p3cupe checkpoint (no 'model_config'). BFA >= 1.3 only "
                f"loads p3cupe; the model2i/model3a checkpoints of earlier versions are not "
                f"compatible.")

        hp = checkpoint.get('hyper_parameters', {}) or {}
        if hasattr(hp, '__dict__') and not isinstance(hp, dict):
            hp = vars(hp)                              # argparse.Namespace
        self.hp = dict(hp)
        self.decoder_hp = {k: self.hp.get(k, v) for k, v in self.DECODER_DEFAULTS.items()}
        # 'mel' reads the onset bonus off the log-mel spectrum, which has no parameters;
        # 'tap' reads it off the model's own first conv. Only matters when a bonus weight is
        # set, but which signal the checkpoint was trained with is not the caller's choice.
        self.flux_source = str(self.hp.get('boundary_flux_source', 'mel'))

        model = CupeCNN()
        model.load_config_state_dict(checkpoint['model_config'])
        model.make_model()
        state = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
        state = {k: v for k, v in state.items() if k.startswith(_TRUNK_PREFIXES)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"checkpoint is missing weights for: {sorted(missing)[:8]}")
        if unexpected and verbose >= 1:
            print(f"WARN: ignoring unexpected checkpoint keys: {sorted(unexpected)[:8]}")

        self.device = device
        self.model = model.to(device).eval()
        self.classes_dim = model.classes_dim
        self.groups_dim = model.groups_dim

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
            print("p3cupe loaded:", model.describe(self.sample_rate))
            print(f"p3cupe geometry: window {self.window_size_ms:g} ms / stride "
                  f"{self.stride_ms:g} ms = {self.stride_frames} frames, "
                  f"{self.ms_per_frame:g} ms/frame, "
                  f"receptive field "
                  f"{1000.0 * model.receptive_field_samples / self.sample_rate:.1f} ms")

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
                by the caller (the model was trained on RMS-normalised audio).
            return_embeddings: also return the [B, T, D] pre-classifier embeddings.
            return_flux: also return [B, T] acoustic change entering each frame, for the
                aligner's onset bonus.
        Returns:
            (logits_classes, logits_groups, embeddings|None, flux|None), each [B, T, ...] on
            the model's device, where T is what the padded clip length implies.
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
        chunks_c, chunks_g, chunks_e, chunks_tap = [], [], [], []
        for i in range(0, flat.shape[0], WINDOW_BATCH):
            part = flat[i:i + WINDOW_BATCH]
            if want_tap:
                emb, tap = self.model.encode(part, return_tap=True)
                chunks_tap.append(tap)
            else:
                emb = self.model.encode(part)
            logits_c, logits_g = self.model.classify(emb)
            chunks_c.append(logits_c)
            chunks_g.append(logits_g)
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

        logits_c = _stitch(chunks_c)
        logits_g = _stitch(chunks_g)
        embeddings = _stitch(chunks_e) if return_embeddings else None

        flux = None
        if return_flux:
            t_full = logits_c.shape[1]
            if self.flux_source == 'mel':
                flux = self._mel_flux(audio_batch, t_full)
            elif self.flux_source == 'tap':
                flux = self._tap_flux(chunks_tap, b, n_win, _stitch)
            else:
                raise ValueError(f"boundary_flux_source must be 'mel' or 'tap', got "
                                 f"{self.flux_source!r}")
        return logits_c, logits_g, embeddings, flux

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
