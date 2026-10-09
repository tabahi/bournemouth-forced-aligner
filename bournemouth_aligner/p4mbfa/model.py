'''
p4mbfa: multilingual CNN phoneme encoder with one fixed and two replaceable heads.

Takes a raw 16 kHz waveform window (120 ms) and emits, per 5 ms frame, a posterior over ONE
LANGUAGE GROUP's phoneme tokens, over the 15 phoneme groups every language shares, and over
22 tone values. This is the inference copy of `bfa_models/p4mbfa/model.py`, kept close to it on
purpose so the two can be diffed; the only thing removed is the training repository's own
checkpoint wrapper, which `extractor.py` replaces.

The TRUNK is p3cupe's CupeCNN unchanged. The heads are what changed:

    phg_head    FIXED        gold_phg, the 15 broad phoneme groups shared by every language
    ph_head     REPLACEABLE  local_ph, the tokens of ONE language group (208 for 'latin')
    tone_head   REPLACEABLE  tone, 22 values shared by every language (0 = no tone)

A checkpoint is therefore specific to a language group: `model_config['lang_group']` names it,
and `ph_head`'s indices mean nothing against any other group's token list.

## Label spaces (from standard_g2p, checked by the collator against the yaml)

Every local token list starts with the same four specials: 0 `<blank>`, 1 `SIL`, 2 `noise`,
3 `<unk>`. `n_tokens` INCLUDES them, so `ph_head` emits exactly `n_tokens` logits and there is no
extra slot as in p3cupe. `<blank>` (0) is never a target - there is no CTC here - and the
collator raises if it ever appears in a label sequence.

## Geometry (the numbers the rest of the package depends on)

    conv strides 4 * 4 * 5              = 80 samples  = 5.0 ms per frame
    120 ms window                       = 1920 samples = 24 frames
    receptive field (default context)   = 623 samples  = 38.9 ms, centred

Every conv has an ODD kernel and `padding = kernel // 2`, which is what makes frame `f` sit
centred on input sample `f * 80` with no offset correction. Keep it that way: an even kernel
anywhere in the stack shifts every timestamp this project produces by half a frame, and
nothing will raise.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# state_dict prefixes (relative to the model) of each part. Training uses these to load a
# trunk alone and to swap the fine heads; nothing else may own a parameter.
TRUNK_PREFIXES = ('feature_extractor.', 'freq_attention.', 'context.', 'blocks.', 'norm.')
FIXED_HEAD_PREFIXES = ('phg_head.',)
FINE_HEAD_PREFIXES = ('ph_head.', 'tone_head.')

# Config keys that decide tensor SHAPES in the trunk. A checkpoint and the yaml must agree on
# every one of them; there is no silent "the checkpoint wins" as in p3cupe.
TRUNK_SHAPE_KEYS = ('input_wav_length', 'cnn_channels', 'embed_dim', 'context_kernels',
                    'frame_blocks', 'ffn_mult', 'attn_radius', 'attn_heads')
# Config keys that change behaviour but not shapes. Taken from the live yaml on a warm start.
REGULARISATION_KEYS = ('cnn_dropout', 'head_dropout', 'noise_level', 'block_dropout')
# What the fine heads are built for.
FINE_HEAD_KEYS = ('lang_group', 'n_tokens', 'n_tones')


class ConvBlock(nn.Module):
    '''Conv -> BatchNorm -> GELU (-> Dropout), with a residual when shapes allow.'''

    def __init__(self, in_dim, out_dim, kernel, stride=1, dropout=0.0, residual=False):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"kernels must be odd to keep frames centred, got {kernel}")
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=stride,
                              padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = bool(residual) and (in_dim == out_dim) and (stride == 1)

    def forward(self, x):
        y = self.drop(self.act(self.norm(self.conv(x))))
        return x + y if self.residual else y


class LocalAttention(nn.Module):
    '''
    Self-attention restricted to a band of +-`radius` frames, with a learned relative-position
    bias per head. Each layer adds 2 * radius frames to the receptive field, and
    `MbfaCNN._receptive_field` counts it.
    '''

    def __init__(self, dim, heads, radius, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"embed_dim {dim} is not divisible by attn_heads {heads}")
        self.heads, self.radius, self.dropout = int(heads), int(radius), float(dropout)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros(self.heads, 2 * self.radius + 1))

    def forward(self, x):
        n, t, d = x.shape
        q, k, v = self.qkv(x).view(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        rel = torch.arange(t, device=x.device)[None, :] - torch.arange(t, device=x.device)[:, None]
        bias = self.rel_bias[:, rel.clamp(-self.radius, self.radius) + self.radius]   # [H, T, T]
        bias = bias.masked_fill(rel.abs() > self.radius, float('-inf')).to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(n, t, d))


class FrameBlock(nn.Module):
    '''
    Pre-norm residual block on the frame embeddings: [optional banded attention] then a
    per-frame MLP. With `attn_radius=0` it adds depth and width and NO time mixing, so the
    receptive field is untouched. Residual branches are zero-initialised.
    '''

    def __init__(self, dim, ffn_mult=4, attn_radius=0, attn_heads=8, dropout=0.1):
        super().__init__()
        self.attn = None
        if attn_radius > 0:
            self.attn_norm = nn.LayerNorm(dim)
            self.attn = LocalAttention(dim, attn_heads, attn_radius, dropout)
            nn.init.zeros_(self.attn.out.weight)
            nn.init.zeros_(self.attn.out.bias)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        '''x: [N, T, dim]'''
        if self.attn is not None:
            x = x + self.drop(self.attn(self.attn_norm(x)))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class MbfaCNN(nn.Module):
    '''
    The model. `config` is the single source of truth for its shape and is what gets written
    into the checkpoint.

    Args (all also readable from `config`):
        input_wav_length: samples per window. Sets `frames_per_window`.
        cnn_channels, embed_dim, context_kernels, frame_blocks, ffn_mult, attn_radius,
            attn_heads: the trunk, exactly as in p3cupe (see `TRUNK_SHAPE_KEYS`).
        n_phg: gold phoneme groups (15). The FIXED head.
        lang_group, n_tokens: the language group the `ph_head` is built for, and its token
            count INCLUDING the four specials (208 for 'latin').
        n_tones: tone values (22, 0 = no tone). The `tone_head`.
    '''

    def __init__(self, input_wav_length=None, cnn_channels=192, embed_dim=512,
                 context_kernels=(5, 3), cnn_dropout=0.1, head_dropout=0.1,
                 noise_level=0.01, frame_blocks=4, ffn_mult=4, attn_radius=0, attn_heads=8,
                 block_dropout=0.1, n_phg=None, lang_group=None, n_tokens=None, n_tones=None):
        super().__init__()
        self.config = None
        if input_wav_length is not None:
            self.config_model(input_wav_length, cnn_channels, embed_dim, context_kernels,
                              cnn_dropout, head_dropout, noise_level, frame_blocks, ffn_mult,
                              attn_radius, attn_heads, block_dropout, n_phg, lang_group,
                              n_tokens, n_tones)
            self.make_model()

    # ------------------------------------------------------------------ config

    def config_model(self, input_wav_length, cnn_channels, embed_dim, context_kernels,
                     cnn_dropout, head_dropout, noise_level, frame_blocks, ffn_mult,
                     attn_radius, attn_heads, block_dropout, n_phg, lang_group, n_tokens,
                     n_tones):
        for name, value in (('n_phg', n_phg), ('lang_group', lang_group),
                            ('n_tokens', n_tokens), ('n_tones', n_tones)):
            if value is None:
                raise ValueError(f"MbfaCNN needs an explicit {name}")
        self.config = {
            'input_wav_length': int(input_wav_length),
            'cnn_channels': int(cnn_channels),
            'embed_dim': int(embed_dim),
            'context_kernels': [int(k) for k in context_kernels],
            'cnn_dropout': float(cnn_dropout),
            'head_dropout': float(head_dropout),
            'noise_level': float(noise_level),
            'frame_blocks': int(frame_blocks),
            'ffn_mult': int(ffn_mult),
            'attn_radius': int(attn_radius),
            'attn_heads': int(attn_heads),
            'block_dropout': float(block_dropout),
            'n_phg': int(n_phg),
            'lang_group': str(lang_group),
            'n_tokens': int(n_tokens),
            'n_tones': int(n_tones),
        }

    def load_config_state_dict(self, config_dict):
        missing = [k for k in (*TRUNK_SHAPE_KEYS, *REGULARISATION_KEYS, 'n_phg', *FINE_HEAD_KEYS)
                   if k not in config_dict]
        if missing:
            raise ValueError(f"model_config is missing {missing}; not a p4mbfa config")
        self.config = dict(config_dict)

    def save_config_state_dict(self):
        return self.config

    # ------------------------------------------------------------------ build

    def make_model(self):
        cfg = self.config
        c = cfg['cnn_channels']
        d = cfg['embed_dim']
        drop = cfg['cnn_dropout']

        self.noise_level = cfg['noise_level']
        self.input_wav_length = int(cfg['input_wav_length'])

        # Downsampler: 4 * 4 * 5 = 80 samples per frame = 5 ms at 16 kHz.
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, c, kernel=15, stride=4, dropout=drop),
            ConvBlock(c, c * 2, kernel=9, stride=4, dropout=drop),
            ConvBlock(c * 2, d, kernel=7, stride=5, dropout=drop),
        )

        # Per-frame channel gate. It uses a pooled summary to rescale channels.
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d, d, 1, bias=True),
            nn.Sigmoid(),
        )

        # Stride-1 context, residual.
        self.context = nn.Sequential(*[
            ConvBlock(d, d, kernel=k, stride=1, dropout=drop, residual=True)
            for k in cfg['context_kernels']
        ])

        self.attn_radius = int(cfg['attn_radius'])
        self.blocks = nn.Sequential(*[
            FrameBlock(d, ffn_mult=int(cfg['ffn_mult']), attn_radius=self.attn_radius,
                       attn_heads=int(cfg['attn_heads']), dropout=float(cfg['block_dropout']))
            for _ in range(int(cfg['frame_blocks']))
        ])

        self.norm = nn.LayerNorm(d)

        # The fixed, language-independent head.
        self.n_phg = int(cfg['n_phg'])
        self.phg_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, self.n_phg),
        )
        self.make_fine_heads(cfg['lang_group'], cfg['n_tokens'], cfg['n_tones'])

        self.frame_stride_samples = self._frame_stride()
        self.receptive_field_samples = self._receptive_field()
        self.frames_per_window = self.update_frames_per_window(self.input_wav_length)

    def make_fine_heads(self, lang_group, n_tokens, n_tones):
        '''
        (Re)build the two replaceable heads from scratch, fresh initialisation, and record what
        they are for in `config`. Called by `make_model`, and by nothing else: a head reset on
        a warm start builds the whole model from the live yaml and discards the checkpoint's
        fine-head tensors (see `trainer.ComposedModel.on_load_checkpoint`).
        '''
        d = self.config['embed_dim']
        self.config['lang_group'] = str(lang_group)
        self.config['n_tokens'] = int(n_tokens)
        self.config['n_tones'] = int(n_tones)
        self.lang_group = str(lang_group)
        self.n_tokens = int(n_tokens)
        self.n_tones = int(n_tones)
        self.ph_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Dropout(self.config['head_dropout']),
            nn.Linear(d * 2, self.n_tokens),
        )
        self.tone_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, self.n_tones),
        )

    def _strided_convs(self):
        return [m for m in self.feature_extractor.modules() if isinstance(m, nn.Conv1d)]

    def _frame_stride(self):
        stride = 1
        for conv in self._strided_convs():
            stride *= int(conv.stride[0])
        return stride

    def _receptive_field(self):
        '''Receptive field of one output frame, in input samples.'''
        rf, jump = 1, 1
        for block in (*self.feature_extractor, *self.context):
            conv = block.conv
            rf += (int(conv.kernel_size[0]) - 1) * jump
            jump *= int(conv.stride[0])
        rf += 2 * self.attn_radius * jump * len(self.blocks)   # 0 unless attention is on
        return rf

    def update_frames_per_window(self, input_wav_length):
        '''Frames the trunk emits for a window of this length, measured by a dry run.'''
        self.input_wav_length = int(input_wav_length)
        self.config['input_wav_length'] = self.input_wav_length
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probe = torch.zeros(1, 1, self.input_wav_length,
                                device=next(self.parameters()).device)
            frames = self.feature_extractor(probe).shape[-1]
        if was_training:
            self.train()
        self.frames_per_window = int(frames)
        return self.frames_per_window

    def describe(self, sample_rate=16000):
        ms = 1000.0 / sample_rate
        return (f"MbfaCNN: {self.frames_per_window} frames/window "
                f"({self.input_wav_length} samples, {self.input_wav_length * ms:.1f} ms), "
                f"frame stride {self.frame_stride_samples} samples "
                f"({self.frame_stride_samples * ms:.3f} ms), "
                f"receptive field {self.receptive_field_samples} samples "
                f"({self.receptive_field_samples * ms:.1f} ms), "
                f"heads: phg {self.n_phg} | ph {self.n_tokens} ({self.lang_group}) | "
                f"tone {self.n_tones}, "
                f"{sum(p.numel() for p in self.parameters()) / 1e6:.2f}M params")

    # ------------------------------------------------------------------ forward

    def encode(self, x, return_tap=False):
        '''
        Args:
            x: [N, window_samples] flattened windows
            return_tap: also return the `feature_extractor` output (before the frequency gate),
                [N, frames_per_window, embed_dim], for `boundary_flux_source: tap`.
        Returns:
            [N, frames_per_window, embed_dim] (, tap)
        '''
        if self.training and self.noise_level > 0:
            x = x + torch.randn_like(x) * self.noise_level

        tap = self.feature_extractor(x.unsqueeze(1))
        h = tap * self.freq_attention(tap)
        h = self.context(h)
        emb = self.norm(self.blocks(h.transpose(1, 2)))
        return (emb, tap.transpose(1, 2)) if return_tap else emb

    def classify(self, embeddings):
        '''embeddings: [N, T, embed_dim] -> (ph logits, phg logits, tone logits)'''
        return self.ph_head(embeddings), self.phg_head(embeddings), self.tone_head(embeddings)

    def forward(self, x):
        return self.classify(self.encode(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    heads = dict(n_phg=15, lang_group='latin', n_tokens=208, n_tones=22)
    model = MbfaCNN(input_wav_length=1920, **heads)
    print(model.describe())

    assert model.frame_stride_samples == 80, model.frame_stride_samples
    assert model.frames_per_window == 24, model.frames_per_window
    assert model.receptive_field_samples == 623, model.receptive_field_samples

    x = torch.randn(7, 1920)
    logits_ph, logits_phg, logits_tone = model(x)
    print("logits:", logits_ph.shape, logits_phg.shape, logits_tone.shape)
    assert logits_ph.shape == (7, 24, 208)
    assert logits_phg.shape == (7, 24, 15)
    assert logits_tone.shape == (7, 24, 22)

    # Centring check: a click at sample 800 must move frame 800/80 = 10.
    model.eval()
    with torch.no_grad():
        base = torch.zeros(1, 1920)
        spike = base.clone()
        spike[0, 800] = 5.0
        delta = (model.encode(spike) - model.encode(base)).abs().mean(-1)[0]
    print("peak frame:", int(delta.argmax()), "(expected 10)")
    assert int(delta.argmax()) == 10, delta

    # Every parameter belongs to exactly one of trunk / fixed head / fine heads, so a trunk
    # load and a head swap between them account for the whole state_dict.
    parts = (TRUNK_PREFIXES, FIXED_HEAD_PREFIXES, FINE_HEAD_PREFIXES)
    for k in model.state_dict():
        owners = [p for p in parts if k.startswith(p)]
        assert len(owners) == 1, (k, owners)
    print("state_dict partitions into trunk / phg_head / fine heads")

    # A config round trip rebuilds identical shapes and loads strictly.
    again = MbfaCNN()
    again.load_config_state_dict(model.save_config_state_dict())
    again.make_model()
    again.load_state_dict(model.state_dict(), strict=True)
    print("model.py self-test OK")
