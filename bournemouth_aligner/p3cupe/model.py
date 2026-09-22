'''
p3cupe: context-free universal phoneme encoder, CNN only.

Takes a raw 16 kHz waveform window (120 ms) and emits one phoneme posterior per 5 ms frame.
No transformer, no boundary head, no refiner, no blank class. This is the inference copy of
the trunk; the training package it was ported from is `bournemouth_aligner/model3p/`, whose
README.md and memory.md hold the design record.

## Geometry (every timestamp in this library comes from these numbers)

    conv strides 4 * 4 * 5              = 80 samples  = 5.0 ms per frame
    120 ms window                       = 1920 samples = 24 frames
    receptive field (default context)   = 623 samples  = 38.9 ms, centred

Every conv has an ODD kernel and `padding = kernel // 2`, which is what makes frame `f` sit
centred on input sample `f * 80` with no offset correction anywhere. Keep it that way: an even
kernel anywhere in the stack shifts every timestamp this project produces by half a frame, and
nothing will raise. The self-test at the bottom of this file asserts it with a click at a
known sample.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    `CupeCNN._receptive_field` counts it.

    Off in every released checkpoint (`attn_radius: 0`); kept so a checkpoint that was trained
    with it still rebuilds.
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
    per-frame MLP. With `attn_radius=0` it is a transformer block with the attention taken
    out - depth and width, and NO time mixing, so the receptive field is untouched.
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
        # Zero-initialised residual output, so an untrained block is the identity. Kept
        # identical to the training copy in `model3p/model.py` so the two cannot drift.
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        '''x: [N, T, dim]'''
        if self.attn is not None:
            x = x + self.drop(self.attn(self.attn_norm(x)))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class CupeCNN(nn.Module):
    '''
    The trunk. `config` is the single source of truth for its shape and is what the checkpoint
    carries, so a reload rebuilds the same tensor sizes without consulting any yaml.

    Args (all also readable from `config`):
        input_wav_length: samples per window. Sets `frames_per_window`.
        cnn_channels: width of the first conv. Layers go C -> 2C -> embed_dim.
        embed_dim: per-frame embedding width, and the width of the context convs.
        context_kernels: stride-1 kernels applied after the downsampler. This list IS the
            receptive-field knob: each entry adds (k-1) * 80 samples = (k-1) * 5 ms.
        phoneme_classes / phoneme_groups: 66 / 16 real labels. The heads emit one more of
            each because index 66 ('noise') and 16 ('noise') exist in the shared ph66 mapper
            and downstream tooling indexes by it. Nothing ever makes them a target: there is
            no blank class, and the slot simply stays unused.
        frame_blocks: number of `FrameBlock`s after the context convs - the capacity knob,
            independent of the receptive field while `attn_radius` is 0.
        ffn_mult: hidden width of each block's MLP, as a multiple of embed_dim.
        attn_radius: 0 = no attention (the released setting). >0 adds banded self-attention of
            +-radius frames to every block, and 2 * radius * 5 ms of receptive field per block.
    '''

    def __init__(self, input_wav_length=None, cnn_channels=192, embed_dim=512,
                 context_kernels=(5, 3), cnn_dropout=0.1, head_dropout=0.1,
                 noise_level=0.01, phoneme_classes=66, phoneme_groups=16,
                 frame_blocks=4, ffn_mult=4, attn_radius=0, attn_heads=8, block_dropout=0.1):
        super().__init__()
        self.config = None
        if input_wav_length is not None:
            self.config_model(input_wav_length, cnn_channels, embed_dim, context_kernels,
                              cnn_dropout, head_dropout, noise_level,
                              phoneme_classes, phoneme_groups, frame_blocks, ffn_mult,
                              attn_radius, attn_heads, block_dropout)
            self.make_model()

    # ------------------------------------------------------------------ config

    def config_model(self, input_wav_length, cnn_channels, embed_dim, context_kernels,
                     cnn_dropout, head_dropout, noise_level, phoneme_classes, phoneme_groups,
                     frame_blocks=0, ffn_mult=4, attn_radius=0, attn_heads=8,
                     block_dropout=0.1):
        self.config = {
            'input_wav_length': int(input_wav_length),
            'cnn_channels': int(cnn_channels),
            'embed_dim': int(embed_dim),
            'context_kernels': [int(k) for k in context_kernels],
            'cnn_dropout': float(cnn_dropout),
            'head_dropout': float(head_dropout),
            'noise_level': float(noise_level),
            'phoneme_classes': int(phoneme_classes),
            'phoneme_groups': int(phoneme_groups),
            'frame_blocks': int(frame_blocks),
            'ffn_mult': int(ffn_mult),
            'attn_radius': int(attn_radius),
            'attn_heads': int(attn_heads),
            'block_dropout': float(block_dropout),
        }

    def load_config_state_dict(self, config_dict):
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
        self.classes_dim = cfg['phoneme_classes'] + 1      # +1: unused 'noise' slot
        self.groups_dim = cfg['phoneme_groups'] + 1

        # Downsampler: 4 * 4 * 5 = 80 samples per frame = 5 ms at 16 kHz.
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, c, kernel=15, stride=4, dropout=drop),
            ConvBlock(c, c * 2, kernel=9, stride=4, dropout=drop),
            ConvBlock(c * 2, d, kernel=7, stride=5, dropout=drop),
        )

        # Per-frame channel gate. Cheap, and it cannot mix across time.
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d, d, 1, bias=True),
            nn.Sigmoid(),
        )

        # Stride-1 context. Residual, so the trunk can route around it.
        self.context = nn.Sequential(*[
            ConvBlock(d, d, kernel=k, stride=1, dropout=drop, residual=True)
            for k in cfg['context_kernels']
        ])

        # Per-frame depth. `.get` with a default so a checkpoint saved before these keys
        # existed rebuilds as exactly the network it was.
        self.attn_radius = int(cfg.get('attn_radius', 0))
        self.blocks = nn.Sequential(*[
            FrameBlock(d, ffn_mult=int(cfg.get('ffn_mult', 4)), attn_radius=self.attn_radius,
                       attn_heads=int(cfg.get('attn_heads', 8)),
                       dropout=float(cfg.get('block_dropout', 0.1)))
            for _ in range(int(cfg.get('frame_blocks', 0)))
        ])

        self.norm = nn.LayerNorm(d)

        self.classifier = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(cfg['head_dropout']),
            nn.Linear(d * 2, self.classes_dim),
        )
        self.group_classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, self.groups_dim),
        )

        self.frame_stride_samples = self._frame_stride()
        self.receptive_field_samples = self._receptive_field()
        self.frames_per_window = self.update_frames_per_window(self.input_wav_length)

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
        '''
        Frames the trunk emits for a window of this length. Measured by a dry run rather than
        by a formula: the formula and the conv stack drifted apart once in this lineage and the
        mismatch was invisible for a whole training run.
        '''
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
        return (f"CupeCNN: {self.frames_per_window} frames/window "
                f"({self.input_wav_length} samples, {self.input_wav_length * ms:.1f} ms), "
                f"frame stride {self.frame_stride_samples} samples "
                f"({self.frame_stride_samples * ms:.3f} ms), "
                f"receptive field {self.receptive_field_samples} samples "
                f"({self.receptive_field_samples * ms:.1f} ms), "
                f"{sum(p.numel() for p in self.parameters()) / 1e6:.2f}M params")

    # ------------------------------------------------------------------ forward

    def encode(self, x, return_tap=False):
        '''
        Args:
            x: [N, window_samples] flattened windows
            return_tap: also return the `feature_extractor` output (before the frequency gate),
                [N, frames_per_window, embed_dim]. Kept because a checkpoint trained with
                `boundary_flux_source: tap` reads its onset bonus off this tensor; the released
                checkpoints use log-mel flux instead, which has no parameters and cannot drift.
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
        return self.classifier(embeddings), self.group_classifier(embeddings)

    def forward(self, x):
        return self.classify(self.encode(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CupeCNN(input_wav_length=1920)
    print(model.describe())

    assert model.frame_stride_samples == 80, model.frame_stride_samples
    assert model.frames_per_window == 24, model.frames_per_window
    assert model.receptive_field_samples == 623, model.receptive_field_samples

    x = torch.randn(7, 1920)
    logits_c, logits_g = model(x)
    assert logits_c.shape == (7, 24, 67), logits_c.shape
    assert logits_g.shape == (7, 24, 17), logits_g.shape

    # Centring check: a click at sample 800 must move frame 800/80 = 10 and leave distant
    # frames alone. This is the assertion that catches an even kernel or a wrong padding.
    model.eval()
    with torch.no_grad():
        base = torch.zeros(1, 1920)
        emb0 = model.encode(base)
        spike = base.clone()
        spike[0, 800] = 5.0
        emb1 = model.encode(spike)
    delta = (emb1 - emb0).abs().mean(-1)[0]
    assert int(delta.argmax()) == 10, delta

    # The per-frame blocks must not mix time: frames beyond the conv receptive field stay put
    # even after the residual branches are given real weights. Not exactly zero -
    # `freq_attention` pools over the whole window, so every frame sees ~1e-4 of the click
    # through the gate, which is 3+ orders below the peak.
    for p in model.blocks.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        delta = (model.encode(spike) - model.encode(base)).abs().mean(-1)[0]
    reach = (623 // 2) // 80 + 1        # frames the conv stack can see either side of 800
    assert int(delta.argmax()) == 10 and \
        float(delta[:10 - reach].max()) < 1e-3 * float(delta.max()), delta
    print("model.py self-test OK")
