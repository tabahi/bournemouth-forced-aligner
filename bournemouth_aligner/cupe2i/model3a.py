'''
Inference-side copy of `modules/p2cupe/model3a.py`: the contextless CUPE base plus a
`BoundaryHead` over the STITCHED pre-transformer CNN features.

## What a model3a checkpoint is, as of session_notes.md §5

The base path (`forward`, `predict`) is byte-for-byte model2i's - that is the whole model now.
The only addition is a small head that reads the fused CNN feature map and emits one boundary
logit per frame. It is a *placement* module: it never touches the posteriors, the decoder, the
phone identities or the sequence. It only offers `core.py` a second signal to snap onsets
against, selected with `boundary_snap_source='boundary_head'`.

    'pre_transformer'   snap on 1 - cos(fused[t], fused[t+1])      spectral flux
    'boundary_head'     snap on sigmoid(boundary_head(fused))      the learned detector

so head-vs-flux is one flag on one checkpoint, with no second training run (§5.4).

## The head may be present but UNTRAINED

`bnd_head_enabled: True` only means the module exists and its weights are in the checkpoint.
Whether it learned anything is `bnd_head_lambda`, and every ue03* checkpoint released so far
carries `bnd_head_lambda: 0.0` ("boundary objective OFF", door 16). Their `out.weight` is still
the zero it was initialised to, so `detect_boundaries` returns the bias (-2.2) for every frame.
`__init__` records this as `self.boundary_head_trained`; see the comment there for why a
constant signal is worse than useless as a snap source, and `core.py` for how it is routed
around.

## The refiner is gone

`ContextualRefiner`, `classifier_ref`, `group_classifier_ref` and `refine_stitched` were
DELETED in §5.1, not disabled. Door 14 refuted the objective they carried: it was applied to the
post-transformer embeddings, which BFA no longer reads for placement, and the measurement showed
it rebuilding downstream what already existed upstream. Checkpoints from before that change
(anything with `classifier_ref.*`, e.g. ud01c) cannot be decoded by this file - see the error
raised in `__init__` for how to score them.
'''

import torch
import torch.nn as nn

from .model2i import ContextFreePhonemeRecognizer


class BoundaryHead(nn.Module):
    """
    Learned onset detector over the stitched pre-transformer CNN features.

    Byte-for-byte the same architecture as `p2cupe/model3a.py::BoundaryHead`. Keep the two in
    sync: a silent divergence here would change where onsets land without raising.
    """

    def __init__(self, in_dim, hidden=256, kernel=5):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd to preserve length, got {kernel}")
        self.conv = nn.Conv1d(in_dim, in_dim, kernel_size=kernel,
                              padding=kernel // 2, groups=in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        '''x: [B, T, in_dim] stitched fused CNN features -> [B, T] logits'''
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.act(self.norm(self.proj(h)))
        return self.out(h).squeeze(-1)


class CUPEEmbeddingsExtractor(nn.Module):
    """
    Loads a model3a checkpoint: the contextless base plus the boundary head.

    The head is applied by `core.py` to the STITCHED feature map, never per window - a
    per-window head would only ever see 10 frames and could not learn a boundary shape spanning
    a window edge, and the stitched grid is the one onsets are placed on.
    """

    def __init__(self, cupe_ckpt_path, device='cuda', verbose=0):
        super(CUPEEmbeddingsExtractor, self).__init__()
        self.device = device

        cupe_model = ContextFreePhonemeRecognizer()

        checkpoint = torch.load(cupe_ckpt_path, map_location='cpu', weights_only=False)
        if 'model_config' not in checkpoint:
            raise ValueError("Model config not found in checkpoint")
        config = dict(checkpoint['model_config'])
        cupe_model.load_config_state_dict(config)
        cupe_model.make_model()

        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        if any(k.startswith('classifier_ref.') for k in state_dict):
            raise ValueError(
                "This checkpoint carries `classifier_ref.*`: it is a PRE-§5.1 model3a, trained "
                "with the post-stitch ContextualRefiner that was deleted (session_notes.md "
                "§5.1, door 14). Its refined heads are what produced its published numbers, so "
                "decoding it with the base heads would silently report a different model. To "
                "score such a checkpoint, check out the extractor that matches it:\n"
                "    git log --oneline -- modules/aligner/bournemouth_aligner_/cupe2i/model3a.py")

        # Split the checkpoint BEFORE the substring filters below, or `'classifier.' in k` also
        # matches head keys that merely contain it.
        head_sd = {k[len('boundary_head.'):]: v for k, v in state_dict.items()
                   if k.startswith('boundary_head.')}
        base_sd = {k: v for k, v in state_dict.items() if not k.startswith('boundary_head.')}

        base_sd = {k: v for k, v in base_sd.items()
                   if ('quantizer.' not in k) and ('prediction_head.' not in k) and ('final_proj.' not in k)
                   and (('feature_extractor.' in k) or ('freq_attention.' in k) or ('temporal_stream.' in k)
                        or ('spectral_stream.' in k) or ('fusion.' in k) or ('window_processor.' in k)
                        or ('classifier.' in k) or ('group_classifier.' in k))}
        cupe_model.load_state_dict(base_sd)

        for param in cupe_model.classifier.parameters():
            param.requires_grad = False
        for param in cupe_model.group_classifier.parameters():
            param.requires_grad = False

        self.model = cupe_model.to(device)

        # --- the boundary head ---------------------------------------------------------
        if not head_sd:
            raise ValueError(
                "model3a extractor was selected but the checkpoint carries no `boundary_head.*` "
                "weights. This is a dispatch bug in core.py::load_model, not a bad checkpoint.")

        # Which tap the head was TRAINED on, carried by the checkpoint. `core.py` reads this to
        # decide what to feed `detect_boundaries`; getting it wrong does not raise.
        self.bnd_source = str(config.get('bnd_source', 'fusion'))

        # Shapes come from the CHECKPOINT's own config, never from the live defaults: old notes
        # §12.7 item 7 records a config value that never reached the model and did not raise.
        in_dim = int(config['n_channels']) * 4
        self.boundary_head = BoundaryHead(
            in_dim=in_dim,
            hidden=int(config.get('bnd_head_hidden', 256)),
            kernel=int(config.get('bnd_head_kernel', 5)),
        )
        self.boundary_head.load_state_dict(head_sd)
        for param in self.boundary_head.parameters():
            param.requires_grad = False
        self.boundary_head.to(device)

        # --- is the head actually TRAINED? ----------------------------------------------
        # `BoundaryHead.out` is zero-initialised, so an untrained head emits its bias for
        # every frame - a CONSTANT. Snapping a constant does not detect anything: the argmax
        # of a flat signal is its first index, so every onset moves by the same
        # `1 + offset - radius` frames and the result is a global shift artifact that can
        # still score above the unsnapped decoder (it absorbs the decoder's placement bias)
        # while carrying no boundary information at all.
        #
        # Every checkpoint shipped so far was trained with `bnd_head_lambda: 0.0` ("boundary
        # objective OFF", door 16), so this is the common case, not the exotic one. Detect it
        # here and let `core.py` decide: `boundary_snap_source='auto'` routes around an
        # untrained head, and an explicit request warns rather than silently returning a shift.
        out_w = self.boundary_head.out.weight
        self.boundary_head_trained = bool(out_w.abs().max().item() > 0.0)
        self.boundary_head_bias = float(self.boundary_head.out.bias.item())

        if verbose >= 2:
            print(f"CUPE model3a loaded successfully (boundary head: in_dim={in_dim}, "
                  f"kernel={self.boundary_head.conv.kernel_size[0]}, "
                  f"trained={self.boundary_head_trained}, bnd_source={self.bnd_source})")

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.boundary_head.to(device)
        return self

    def patch_model(self):
        pass

    @torch.no_grad()
    def detect_boundaries(self, stitched_cnn_features):
        '''
        One boundary logit per frame, from the STITCHED pre-transformer feature map.

        The mirror of `p2cupe/trainer.py::forward`, which is the definition of correct here:
        the inference path must reproduce the training path or every number is uninterpretable.
        The head was trained on the stitched tensor, so it is applied to the stitched tensor.

        Args:
            stitched_cnn_features: [B, T, n_channels*4]
        Returns:
            [B, T] logits (NOT sigmoid - the snapper only needs the ordering)
        '''
        if stitched_cnn_features is None:
            return None
        return self.boundary_head(stitched_cnn_features)

    def forward(self, audio_batch, layer=-1, return_pre_window=False, tap='fusion'):
        '''Contextless embeddings, per window. Identical to model2i.'''
        x = audio_batch.to(self.device)
        if self.model.training:
            x = x + torch.randn_like(x) * self.model.noise_level
        x = x.unsqueeze(1)
        features = self.model.feature_extractor(x)
        att = self.model.freq_attention(features)
        features = features * att
        temporal = self.model.temporal_stream(features)
        spectral = self.model.spectral_stream(features)
        fused = torch.cat([temporal, spectral], dim=1)
        fused = self.model.fusion(fused)
        fused = fused.transpose(1, 2)
        if return_pre_window:
            pre = spectral.transpose(1, 2) if tap == 'spectral_stream' else fused
            return self.model.window_processor(fused), pre
        return self.model.window_processor(fused)

    @torch.no_grad()
    def predict(self, audio_batch, return_embeddings=False, groups_only=False,
                return_pre_window=False, tap='fusion'):
        '''
        Per-window logits and embeddings.

        `return_pre_window` also returns a PRE-TRANSFORMER tap, which is where the boundary
        signal lives (session_notes.md §1) and what every snap source reads - the flux directly,
        the head through `detect_boundaries` after stitching. Returned from the same forward, so
        it costs nothing extra.

        `tap` selects `'fusion'` or `'spectral_stream'`. For `boundary_snap_source
        ='boundary_head'` the caller must pass `self.bnd_source`: the head was trained on one
        tap and reading it against the other does not raise, it just scores low.
        '''
        if return_pre_window:
            embeddings, pre = self.forward(audio_batch, return_pre_window=True, tap=tap)
        else:
            embeddings, pre = self.forward(audio_batch), None
        logits_group = self.model.group_classifier(embeddings)
        logits_class = None if groups_only else self.model.classifier(embeddings)
        if return_pre_window:
            return logits_class, logits_group, embeddings, pre
        if return_embeddings:
            return logits_class, logits_group, embeddings
        return logits_class, logits_group
