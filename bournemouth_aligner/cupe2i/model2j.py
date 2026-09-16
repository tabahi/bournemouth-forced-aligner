''' model2j date: 2026-08-2
Core model for Context-Free Universal Phoneme Embeddings (CUPE).
This model takes a raw audio waveform as input and outputs phoneme class probabilities and phoneme group probabilities.
The model is based on a convolutional neural network (CNN) followed by a transformer architecture.
The model expects 1D 16Khz audio input no longer than 120ms. The model is trained to predict phoneme classes and groups from the input audio. The training pipeline assumes phoneme labels are provided in a specific format (66 classes, 11 groups).

'''

import torch
import torch.nn as nn
import math
try:
    from . import model_utils as model_utils
except ImportError:
    # not installed as a package, try relative import
    import sys
    sys.path.append('./bournemouth_aligner/cupe2i')
    import model_utils
# big change over model2i: adding a ContextualRefiner class that takes the embeddings from the base model and refines them using a transformer architecture. This allows the model to capture longer-range dependencies and improve phoneme classification accuracy.
# big change over model2h: adding a grouped classification task for phoneme groups. IT's not a multi-task learning, but a single task with multiple outputs


class WindowwiseTransformer(nn.Module):
    """Process entire window using transformer architecture with sinusoidal position encodings.
    Allows cross-frame attention within the window while maintaining position-awareness."""
    
    def __init__(self, input_dim, context_dim, frames_per_window, num_context_layers=4, context_dropout=0.1, num_transformer_heads=8):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, context_dim)
        
        # Generate fixed sinusoidal position encodings
        self.register_buffer(
            "pos_encoding",
            self._create_sinusoidal_encoding(frames_per_window, context_dim)
        )
        
        self.dropout = nn.Dropout(context_dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=context_dim,
                nhead=num_transformer_heads,
                dropout=context_dropout,
                batch_first=True,
                dim_feedforward=context_dim * 4,
                norm_first=True  # Pre-norm architecture for better stability
            ) for _ in range(num_context_layers)
        ])
        self.norm = nn.LayerNorm(context_dim)
        
        # Initialize a scale factor for position encodings
        self.pos_encoding_scale = nn.Parameter(torch.ones(1))
        
    def _create_sinusoidal_encoding(self, max_len, d_model):
        """Create sinusoidal position encodings.
        
        Args:
            max_len: Maximum sequence length (frames_per_window)
            d_model: Embedding dimension (final_projection_dim)
            
        Returns:
            pos_encoding: Positional encoding matrix of shape (1, max_len, d_model)
        """
        pe = torch.zeros(int(max_len), d_model)
        position = torch.arange(0, int(max_len), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and normalize
        pe = pe.unsqueeze(0)
        
        return pe
    
    def _get_pos_encoding_subset(self, seq_len):
        """Get position encodings for the actual sequence length."""
        return self.pos_encoding[:, :seq_len, :]
    
    def forward(self, x):
        """
        Forward pass with scaled positional encodings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, last_cnn_output_dim)
            
        Returns:
            output: Processed tensor of shape (batch_size, seq_len, final_projection_dim)
        """
        x = self.input_projection(x)
        
        # Get positional encodings for the actual sequence length
        pos_enc = self._get_pos_encoding_subset(x.size(1))
        
        # Add scaled positional encodings
        x = x + (self.pos_encoding_scale * pos_enc)
        
        # Apply dropout after position encoding
        x = self.dropout(x)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)
    
    def reset_parameters(self):
        """Reset learnable parameters while keeping position encodings fixed."""
        nn.init.normal_(self.pos_encoding_scale, mean=1.0, std=0.1)
        
        # Reset input projection
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        
        # Reset transformer layers
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)


class ContextFreePhonemeRecognizer(nn.Module):
    def __init__(self, input_wav_length=None, CNN_n_channels=None, CNN_dropout_rate=None, window_layers_dim=None, window_layers_num=None, window_layers_heads=None, window_dropout=None, noise_level=None, phoneme_classes=None, phoneme_groups=None, cnn_last_stride=2):
        '''
        Initialize the empty model
        '''
        super().__init__()

        if input_wav_length is not None:
            self.config_model(input_wav_length, CNN_n_channels, CNN_dropout_rate, window_layers_dim, window_layers_num, window_layers_heads, window_dropout, noise_level, phoneme_classes, phoneme_groups, cnn_last_stride)
            self.make_model()
    
    def config_model(self, input_wav_length, CNN_n_channels, CNN_dropout_rate, window_layers_dim, window_layers_num, window_layers_heads, window_dropout, noise_level, phoneme_classes, phoneme_groups, cnn_last_stride=2):
        '''
        requires hp.CNN_n_channels, hp.CNN_dropout_rate, hp.window_layers_dim, hp.window_layers_num, hp.window_layers_heads, hp.window_dropout, hp.noise_level, hp.phoneme_classesm, input_wav_length
        '''
        
        self.config = {'n_channels': CNN_n_channels, 'dropout_rate': CNN_dropout_rate, 'window_layers_dim': window_layers_dim, 'window_layers_num': window_layers_num, 'window_layers_heads': window_layers_heads, 'window_dropout': window_dropout, 'noise_level': noise_level, 'phoneme_classes': phoneme_classes, 'phoneme_groups': phoneme_groups, 'input_wav_length': input_wav_length, 'cnn_last_stride': cnn_last_stride}

    def load_config_state_dict(self, config_dict):
        self.config = config_dict
        #self.make_model()

    
    def save_config_state_dict(self):
        #print("Saving model with input_wav_length:", self.input_wav_length)
        return self.config

    def frame_stride_samples(self):
        '''
        Samples advanced per output frame: the product of the feature extractor's conv strides.

        This is the *true* frame spacing and the only correct basis for a frames->ms conversion.
        It is NOT `input_wav_length / frames_per_window`, which is the average frame density
        including the padding-compressed edge frames (§7.1), and it is not the clip duration
        divided by the frame count, which drifts (+182 ms over 10 s, §8.3).
        '''
        stride = 1
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.Conv1d):
                stride *= int(module.stride[0])
        return stride

    def receptive_field_ms(self, sample_rate=16000):
        '''
        Time-mixing receptive field in ms, over feature_extractor + temporal_stream.

        Recorded because halving `cnn_last_stride` halves it as a side effect: `temporal_stream`
        mixes over *frames*, so at twice the frame rate its kernels span half the time
        (149.7 -> ~75 ms). Door 3 drove the RF to exact MFA parity (44.7 ms) and it bought
        nothing, so this is expected to be inert - but it is a confound and belongs on the record.
        '''
        rf, stride = 1, 1
        for module in list(self.feature_extractor.modules()) + list(self.temporal_stream.modules()):
            if isinstance(module, nn.Conv1d):
                rf += (int(module.kernel_size[0]) - 1) * stride
                stride *= int(module.stride[0])
        return 1000.0 * rf / sample_rate

    def make_model(self):
       
        
        # configurable dims:
        bias = False
        n_channels = self.config['n_channels']
        cnn_dropout_rate = self.config['dropout_rate']
        
        window_layers_dim = self.config['window_layers_dim']
        window_layers_num = self.config['window_layers_num']
        window_layers_heads = self.config['window_layers_heads']
        window_dropout = self.config['window_dropout']

        phoneme_classes = self.config['phoneme_classes']
        phoneme_groups = self.config['phoneme_groups']
        noise_level = self.config['noise_level']
        # `.get` so checkpoints predating this key still load at the original
        # 210-sample geometry. 1 -> 105 samples, required for a 3-state topology.
        cnn_last_stride = int(self.config.get('cnn_last_stride', 2))
        
        # calculated dims
        last_cnn_output_dim = n_channels*4
        self.classes_dim = phoneme_classes + 1 # +1 for blank token
        self.groups_dim = phoneme_groups + 1 # +1 for blank token
        
        self.noise_level = noise_level
        self.input_wav_length = int(self.config['input_wav_length'])
        # Sanity checks
        
        assert(self.input_wav_length > (0.005*16000))
        assert(window_layers_dim <= last_cnn_output_dim)
        assert(self.classes_dim > 1)
        assert(self.classes_dim <= window_layers_dim)
        assert(self.groups_dim < self.classes_dim)

        # Feature Extractor - Fine-tuned for 8-10 frames per window, 20ms temporal resolution
        self.feature_extractor = nn.Sequential(
            # Layer 1: Increased stride from 6 to 7
            nn.Conv1d(1, n_channels, kernel_size=15, stride=7, padding=7, bias=bias),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
            nn.Dropout(cnn_dropout_rate),
            
            # Layer 2: Increased stride from 4 to 5
            nn.Conv1d(n_channels, n_channels*2, kernel_size=11, stride=5, padding=5, bias=bias),
            nn.BatchNorm1d(n_channels*2),
            nn.GELU(),
            nn.Dropout(cnn_dropout_rate),
            
            # Layer 3: Keep stride at 3
            nn.Conv1d(n_channels*2, n_channels*4, kernel_size=7, stride=3, padding=3, bias=bias),
            nn.BatchNorm1d(n_channels*4),
            nn.GELU(),
            nn.Dropout(cnn_dropout_rate),
            
            # Layer 4: Keep stride at 2
            nn.Conv1d(n_channels*4, last_cnn_output_dim, kernel_size=5, stride=cnn_last_stride, padding=2, bias=bias),
            nn.BatchNorm1d(last_cnn_output_dim),
            nn.GELU(),
            nn.Dropout(cnn_dropout_rate)
        )


        # Frequency attention mechanism (unchanged)
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(last_cnn_output_dim, last_cnn_output_dim, 1, bias=True),
            nn.Sigmoid()
        )
        
        # Temporal stream - kernel sizes come from the checkpoint's config so that both
        # pre-patch checkpoints (kernel 7/3) and receptive-field-narrowed ones (kernel 1/1,
        # see patch_model's docstring) load without a shape mismatch.
        temporal_k0, temporal_k1 = self.config.get('temporal_kernels', (7, 3))
        self.temporal_stream = nn.Sequential(
            # Broad temporal context (reduced kernel size due to halved temporal resolution)
            nn.Conv1d(last_cnn_output_dim, last_cnn_output_dim,
                    kernel_size=temporal_k0, stride=1, padding=temporal_k0 // 2, groups=8, bias=True),
            nn.BatchNorm1d(last_cnn_output_dim),
            nn.GELU(),
            # Fine detail processing (reduced kernel size due to halved temporal resolution)
            nn.Conv1d(last_cnn_output_dim, last_cnn_output_dim,
                    kernel_size=temporal_k1, stride=1, padding=temporal_k1 // 2, groups=8, bias=True)
        )
        
        # Spectral stream (unchanged)
        self.spectral_stream = nn.Sequential(
            nn.Conv1d(last_cnn_output_dim, n_channels*12, 
                    kernel_size=1, stride=1, padding=0, groups=8, bias=True),
            nn.BatchNorm1d(n_channels*12),
            nn.GELU(),
            nn.Conv1d(n_channels*12, last_cnn_output_dim,
                    kernel_size=1, stride=1, padding=0, groups=8, bias=True),
            nn.BatchNorm1d(last_cnn_output_dim),
            nn.GELU()
        )
        
        # Feature fusion (unchanged)
        self.fusion = nn.Sequential(
            nn.Conv1d(last_cnn_output_dim*2, last_cnn_output_dim, 1, bias=True),
            nn.BatchNorm1d(last_cnn_output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        assert(last_cnn_output_dim == window_layers_dim*2)
        #self.layer_dims = self.make_layer_sizer()
        self.layer_dims = model_utils.ModelUtils.extract_layer_dims(self)
        self.frames_per_window = model_utils.ModelUtils.calculate_layer_sizes(self.layer_dims, torch.tensor([self.input_wav_length]), -1)[0].int()
        self.model_utils = model_utils.ModelUtils(self.layer_dims, self.input_wav_length, self.frames_per_window)
        
        

        # Window processor (unchanged)
        self.window_processor = WindowwiseTransformer(
            input_dim=last_cnn_output_dim, 
            context_dim=window_layers_dim, 
            frames_per_window=self.frames_per_window, 
            num_context_layers=window_layers_num, 
            context_dropout=window_dropout, 
            num_transformer_heads=window_layers_heads
        )

        # Final classifier (added Relu and dropout)
        self.classifier = nn.Sequential(
            nn.Linear(window_layers_dim, window_layers_dim*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(window_layers_dim*4, self.classes_dim)
        )

        self.group_classifier = nn.Sequential(
            nn.Linear(window_layers_dim, window_layers_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(window_layers_dim // 2, self.groups_dim),
        )

    
    def update_frames_per_window(self, input_wav_length):
        self.input_wav_length = int(input_wav_length)
        self.config['input_wav_length'] = self.input_wav_length
        self.frames_per_window = self.model_utils.calculate_layer_sizes(self.layer_dims, torch.tensor([self.input_wav_length]), -1)[0].int()
        self.frames_per_window = torch.ceil((self.frames_per_window)).int()
        print("frames_per_window (frames per clip if disable_windowing):", self.frames_per_window.item())
        return self.frames_per_window
        
    def patch_model(self, temporal_kernels=(1, 1)):
        '''
        Applied AFTER an old model2i/model2j checkpoint is loaded. Narrows the **time-mixing
        receptive field** by shrinking the convolutions in `temporal_stream`.

        ## Why

        Receptive fields measured empirically (by gradient) on the time-mixing path:

            feature_extractor alone             44.7 ms   <- already MFA-like
            + spectral_stream (1x1 convs)       44.7 ms   <- pointwise, adds nothing
            + temporal_stream [k=7, k=3]       149.7 ms   <- the whole excess is here
            MFA acoustic model (25ms + deltas)  ~45 ms

        (`freq_attention` contains `AdaptiveAvgPool1d(1)`, so *any-dependence* RF is global
        by construction. That is a per-channel scalar gate - it does not mix time - so the
        number that governs boundary smearing is the time-mixing path above.)

        By kernel pair, measured:

            [7, 3] = 149.7 ms  (3.3x MFA)   [3, 3] = 97.2 ms  (2.2x)
            [1, 3] =  70.9 ms  (1.6x)       [1, 1] = 44.7 ms  (1.0x, MFA parity)

        These convs are a **mandatory** low-pass applied to every frame. That is the key
        difference from the window transformer, whose attention is *optional* context the
        model can learn to keep local. A k=7 conv at a 210-sample stride smears every output
        frame across ~6 neighbours - larger than the 1.7-frame boundary error being removed.

        Default `(1, 1)` targets MFA parity: the only configuration known to reach ~12.6ms.
        Local temporal structure is still available two ways - `feature_extractor`'s own
        44.7ms window, and the window transformer's learned attention - so what is removed
        is forced smoothing, not the ability to see neighbours.

        Warm-started from the CENTRE TAPS of each existing kernel, so a patched layer begins
        as a truncation of what it already computed rather than from noise.

        Args:
            temporal_kernels: new kernel size per Conv1d in `temporal_stream`, in order.
                An int applies to the first conv only. `None` (or a value >= the current
                size) leaves that conv alone. `0`/`None` for the whole argument disables.
        Returns:
            True if anything was modified.
        '''
        '''
        if temporal_kernels is None or temporal_kernels == 0:
            print("patch_model: disabled, temporal_stream unchanged")
            return False
        if isinstance(temporal_kernels, int):
            temporal_kernels = (temporal_kernels,)

        conv_idx = [i for i, m in enumerate(self.temporal_stream) if isinstance(m, nn.Conv1d)]
        changed = False
        report = []
        for slot, new_k in zip(conv_idx, temporal_kernels):
            conv = self.temporal_stream[slot]
            old_k = conv.kernel_size[0]
            if new_k is None or new_k <= 0 or new_k >= old_k:
                report.append(f"[{slot}] k={old_k} (unchanged)")
                continue
            assert new_k % 2 == 1, "kernel size must be odd to keep frames aligned"

            new_conv = nn.Conv1d(conv.in_channels, conv.out_channels,
                                 kernel_size=new_k, stride=conv.stride[0],
                                 padding=new_k // 2, groups=conv.groups,
                                 bias=conv.bias is not None)
            with torch.no_grad():
                lo = (old_k - new_k) // 2
                new_conv.weight.copy_(conv.weight[:, :, lo:lo + new_k])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)
            self.temporal_stream[slot] = new_conv.to(conv.weight.device)
            report.append(f"[{slot}] k={old_k} -> {new_k}")
            changed = True

        self.config['temporal_kernels'] = [self.temporal_stream[i].kernel_size[0] for i in conv_idx]
        print("patch_model: temporal_stream " + ", ".join(report) +
              (" (warm-started from centre taps)" if changed else ""))
        return changed
        '''
        return False  # patching is disabled for now, as the model is already updated to the new version

    def forward(self, x, return_embeddings=False):
        if self.training:
            x = x + torch.randn_like(x) * self.noise_level

        x = x.unsqueeze(1)  # Add channel dim

        # Feature extraction (B, 1, T) -> (B, 8n, T')
        features = self.feature_extractor(x)

        # Attention
        att = self.freq_attention(features)
        features = features * att

        # Dual stream processing
        temporal = self.temporal_stream(features)
        spectral = self.spectral_stream(features)

        # Combine streams and fuse
        fused = torch.cat([temporal, spectral], dim=1)
        fused = self.fusion(fused)

        # Prepare for transformer
        fused = fused.transpose(1, 2)  # (B, T', 8n)

        # Apply window processor
        features = self.window_processor(fused)

        # Classify each frame
        logits_class = self.classifier(features)
        logits_group = self.group_classifier(features)

        if return_embeddings:
            # window_processor output: the richest per-frame representation, before the
            # classifier bottleneck discards everything that is not phone identity.
            # This is what the ContextualRefiner consumes.
            return logits_class, logits_group, features
        return logits_class, logits_group
    

class ContextualRefiner(nn.Module):
    """
    Second-stage context model over the **stitched embeddings** of the context-free base.

    ## Why this exists

    Three rounds of training-objective work (see `plan-for-improvements.md` §6) closed the
    blank ambiguity window completely - `blank_rate` 0.70 -> 0.17, `path_blank_frac` 0.72 ->
    0.08, `direct_transition_frac` 0.98, realistic segment durations - and TIMIT `f1@25ms`
    did not move (49.93 -> 49.30). That rules the objective out: the boundary uncertainty
    is in the acoustic model, not in how the loss expresses it.

    What the base provably lacks is **cross-window context**. Each base frame sees only its
    own 120ms window, so the model has no way to represent a phone duration prior, no view
    of the neighbouring phones, and nothing analogous to MFA's learned HMM transition
    probabilities. That is the one structural advantage MFA has that no amount of
    reweighting inside a single window can supply.

    ## Why embeddings and not logits

    `stich_window_predictions` is dimension-agnostic, so stitching the 512-d
    `window_processor` output costs exactly the same as stitching the 67-d logits. The
    logits have been through the classifier bottleneck, which keeps only what is needed to
    name the phone and discards sub-frame acoustic detail - which is precisely where a
    boundary lives. Feeding the refiner logits would hand it the answer to the wrong
    question.

    Note the stitch is itself lossy: with 10 frames per window and 50% overlap, the cosine
    weights make frames at local index 2 and 7 an exact 50/50 blend of two views 80ms
    apart. The refiner sees the blended signal; it cannot undo that, which is why the
    receptive-field patch (`patch_model`) is applied in the same round.

    ## Shape

    in:  [B, T, base_dim]   stitched embeddings, T = whole clip (up to ~620 frames)
    out: [B, T, classes], [B, T, groups]

    Bidirectional - boundary evidence is on both sides. Padding beyond `lengths` is masked
    out of attention so pad frames cannot leak into real ones.
    """

    def __init__(self, base_dim, classes_dim, groups_dim, dim=512, layers=4, heads=8,
                 dropout=0.1, max_len=4096, residual_logits=True):
        super().__init__()
        self.residual_logits = residual_logits
        self.input_projection = nn.Linear(base_dim, dim)
        self.register_buffer("pos_encoding", self._sinusoidal(max_len, dim), persistent=False)
        self.pos_scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dropout=dropout, batch_first=True,
                dim_feedforward=dim * 4, norm_first=True,
            ) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, classes_dim)
        self.group_classifier = nn.Linear(dim, groups_dim)

        # Start as a no-op: with zeroed heads and residual_logits=True the refiner's first
        # forward reproduces the base model exactly, so resuming from a base checkpoint
        # does not destroy it before the refiner has learned anything.
        nn.init.zeros_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)
        nn.init.zeros_(self.group_classifier.weight); nn.init.zeros_(self.group_classifier.bias)

    @staticmethod
    def _sinusoidal(max_len, d_model):
        pe = torch.zeros(int(max_len), d_model)
        position = torch.arange(0, int(max_len), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, embeddings, lengths=None, base_logits_class=None, base_logits_group=None):
        """
        Args:
            embeddings: [B, T, base_dim] stitched base embeddings
            lengths: [B] valid frames per clip; frames past this are masked out of attention
            base_logits_class/group: base outputs, added residually when
                `residual_logits` so the refiner only has to learn the correction.
        """
        T = embeddings.shape[1]
        if T > self.pos_encoding.shape[1]:
            # max_len was sized for training-length clips; inference segments (BFA's
            # duration_max) can run longer, and this buffer is non-persistent (not part of
            # the checkpoint - only pos_scale is learned) so it's safe to regrow here.
            self.pos_encoding = self._sinusoidal(T, self.pos_encoding.shape[2]).to(embeddings.device)
        x = self.input_projection(embeddings)
        x = x + self.pos_scale * self.pos_encoding[:, :T, :].to(x.dtype)
        x = self.dropout(x)

        pad_mask = None
        if lengths is not None:
            idx = torch.arange(T, device=embeddings.device).unsqueeze(0)
            pad_mask = idx >= lengths.to(embeddings.device).unsqueeze(1)   # True = ignore
            # A fully-masked row makes softmax produce NaN; keep frame 0 alive.
            pad_mask[:, 0] = False

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        logits_class = self.classifier(x)
        logits_group = self.group_classifier(x)
        if self.residual_logits:
            if base_logits_class is not None:
                logits_class = logits_class + base_logits_class
            if base_logits_group is not None:
                logits_group = logits_group + base_logits_group
        return logits_class, logits_group


class CUPEEmbeddingsExtractor(nn.Module):
    def __init__(self, cupe_ckpt_path, device='cuda'):
        super(CUPEEmbeddingsExtractor, self).__init__()  # Call nn.Module's init
        self.device = device

        cupe_model = ContextFreePhonemeRecognizer()

        #from argparse import Namespace
        checkpoint = torch.load(cupe_ckpt_path, weights_only=False)
        if 'model_config' not in checkpoint: raise ValueError("Model config not found in checkpoint")
        cupe_model.load_config_state_dict(checkpoint['model_config'])
        cupe_model.make_model()
        print("Loaded CUPE config successfully")
        
        
        state_dict = checkpoint['state_dict']

        # Remove potential 'model.' prefix from keys if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # The refiner is a separate top-level module in the checkpoint (not nested inside
        # the base model), so pull its keys out before filtering the base state_dict below -
        # otherwise 'classifier.' in k also matches 'refiner.classifier.weight'.
        refiner_state_dict = {k[len('refiner.'):]: v for k, v in state_dict.items() if k.startswith('refiner.')}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('refiner.')}

        # remove quantization keys
        state_dict = {k: v for k, v in state_dict.items() if ('quantizer.' not in k) and ('prediction_head.' not in k) and ('final_proj.' not in k) and (('feature_extractor.' in k) or ('freq_attention.' in k)  or ('temporal_stream.' in k) or ('spectral_stream.' in k) or ('fusion.' in k) or ('window_processor.' in k) or ('classifier.' in k) or ('group_classifier.' in k) ) }
        cupe_model.load_state_dict(state_dict)

        # disable grad for classifier and group_classifier
        for param in cupe_model.classifier.parameters():
            param.requires_grad = False
        for param in cupe_model.group_classifier.parameters():
            param.requires_grad = False

        self.model = cupe_model.to(device)
        #self.model.eval()  # Set to evaluation mode

        # Second-stage ContextualRefiner over stitched embeddings - only present in
        # checkpoints trained with refiner_enabled (see model2j's ContextualRefiner docstring).
        self.refiner = None
        if refiner_state_dict:
            hp = checkpoint.get('hyper_parameters', {})
            refiner = ContextualRefiner(
                base_dim=cupe_model.config['window_layers_dim'],
                classes_dim=cupe_model.classes_dim,
                groups_dim=cupe_model.groups_dim,
                dim=hp.get('refiner_dim', 512),
                layers=hp.get('refiner_layers', 4),
                heads=hp.get('refiner_heads', 8),
                dropout=hp.get('refiner_dropout', 0.1),
            )
            refiner.load_state_dict(refiner_state_dict)
            for param in refiner.parameters():
                param.requires_grad = False
            self.refiner = refiner.to(device)
            print("Loaded ContextualRefiner successfully")

        print("CUPE loaded successfully")
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        if self.refiner is not None:
            self.refiner.to(device)
        return self

    def patch_model(self, ):
        # patch the model updates (remove blank), after loading old checkpoints from model2i
        # this code will be commented out after the model is updated to the new version
        #print("Patching model to remove blank token from classifier and group_classifier")
        pass

    def forward(self, audio_batch, layer = -1):
        '''
        audio_batch: a tensor of shape (batch_size, wav_length)
        
        '''
        # Forward pass up to the window processor output
        x = audio_batch.to(self.device)
        
        if self.model.training:
            x = x + torch.randn_like(x) * self.model.noise_level
        
        x = x.unsqueeze(1)  # Add channel dim
        
        # Feature extraction (B, 1, T) -> (B, 8n, T')
        features = self.model.feature_extractor(x)
        
        # Attention
        att = self.model.freq_attention(features)
        features = features * att
        
        # Dual stream processing
        temporal = self.model.temporal_stream(features)
        spectral = self.model.spectral_stream(features)
        
        # Combine streams and fuse
        fused = torch.cat([temporal, spectral], dim=1)
        fused = self.model.fusion(fused)
        
        # Prepare for transformer
        fused = fused.transpose(1, 2)  # (B, T', 8n)

        # Apply window processor - this is our rich embedding space
        embeddings = self.model.window_processor(fused)
        
        return embeddings
        
    def predict(self, audio_batch, return_embeddings=False, groups_only=False, return_pre_window=False):
        '''
        audio_batch: a tensor of shape (batch_size, wav_length)
        return_embeddings: if True, returns the embeddings as well as the logits
        groups_only: if True, only returns the group logits
        
        Return sahpe: (batch_size, phoneme_groups) ...or... (batch_size, phoneme_classes), (batch_size, phoneme_groups)
        or if return_embeddings is True: (batch_size, T', 8n), (batch_size, phoneme_groups) ...or... (batch_size, T', 8n), (batch_size, phoneme_classes), (batch_size, phoneme_groups)
        '''
        with torch.no_grad():
            # Forward pass up to the window processor output
            x = audio_batch.to(self.device)
            
            if self.model.training:
                x = x + torch.randn_like(x) * self.model.noise_level
            
            x = x.unsqueeze(1)  # Add channel dim
            
            # Feature extraction (B, 1, T) -> (B, 8n, T')
            features = self.model.feature_extractor(x)
            
            # Attention
            att = self.model.freq_attention(features)
            features = features * att
            
            # Dual stream processing
            temporal = self.model.temporal_stream(features)
            spectral = self.model.spectral_stream(features)
            
            # Combine streams and fuse
            fused = torch.cat([temporal, spectral], dim=1)
            fused = self.model.fusion(fused)
            
            # Prepare for transformer
            fused = fused.transpose(1, 2)  # (B, T', 8n)

            # Apply window processor - this is our rich embedding space
            embeddings = self.model.window_processor(fused)
            
            logits_group = self.model.group_classifier(embeddings)

            if (not groups_only):
                logits_class = self.model.classifier(embeddings)
            else: logits_class = None

            # `fused` is the PRE-TRANSFORMER feature map. session_notes.md §11/§12: the
            # boundary signal peaks here (f1@25 ~64, abs_err ~17 ms) and is roughly halved by
            # `window_processor`, so onset snapping reads this rather than `embeddings`.
            if return_pre_window:
                return logits_class, logits_group, embeddings, fused
            if return_embeddings:
                return logits_class, logits_group, embeddings
            else:
                return logits_class, logits_group