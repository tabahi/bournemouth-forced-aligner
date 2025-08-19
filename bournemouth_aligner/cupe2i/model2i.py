'''
Core model for Context-Free Universal Phoneme Embeddings (CUPE).
This model takes a raw audio waveform as input and outputs phoneme class probabilities and phoneme group probabilities.
The model is based on a convolutional neural network (CNN) followed by a transformer architecture.
The model expects 1D 16Khz audio input no longer than 120ms. The model is trained to predict phoneme classes and groups from the input audio. The training pipeline assumes phoneme labels are provided in a specific format (66 classes, 11 groups).

Class ContextFreePhonemeRecognizer is the main model class used for training and inference.
Class AllophoneExtractor is a wrapper around the ContextFreePhonemeRecognizer class that loads a pre-trained model and provides a simple interface for extracting phoneme embeddings and making predictions.
'''

import torch
import torch.nn as nn
import math
from . import model_utils as model_utils

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
    def __init__(self, input_wav_length=None, CNN_n_channels=None, CNN_dropout_rate=None, window_layers_dim=None, window_layers_num=None, window_layers_heads=None, window_dropout=None, noise_level=None, phoneme_classes=None, phoneme_groups=None):
        '''
        Initialize the empty model
        '''
        super().__init__()

        if input_wav_length is not None:
            self.config_model(input_wav_length, CNN_n_channels, CNN_dropout_rate, window_layers_dim, window_layers_num, window_layers_heads, window_dropout, noise_level, phoneme_classes, phoneme_groups)
            self.make_model()
    
    def config_model(self, input_wav_length, CNN_n_channels, CNN_dropout_rate, window_layers_dim, window_layers_num, window_layers_heads, window_dropout, noise_level, phoneme_classes, phoneme_groups):
        '''
        requires hp.CNN_n_channels, hp.CNN_dropout_rate, hp.window_layers_dim, hp.window_layers_num, hp.window_layers_heads, hp.window_dropout, hp.noise_level, hp.phoneme_classesm, input_wav_length
        '''
        
        self.config = {'n_channels': CNN_n_channels, 'dropout_rate': CNN_dropout_rate, 'window_layers_dim': window_layers_dim, 'window_layers_num': window_layers_num, 'window_layers_heads': window_layers_heads, 'window_dropout': window_dropout, 'noise_level': noise_level, 'phoneme_classes': phoneme_classes, 'phoneme_groups': phoneme_groups, 'input_wav_length': input_wav_length}

    def load_config_state_dict(self, config_dict):
        self.config = config_dict
        #self.make_model()

    
    def save_config_state_dict(self):
        #print("Saving model with input_wav_length:", self.input_wav_length)
        return self.config

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
            nn.Conv1d(n_channels*4, last_cnn_output_dim, kernel_size=5, stride=2, padding=2, bias=bias),
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
        
        # Temporal stream - Modified for 20ms temporal field
        self.temporal_stream = nn.Sequential(
            # Broad temporal context (reduced kernel size due to halved temporal resolution)
            nn.Conv1d(last_cnn_output_dim, last_cnn_output_dim, 
                    kernel_size=7, stride=1, padding=3, groups=8, bias=True),
            nn.BatchNorm1d(last_cnn_output_dim),
            nn.GELU(),
            # Fine detail processing (reduced kernel size due to halved temporal resolution)
            nn.Conv1d(last_cnn_output_dim, last_cnn_output_dim, 
                    kernel_size=3, stride=1, padding=1, groups=8, bias=True)
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
        #print("frames_per_window (frames per clip if disable_windowing):", self.frames_per_window.item())
        return self.frames_per_window
        
    def forward(self, x):
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
        #print("Loaded CUPE config successfully")
        
        
        state_dict = checkpoint['state_dict']
        
        # Remove potential 'model.' prefix from keys if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
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
        
        
        #print("CUPE loaded successfully")
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

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
        
    def predict(self, audio_batch, return_embeddings=False, groups_only=False):
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

            if return_embeddings:
                return logits_class, logits_group, embeddings
            else:
                return logits_class, logits_group