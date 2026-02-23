"""
PyTorch to ONNX Conversion Script for CUPE Models
Bournemouth Forced Aligner - Model Export Utility

Exports the CUPE predict() pipeline (raw audio -> phoneme logits + group logits + embeddings)
to ONNX format for C++ inference.

Input:  raw audio windows, shape [batch_size, wav_length]  (e.g. [B, 1920] for 120ms @ 16kHz)
Output: phoneme_logits [B, T, 67], group_logits [B, T, 17], embeddings [B, T, D]

Usage:
    python bournemouth_aligner/cpp_onix/export_cupe_to_onnx.py --input model.ckpt --output model.onnx
    python bournemouth_aligner/cpp_onix/export_cupe_to_onnx.py --input model.ckpt --output model.onnx --opset 14 --dynamic

    # export the large multi-lingual model (released with v0.1.7)
    python bournemouth_aligner/cpp_onix/export_cupe_to_onnx.py --input models/large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt --output models/large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt.onnx
    
    # export en-us libri1000 model (released with v0.1.7)
    python bournemouth_aligner/cpp_onix/export_cupe_to_onnx.py --input models/en_libri1000_ua01c_e4_val_GER=0.2186.ckpt --output models/en_libri1000_ua01c_e4_val_GER=0.2186.ckpt.onnx
    
"""

import torch
import torch.nn as nn
import torch.onnx
import argparse
import sys
from pathlib import Path

try:
    from bournemouth_aligner.cupe2i.model2i import CUPEEmbeddingsExtractor
except ImportError:
    sys.path.append('./bournemouth_aligner/cupe2i')
    from model2i import CUPEEmbeddingsExtractor


class CUPEPredictWrapper(nn.Module):
    """
    Thin wrapper around CUPEEmbeddingsExtractor that mirrors predict() as a
    traceable forward() returning (logits_class, logits_group, embeddings).

    Input:  audio_batch [batch_size, wav_length]   (raw 16kHz audio window)
    Output: logits_class [B, T, num_phonemes],
            logits_group [B, T, num_groups],
            embeddings   [B, T, D]
    """

    def __init__(self, extractor: CUPEEmbeddingsExtractor):
        super().__init__()
        self.model = extractor.model

    def forward(self, audio_batch):
        x = audio_batch
        x = x.unsqueeze(1)  # (B, T) -> (B, 1, T) for conv1d

        # Feature extraction
        features = self.model.feature_extractor(x)

        # Frequency attention
        att = self.model.freq_attention(features)
        features = features * att

        # Dual-stream processing
        temporal = self.model.temporal_stream(features)
        spectral = self.model.spectral_stream(features)

        # Fuse
        fused = torch.cat([temporal, spectral], dim=1)
        fused = self.model.fusion(fused)
        fused = fused.transpose(1, 2)  # (B, C, T') -> (B, T', C)

        # Embeddings
        embeddings = self.model.window_processor(fused)

        # Classification heads
        logits_class = self.model.classifier(embeddings)
        logits_group = self.model.group_classifier(embeddings)

        return logits_class, logits_group, embeddings


class CUPEONNXExporter:
    """Export CUPE PyTorch models to ONNX format"""

    def __init__(self, pytorch_ckpt_path, onnx_output_path, opset_version=14,
                 dynamic_axes=True, verify_export=True, simplify=False,
                 window_size_ms=120, sample_rate=16000):
        self.pytorch_ckpt_path = Path(pytorch_ckpt_path)
        self.onnx_output_path = Path(onnx_output_path)
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes
        self.verify_export = verify_export
        self.simplify = simplify

        # Audio window params
        self.window_size_samples = int(window_size_ms * sample_rate / 1000)

        if not self.pytorch_ckpt_path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {pytorch_ckpt_path}")
        self.onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_pytorch_model(self):
        """Load CUPE model from checkpoint and wrap for export."""
        print(f"Loading PyTorch checkpoint: {self.pytorch_ckpt_path}")

        extractor = CUPEEmbeddingsExtractor(str(self.pytorch_ckpt_path), device='cpu')
        extractor.eval()

        # Determine frames_per_window for this window size
        frames_per_window = extractor.model.update_frames_per_window(self.window_size_samples)
        print(f"  Window: {self.window_size_samples} samples -> {frames_per_window.item()} output frames")

        wrapper = CUPEPredictWrapper(extractor)
        wrapper.eval()
        return wrapper

    def create_dummy_input(self):
        """Create dummy raw-audio input matching one window."""
        dummy = torch.randn(1, self.window_size_samples)
        print(f"  Dummy input shape: {dummy.shape}  (batch=1, samples={self.window_size_samples})")
        return dummy

    def export_to_onnx(self, model, dummy_input):
        """Export model to ONNX format"""
        print(f"\nExporting to ONNX (opset {self.opset_version})...")

        input_names = ['audio_window']
        output_names = ['phoneme_logits', 'group_logits', 'embeddings']

        dynamic_axes = None
        if self.dynamic_axes:
            dynamic_axes = {
                'audio_window':    {0: 'batch_size'},
                'phoneme_logits':  {0: 'batch_size'},
                'group_logits':    {0: 'batch_size'},
                'embeddings':      {0: 'batch_size'},
            }

        torch.onnx.export(
            model,
            dummy_input,
            str(self.onnx_output_path),
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

        size_mb = self.onnx_output_path.stat().st_size / (1024 ** 2)
        print(f"  ONNX export successful: {self.onnx_output_path} ({size_mb:.2f} MB)")

    def verify_onnx_export(self, pytorch_model, dummy_input):
        """Verify ONNX model produces same outputs as PyTorch model"""
        print(f"\nVerifying ONNX export...")

        try:
            import onnx
            import onnxruntime as ort

            onnx_model = onnx.load(str(self.onnx_output_path))
            onnx.checker.check_model(onnx_model)
            print("  ONNX model is valid")

            ort_session = ort.InferenceSession(str(self.onnx_output_path))

            with torch.no_grad():
                pytorch_outputs = pytorch_model(dummy_input)

            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            onnx_outputs = ort_session.run(None, ort_inputs)

            all_close = True
            names = ['phoneme_logits', 'group_logits', 'embeddings']
            for i, (pt_out, onnx_out) in enumerate(zip(pytorch_outputs, onnx_outputs)):
                pt_np = pt_out.numpy()
                max_diff = abs(pt_np - onnx_out).max()
                mean_diff = abs(pt_np - onnx_out).mean()
                name = names[i] if i < len(names) else f"output_{i}"
                status = "OK" if max_diff < 1e-4 else "MISMATCH"
                print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}  [{status}]")
                if max_diff >= 1e-4:
                    all_close = False

            if all_close:
                print("  Verification passed!")
            else:
                print("  WARNING: outputs differ between PyTorch and ONNX")

        except ImportError:
            print("  Verification requires onnx and onnxruntime: pip install onnx onnxruntime")
        except Exception as e:
            print(f"  Verification failed: {e}")

    def simplify_onnx(self):
        """Simplify ONNX model using onnx-simplifier"""
        print(f"\nSimplifying ONNX model...")
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(str(self.onnx_output_path))
            model_simplified, check = simplify(onnx_model)

            if check:
                simplified_path = self.onnx_output_path.with_suffix('.simplified.onnx')
                onnx.save(model_simplified, str(simplified_path))
                orig_mb = self.onnx_output_path.stat().st_size / (1024 ** 2)
                simp_mb = simplified_path.stat().st_size / (1024 ** 2)
                print(f"  Simplified: {orig_mb:.2f} MB -> {simp_mb:.2f} MB  ({simplified_path})")
            else:
                print("  Simplification check failed")

        except ImportError:
            print("  Simplification requires onnx-simplifier: pip install onnx-simplifier")
        except Exception as e:
            print(f"  Simplification failed: {e}")

    def run(self):
        """Run the complete export pipeline"""
        print("=" * 60)
        print("CUPE PyTorch to ONNX Converter")
        print("=" * 60)

        try:
            pytorch_model = self.load_pytorch_model()
            dummy_input = self.create_dummy_input()
            self.export_to_onnx(pytorch_model, dummy_input)

            if self.verify_export:
                self.verify_onnx_export(pytorch_model, dummy_input)
            if self.simplify:
                self.simplify_onnx()

            print("\n" + "=" * 60)
            print(f"Export completed: {self.onnx_output_path}")
            print("=" * 60)

        except Exception as e:
            print("\n" + "=" * 60)
            print(f"Export failed: {e}")
            print("=" * 60)
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CUPE PyTorch model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bournemouth_aligner/cpp_onix/export_cupe_to_onnx.py -i model.ckpt -o model.onnx
  python bournemouth_aligner/cpp_onix/export_cupe_to_onnx.py -i model.ckpt -o model.onnx --dynamic
  python bournemouth_aligner/cpp_onix/export_cupe_to_onnx.py -i model.ckpt -o model.onnx --simplify
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Path to PyTorch checkpoint (.ckpt)')
    parser.add_argument('--output', '-o', required=True, help='Path to output ONNX model (.onnx)')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version (default: 14)')
    parser.add_argument('--no-dynamic', action='store_true', help='Disable dynamic batch axis (fixed batch=1)')
    parser.add_argument('--no-verify', action='store_true', help='Skip verification step')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model (requires onnxsim)')
    args = parser.parse_args()

    exporter = CUPEONNXExporter(
        pytorch_ckpt_path=args.input,
        onnx_output_path=args.output,
        opset_version=args.opset,
        dynamic_axes=not args.no_dynamic,
        verify_export=not args.no_verify,
        simplify=args.simplify,
    )
    exporter.run()


if __name__ == "__main__":
    main()
