"""
One call that turns a model on disk into the right extractor.

The family is read off the model rather than asked for, so `ckpt_path=` needs nothing else
said about it: a p3cupe model loads as p3cupe, a p4mbfa model as p4mbfa. The path is a release
directory (config.json + model.safetensors) or a training checkpoint file.
"""

from .segmental.extractor import checkpoint_family, load_checkpoint


def load_extractor(ckpt_path, device='cpu', sample_rate=16000, verbose=1):
    """Load `ckpt_path` as a `CupeExtractor` or an `MbfaExtractor`, whichever it is."""
    checkpoint = load_checkpoint(ckpt_path)
    if checkpoint_family(checkpoint) == 'p4mbfa':
        from .p4mbfa import MbfaExtractor as Extractor
    else:
        from .p3cupe import CupeExtractor as Extractor
    return Extractor(ckpt_path, device=device, sample_rate=sample_rate, verbose=verbose,
                     checkpoint=checkpoint)
