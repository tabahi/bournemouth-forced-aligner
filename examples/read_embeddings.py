"""
Read the per-phoneme embeddings that `process_srt_file(..., extract_embeddings=True, vspt_path=...)`
saves (the CLI's `--embeddings PATH` writes the same file). Run from the repository root.
"""
import os

import torch

EMBEDDINGS_PATH = "output_vspt_embd.pt"


def _write_embeddings():
    """Align a sample clip and save its embeddings, so this example runs on its own."""
    from bournemouth_aligner import BFA

    aligner = BFA(preset="en-us", duration_max=10)
    aligner.process_srt_file("examples/samples/LJSpeech/LJ001-0002.srt.json",
                             "examples/samples/LJSpeech/LJ001-0002.wav",
                             ts_out_path="output_vspt.vs.json",
                             extract_embeddings=True, vspt_path=EMBEDDINGS_PATH, do_groups=True)


def _read_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        _write_embeddings()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One tensor per segment: phoneme embeddings, and phoneme-group embeddings (do_groups=True).
    vspt_p_embd, vspt_g_embd = torch.load(EMBEDDINGS_PATH, map_location=device, weights_only=True)

    print("Total segments:", len(vspt_p_embd))
    total_phonemes, embeddings_dim = vspt_p_embd[0].shape
    print("Phonemes in the first segment:", total_phonemes)
    print("Embedding dimension:", embeddings_dim)
    if vspt_g_embd is not None and vspt_g_embd[0] is not None:
        print("Group embeddings of the first segment:", tuple(vspt_g_embd[0].shape))


if __name__ == "__main__":
    _read_embeddings()
