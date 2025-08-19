import torch


def _read_embeddings():
    embeddings_path = "output_vspt_embd.pt"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the embeddings
    vspt_p_embd, vspt_g_embds = torch.load(embeddings_path, map_location=device, weights_only=True)

    # Analyze the embeddings
    print("Total segments:", len(vspt_p_embd))
    print("First segment embeddings shape:", vspt_p_embd[0].shape)
    print("First segment embeddings shape:", vspt_g_embds[0].shape)


    total_phonemes = vspt_p_embd[0].shape[0]
    embeddings_dim = vspt_p_embd[0].shape[1]
    print("Phoneme embeddings:")
    print("Total phonemes:", total_phonemes)       # 1 to 625
    print("Embeddings dimension:", embeddings_dim) # 512

if __name__ == "__main__":
    _read_embeddings()