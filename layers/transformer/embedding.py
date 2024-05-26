import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int=0) -> None:
        super(Embedding, self).__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )
        self.hidden_size = hidden_size

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(input_batch)