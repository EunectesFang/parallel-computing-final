"""Tiny CPU text encoder for source and target prefixes.

Design note: pooling uses a boolean mask over non-PAD tokens instead of
`pack_padded_sequence`. With 16-token windows the extra speed is negligible,
and the masked-mean formulation also handles sequences of length zero (and
batches that mix zero-length rows with non-empty rows), which packed sequences
cannot represent.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn

__all__ = ["Vocab", "TextEncoder"]


class Vocab:
    """Word vocabulary with reserved PAD=0 and UNK=1 ids."""

    PAD = 0
    UNK = 1

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str | None] = [None, None]

    @classmethod
    def from_texts(cls, texts: Iterable) -> "Vocab":
        """Build a vocabulary in first-seen order.

        Each item may be a raw sentence (split on whitespace) or an iterable of
        already-tokenized words.
        """
        vocab = cls()
        for item in texts:
            words = item.split() if isinstance(item, str) else list(item)
            for word in words:
                vocab.add(word)
        return vocab

    def add(self, token: str) -> int:
        """Add `token` if new and return its id (PAD/UNK words are not remapped)."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        token_id = len(self.id_to_token)
        self.token_to_id[token] = token_id
        self.id_to_token.append(token)
        return token_id

    def encode(self, words: Sequence[str], window: int) -> tuple[int, ...]:
        """Encode the last `window` words; unknown words become UNK."""
        words = list(words)[-window:] if window > 0 else []
        return tuple(self.token_to_id.get(word, self.UNK) for word in words)

    def __len__(self) -> int:
        """Total embedding rows, including the two reserved ids."""
        return len(self.id_to_token)


class TextEncoder(nn.Module):
    """Two embeddings and two GRUs, pooled and projected to `2 * hidden`."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        emb: int = 32,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.out_dim = 2 * hidden
        self.src_embed = nn.Embedding(src_vocab_size, emb, padding_idx=Vocab.PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, emb, padding_idx=Vocab.PAD)
        self.src_gru = nn.GRU(emb, hidden, batch_first=True)
        self.tgt_gru = nn.GRU(emb, hidden, batch_first=True)
        self.proj = nn.Linear(self.out_dim, self.out_dim)

    def _pool(self, embed: nn.Embedding, gru: nn.GRU, ids: torch.Tensor) -> torch.Tensor:
        batch = ids.size(0)
        if ids.size(1) == 0:
            return torch.zeros(batch, self.hidden, dtype=self.proj.weight.dtype, device=ids.device)
        valid = ids.ne(Vocab.PAD)
        outputs, _ = gru(embed(ids))
        mask = valid.unsqueeze(-1).to(outputs.dtype)
        counts = valid.sum(dim=1, keepdim=True).clamp(min=1).to(outputs.dtype)
        pooled = (outputs * mask).sum(dim=1) / counts
        # Rows that are entirely PAD contribute an exact zero.
        return pooled * (valid.sum(dim=1, keepdim=True) > 0).to(outputs.dtype)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """Return a `[batch, 2 * hidden]` vector for paired prefix id tensors."""
        src = self._pool(self.src_embed, self.src_gru, src_ids)
        tgt = self._pool(self.tgt_embed, self.tgt_gru, tgt_ids)
        output = self.proj(torch.cat([src, tgt], dim=-1))
        has_tokens = (src_ids.ne(Vocab.PAD).any(dim=1) | tgt_ids.ne(Vocab.PAD).any(dim=1))
        # A batch row with no source and no target tokens must encode to zeros,
        # not to the projection bias.
        return output * has_tokens.to(output.dtype).unsqueeze(-1)
