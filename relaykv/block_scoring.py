from dataclasses import dataclass
import torch

from .block_metadata import BlockMetadata


@dataclass
class BlockScore:
    layer_idx: int
    block_id: int
    start: int
    end: int
    score: float

    def summary(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "block_id": self.block_id,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


def score_blocks_with_query(
    metadata: list[BlockMetadata],
    query: torch.Tensor,
) -> list[BlockScore]:
    """
    query shape: [1, heads, head_dim]
    metadata k_mean shape: [1, heads, head_dim]
    """
    scores: list[BlockScore] = []

    query = query.detach().cpu().float()

    for m in metadata:
        k_mean = m.k_mean.float()
        score = float((query * k_mean).sum().item())

        scores.append(
            BlockScore(
                layer_idx=m.layer_idx,
                block_id=m.block_id,
                start=m.start,
                end=m.end,
                score=score,
            )
        )

    return scores


def top_k_blocks(
    scores: list[BlockScore],
    k: int = 4,
) -> list[BlockScore]:
    if k <= 0:
        raise ValueError("k must be > 0")
    return sorted(scores, key=lambda x: x.score, reverse=True)[:k]