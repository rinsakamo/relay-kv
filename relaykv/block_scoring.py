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
    variant: str = "mean_only",
    norm_weight: float = 0.0,
) -> list[BlockScore]:
    """
    metadata: list[BlockMetadata]
    query: [1, heads, head_dim]
    """

    query = query.detach().cpu().float()

    scores: list[BlockScore] = []

    for m in metadata:
        k_mean = m.k_mean.detach().cpu().float()  # [1, heads, head_dim]

        mean_score = torch.sum(query * k_mean).item()

        if variant == "mean_only":
            score_value = mean_score

        elif variant == "mean_plus_norm":
            score_value = mean_score + norm_weight * float(m.k_norm)

        elif variant == "mean_plus_vnorm":
            score_value = mean_score + norm_weight * float(m.v_norm)

        elif variant == "headwise_max_mean":
            per_head_scores = torch.sum(query * k_mean, dim=-1)  # [1, heads]
            score_value = torch.max(per_head_scores).item()

        else:
            raise ValueError(f"Unsupported scoring variant: {variant}")
        scores.append(
            BlockScore(
                layer_idx=m.layer_idx,
                block_id=m.block_id,
                start=m.start,
                end=m.end,
                score=float(score_value),
            )
        )

    scores.sort(key=lambda x: x.score, reverse=True)
    return scores


def top_k_blocks(scores: list[BlockScore], k: int) -> list[BlockScore]:
    return scores[:k]