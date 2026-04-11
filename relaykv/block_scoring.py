from dataclasses import dataclass
import torch

from .block_metadata import BlockMetadata
from .cold_cache import ColdBlock


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


def score_block_query_to_block_max(
    block_k: torch.Tensor,
    query: torch.Tensor,
) -> float:
    """
    block_k: [1, heads, block_len, head_dim]
    query:   [1, heads, head_dim]
    """
    q = query.detach().cpu().float().unsqueeze(2)   # [1, heads, 1, head_dim]
    k = block_k.detach().cpu().float()              # [1, heads, block_len, head_dim]

    # [1, heads, block_len]
    per_token_per_head = torch.sum(q * k, dim=-1)

    # [1, block_len]
    per_token = per_token_per_head.mean(dim=1)

    return float(per_token.max().item())


def score_blocks_with_query(
    metadata: list[BlockMetadata],
    query: torch.Tensor,
    variant: str = "mean_only",
    norm_weight: float = 0.0,
    all_blocks: list[ColdBlock] | None = None,
) -> list[BlockScore]:
    """
    metadata: list[BlockMetadata]
    query: [1, heads, head_dim]
    """

    query = query.detach().cpu().float()

    block_map = None
    if variant == "query_to_block_max":
        if all_blocks is None:
            raise ValueError("all_blocks is required for query_to_block_max")
        block_map = {
            (block.layer_idx, block.block_id): block
            for block in all_blocks
        }

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

        elif variant == "mean_plus_max":
            per_head_scores = torch.sum(query * k_mean, dim=-1)  # [1, heads]
            max_score = torch.max(per_head_scores).item()
            score_value = mean_score + max_score

        elif variant == "query_to_block_max":
            block = block_map[(m.layer_idx, m.block_id)]
            if block.k is None:
                raise ValueError(f"Block has no K tensor: {(m.layer_idx, m.block_id)}")
            score_value = score_block_query_to_block_max(block.k, query)

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