from dataclasses import dataclass

from .cold_cache import ColdBlock
from .block_scoring import BlockScore


@dataclass
class RetrievedBlock:
    layer_idx: int
    block_id: int
    start: int
    end: int
    k: object
    v: object

    @property
    def length(self) -> int:
        return self.end - self.start

    def summary(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "block_id": self.block_id,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "k_shape": list(self.k.shape) if self.k is not None else None,
            "v_shape": list(self.v.shape) if self.v is not None else None,
            "k_device": str(self.k.device) if self.k is not None else None,
            "v_device": str(self.v.device) if self.v is not None else None,
        }


def retrieve_blocks(
    all_blocks: list[ColdBlock],
    selected_scores: list[BlockScore],
) -> list[RetrievedBlock]:
    block_map = {
        (block.layer_idx, block.block_id): block
        for block in all_blocks
    }

    retrieved: list[RetrievedBlock] = []

    for score in selected_scores:
        key = (score.layer_idx, score.block_id)
        if key not in block_map:
            raise KeyError(f"Block not found for key={key}")

        block = block_map[key]
        retrieved.append(
            RetrievedBlock(
                layer_idx=block.layer_idx,
                block_id=block.block_id,
                start=block.start,
                end=block.end,
                k=block.k,
                v=block.v,
            )
        )

    return retrieved