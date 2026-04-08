from dataclasses import dataclass
import torch

from .cold_cache import ColdBlock


@dataclass
class BlockMetadata:
    layer_idx: int
    block_id: int
    start: int
    end: int
    length: int
    k_mean: torch.Tensor
    v_mean: torch.Tensor
    k_norm: float
    v_norm: float

    def summary(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "block_id": self.block_id,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "k_mean_shape": list(self.k_mean.shape),
            "v_mean_shape": list(self.v_mean.shape),
            "k_norm": self.k_norm,
            "v_norm": self.v_norm,
        }


def build_block_metadata(block: ColdBlock) -> BlockMetadata:
    if block.k is None or block.v is None:
        raise ValueError("block has no KV tensors")

    # shape: [batch, heads, seq_len, head_dim]
    k_mean = block.k.mean(dim=2).cpu()
    v_mean = block.v.mean(dim=2).cpu()

    k_norm = float(block.k.float().norm().item())
    v_norm = float(block.v.float().norm().item())

    return BlockMetadata(
        layer_idx=block.layer_idx,
        block_id=block.block_id,
        start=block.start,
        end=block.end,
        length=block.length,
        k_mean=k_mean,
        v_mean=v_mean,
        k_norm=k_norm,
        v_norm=v_norm,
    )


def build_metadata_for_blocks(blocks: list[ColdBlock]) -> list[BlockMetadata]:
    return [build_block_metadata(block) for block in blocks]