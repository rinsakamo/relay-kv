from dataclasses import dataclass
from typing import Any


@dataclass
class ColdBlock:
    layer_idx: int
    block_id: int
    start: int
    end: int
    k: Any | None = None
    v: Any | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class ColdSegment:
    layer_idx: int
    start: int
    end: int
    k: Any | None = None
    v: Any | None = None

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_blocks(self, block_size: int = 128) -> list["ColdBlock"]:
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.k is None or self.v is None:
            raise ValueError("segment has no KV tensors")

        blocks: list[ColdBlock] = []
        cursor = self.start
        block_id = 0

        while cursor < self.end:
            block_end = min(cursor + block_size, self.end)

            local_start = cursor - self.start
            local_end = block_end - self.start

            # shape: [batch, heads, seq_len, head_dim]
            block_k = self.k[:, :, local_start:local_end, :]
            block_v = self.v[:, :, local_start:local_end, :]

            blocks.append(
                ColdBlock(
                    layer_idx=self.layer_idx,
                    block_id=block_id,
                    start=cursor,
                    end=block_end,
                    k=block_k,
                    v=block_v,
                )
            )

            cursor = block_end
            block_id += 1

        return blocks


class ColdCache:
    def __init__(self) -> None:
        self.segments: list[ColdSegment] = []

    def add_segment(self, segment: ColdSegment) -> None:
        self.segments.append(segment)

    def clear(self) -> None:
        self.segments.clear()

    def summary(self) -> list[dict]:
        return [
            {
                "layer_idx": seg.layer_idx,
                "start": seg.start,
                "end": seg.end,
                "length": seg.length,
                "k_shape": list(seg.k.shape) if seg.k is not None else None,
                "v_shape": list(seg.v.shape) if seg.v is not None else None,
                "k_device": str(seg.k.device) if seg.k is not None else None,
                "v_device": str(seg.v.device) if seg.v is not None else None,
            }
            for seg in self.segments
        ]

    def blockify(self, block_size: int = 128) -> list[ColdBlock]:
        all_blocks: list[ColdBlock] = []
        for seg in self.segments:
            all_blocks.extend(seg.to_blocks(block_size=block_size))
        return all_blocks