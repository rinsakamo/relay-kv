from dataclasses import dataclass


@dataclass
class Block:
    block_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def build_blocks(cold_range: tuple[int, int], block_size: int = 128) -> list[Block]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    start, end = cold_range
    if start < 0 or end < start:
        raise ValueError("invalid cold_range")

    blocks: list[Block] = []
    cursor = start
    block_id = 0

    while cursor < end:
        block_end = min(cursor + block_size, end)
        blocks.append(Block(block_id=block_id, start=cursor, end=block_end))
        cursor = block_end
        block_id += 1

    return blocks