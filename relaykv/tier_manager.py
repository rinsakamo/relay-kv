from dataclasses import dataclass


@dataclass
class TierSplit:
    cold_start: int
    cold_end: int
    hot_start: int
    hot_end: int

    @property
    def cold_range(self) -> tuple[int, int]:
        return (self.cold_start, self.cold_end)

    @property
    def hot_range(self) -> tuple[int, int]:
        return (self.hot_start, self.hot_end)


class TierManager:
    def __init__(self, hot_window: int = 1024) -> None:
        if hot_window <= 0:
            raise ValueError("hot_window must be > 0")
        self.hot_window = hot_window

    def split_range(self, seq_len: int) -> TierSplit:
        if seq_len < 0:
            raise ValueError("seq_len must be >= 0")

        hot_start = max(0, seq_len - self.hot_window)
        return TierSplit(
            cold_start=0,
            cold_end=hot_start,
            hot_start=hot_start,
            hot_end=seq_len,
        )