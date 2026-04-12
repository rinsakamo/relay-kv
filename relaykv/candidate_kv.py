from dataclasses import dataclass
import torch

from .block_retrieval import RetrievedBlock


@dataclass
class CandidateKV:
    layer_idx: int
    start: int
    end: int
    k: torch.Tensor
    v: torch.Tensor
    selected_spans: list[list[int]]
    is_contiguous: bool

    @property
    def length(self) -> int:
        return self.end - self.start

    def summary(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "range_length": self.end - self.start,
            "tensor_length": int(self.k.shape[2]),
            "k_shape": list(self.k.shape),
            "v_shape": list(self.v.shape),
            "k_device": str(self.k.device),
            "v_device": str(self.v.device),
            "selected_spans": self.selected_spans,
            "is_contiguous": self.is_contiguous,
        }


def build_candidate_kv(retrieved_blocks: list[RetrievedBlock]) -> CandidateKV:
    if not retrieved_blocks:
        raise ValueError("retrieved_blocks must not be empty")

    blocks = sorted(retrieved_blocks, key=lambda b: b.start)

    layer_idx = blocks[0].layer_idx
    for b in blocks:
        if b.layer_idx != layer_idx:
            raise ValueError("all retrieved blocks must belong to the same layer")
        if b.k is None or b.v is None:
            raise ValueError("retrieved block has no KV tensors")

    k = torch.cat([b.k for b in blocks], dim=2)
    v = torch.cat([b.v for b in blocks], dim=2)

    selected_spans = [[b.start, b.end] for b in blocks]
    is_contiguous = all(
        blocks[i].end == blocks[i + 1].start
        for i in range(len(blocks) - 1)
    )

    return CandidateKV(
        layer_idx=layer_idx,
        start=blocks[0].start,
        end=blocks[-1].end,
        k=k,
        v=v,
        selected_spans=selected_spans,
        is_contiguous=is_contiguous,
    )