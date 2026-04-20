from dataclasses import dataclass
import torch

from .candidate_kv import CandidateKV


@dataclass
class WorkingKV:
    layer_idx: int
    anchor_start: int | None
    anchor_end: int | None
    candidate_start: int
    candidate_end: int
    hot_start: int
    hot_end: int
    k: torch.Tensor
    v: torch.Tensor
    anchor_tensor_length: int
    candidate_tensor_length: int
    anchor_selected_spans: list[list[int]]
    candidate_is_contiguous: bool
    candidate_selected_spans: list[list[int]]

    @property
    def length(self) -> int:
        return self.k.shape[2]

    def summary(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "anchor_range": [self.anchor_start, self.anchor_end]
            if self.anchor_start is not None and self.anchor_end is not None
            else None,
            "candidate_range": [self.candidate_start, self.candidate_end],
            "hot_range": [self.hot_start, self.hot_end],
            "length": self.length,
            "anchor_tensor_length": self.anchor_tensor_length,
            "candidate_tensor_length": self.candidate_tensor_length,
            "anchor_selected_spans": self.anchor_selected_spans,
            "candidate_is_contiguous": self.candidate_is_contiguous,
            "candidate_selected_spans": self.candidate_selected_spans,
            "k_shape": list(self.k.shape),
            "v_shape": list(self.v.shape),
            "k_device": str(self.k.device),
            "v_device": str(self.v.device),
        }


def build_working_kv(
    candidate_kv: CandidateKV,
    hot_k: torch.Tensor,
    hot_v: torch.Tensor,
    hot_range: tuple[int, int],
    anchor_k: torch.Tensor | None = None,
    anchor_v: torch.Tensor | None = None,
    anchor_spans: list[list[int]] | None = None,
) -> WorkingKV:
    """
    candidate_kv: typically on CPU
    hot_k/hot_v: typically on GPU or current device
    shape: [batch, heads, seq_len, head_dim]
    """
    hot_start, hot_end = hot_range

    # device を合わせる。最初は CPU 側に統一でよい
    target_device = candidate_kv.k.device
    hot_k = hot_k.detach().to(target_device)
    hot_v = hot_v.detach().to(target_device)

    anchor_start: int | None = None
    anchor_end: int | None = None
    anchor_tensor_length = 0
    anchor_selected_spans = anchor_spans or []

    parts_k = []
    parts_v = []
    if anchor_k is not None and anchor_v is not None:
        anchor_k = anchor_k.detach().to(target_device)
        anchor_v = anchor_v.detach().to(target_device)
        parts_k.append(anchor_k)
        parts_v.append(anchor_v)
        anchor_tensor_length = int(anchor_k.shape[2])
        if anchor_selected_spans:
            anchor_start = anchor_selected_spans[0][0]
            anchor_end = anchor_selected_spans[-1][1]

    parts_k.extend([candidate_kv.k, hot_k])
    parts_v.extend([candidate_kv.v, hot_v])

    k = torch.cat(parts_k, dim=2)
    v = torch.cat(parts_v, dim=2)

    return WorkingKV(
        layer_idx=candidate_kv.layer_idx,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        candidate_start=candidate_kv.start,
        candidate_end=candidate_kv.end,
        hot_start=hot_start,
        hot_end=hot_end,
        k=k,
        v=v,
        anchor_tensor_length=anchor_tensor_length,
        candidate_tensor_length=int(candidate_kv.k.shape[2]),
        anchor_selected_spans=anchor_selected_spans,
        candidate_is_contiguous=candidate_kv.is_contiguous,
        candidate_selected_spans=candidate_kv.selected_spans,
    )
