from dataclasses import dataclass
import torch

from .candidate_kv import CandidateKV


@dataclass
class WorkingKV:
    layer_idx: int
    candidate_start: int
    candidate_end: int
    hot_start: int
    hot_end: int
    k: torch.Tensor
    v: torch.Tensor

    @property
    def length(self) -> int:
        return self.k.shape[2]

    def summary(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "candidate_range": [self.candidate_start, self.candidate_end],
            "hot_range": [self.hot_start, self.hot_end],
            "length": self.length,
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

    k = torch.cat([candidate_kv.k, hot_k], dim=2)
    v = torch.cat([candidate_kv.v, hot_v], dim=2)

    return WorkingKV(
        layer_idx=candidate_kv.layer_idx,
        candidate_start=candidate_kv.start,
        candidate_end=candidate_kv.end,
        hot_start=hot_start,
        hot_end=hot_end,
        k=k,
        v=v,
    )