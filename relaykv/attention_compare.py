from dataclasses import dataclass
import torch


@dataclass
class AttentionCompareResult:
    full_output: torch.Tensor
    approx_output: torch.Tensor
    mean_abs_diff: float
    max_abs_diff: float
    l2_diff: float

    def summary(self) -> dict:
        return {
            "full_output_shape": list(self.full_output.shape),
            "approx_output_shape": list(self.approx_output.shape),
            "mean_abs_diff": self.mean_abs_diff,
            "max_abs_diff": self.max_abs_diff,
            "l2_diff": self.l2_diff,
        }


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    query: [1, heads, 1, head_dim]
    key:   [1, heads, seq_len, head_dim]
    value: [1, heads, seq_len, head_dim]
    output:[1, heads, 1, head_dim]
    """
    head_dim = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, value)
    return output


def compare_attention_outputs(
    query: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    approx_k: torch.Tensor,
    approx_v: torch.Tensor,
) -> AttentionCompareResult:
    # 比較しやすいように CPU float32 に揃える
    query = query.detach().cpu().float()
    full_k = full_k.detach().cpu().float()
    full_v = full_v.detach().cpu().float()
    approx_k = approx_k.detach().cpu().float()
    approx_v = approx_v.detach().cpu().float()

    full_output = scaled_dot_product_attention(query, full_k, full_v)
    approx_output = scaled_dot_product_attention(query, approx_k, approx_v)

    diff = (full_output - approx_output).abs()

    return AttentionCompareResult(
        full_output=full_output,
        approx_output=approx_output,
        mean_abs_diff=float(diff.mean().item()),
        max_abs_diff=float(diff.max().item()),
        l2_diff=float(torch.norm(full_output - approx_output).item()),
    )