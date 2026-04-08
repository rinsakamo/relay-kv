from dataclasses import dataclass

import torch

from .cold_cache import ColdCache, ColdSegment
from .tier_manager import TierSplit


@dataclass
class HotKV:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]


def split_dynamic_cache_layers(
    layers,
    split: TierSplit,
) -> tuple[HotKV, ColdCache]:
    hot_keys: list[torch.Tensor] = []
    hot_values: list[torch.Tensor] = []
    cold_cache = ColdCache()

    cold_start, cold_end = split.cold_range
    hot_start, hot_end = split.hot_range

    for layer_idx, layer in enumerate(layers):
        key_states = layer.keys
        value_states = layer.values

        # shape: [batch, heads, seq_len, head_dim]
        cold_k = key_states[:, :, cold_start:cold_end, :].detach().cpu()
        cold_v = value_states[:, :, cold_start:cold_end, :].detach().cpu()

        hot_k = key_states[:, :, hot_start:hot_end, :].detach()
        hot_v = value_states[:, :, hot_start:hot_end, :].detach()

        cold_cache.add_segment(
            ColdSegment(
                layer_idx=layer_idx,
                start=cold_start,
                end=cold_end,
                k=cold_k,
                v=cold_v,
            )
        )

        hot_keys.append(hot_k)
        hot_values.append(hot_v)

    return HotKV(keys=hot_keys, values=hot_values), cold_cache