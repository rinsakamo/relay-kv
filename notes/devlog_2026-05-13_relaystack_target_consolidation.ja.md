# Devlog 2026-05-13 JST: RelayStack target consolidation

## 概要

2026-05-13 JST 時点で、RelayKV の文書上の位置づけを整理した。

今回の整理では、RelayKV を単なる小規模 KV 近似プロトタイプとしてだけでなく、**VRAM-aware active-context KV routing layer** かつ **tiered KV memory manager** として定義し直した。そのうえで、より上位の設計方向として **RelayMEM** と **RelayStack** の名前を明確化した。

## 命名整理

- **RelayKV**: モデルの対応最大コンテキストの内側で、active context の KV block を GPU / RAM / SSD 前提で管理し、VRAM 予算下で working set を制御する層
- **RelayMEM**: モデルの active context の外側にある max-context-external memory 層。RAG、Summary Memory、Profile Memory、Episode Memory、Structured Memory、Context Assembly、KV checkpoint metadata を担当
- **RelayStack**: `RelayKV + RelayMEM + runtime policy`

重要なのは、RelayKV 単体ではモデルの trained / supported max context を伸ばしたことにはならない、という境界を明文化した点である。

## 評価ターゲット整理

研究評価の柱は次の 3 本に整理した。

1. **Code / structured retrieval**
   `Qwen2.5-Coder-7B-Instruct-AWQ`
2. **Ultra-long context**
   `Qwen2.5-7B-Instruct-1M`
3. **Japanese long-form / character consistency**
   `LLM-jp-4-8B`

加えて、実用デモのターゲットとして次を置く。

4. **Low-VRAM AI character demo**
   `Open-LLM-VTuber + LLM-jp-4 8B 4bit + RelayMEM + RelayKV`

これにより、RelayKV 単独評価と、RelayMEM を含む統合デモ評価を分けて考えやすくした。

## Open-LLM-VTuber をターゲットに置く理由

Open-LLM-VTuber は、長期対話、キャラクタ一貫性、低 VRAM 制約、TTS / ASR / Avatar との共存という条件が同時に立つため、RelayStack の実用ターゲットとしてわかりやすい。

特に 12GB VRAM 級では、

```text
total VRAM
- model weights
- TTS
- ASR
- Avatar
- safety margin
= LLM working KV budget
```

という考え方が重要になる。ここで RelayKV は残余 VRAM を working KV budget として扱う層、RelayMEM は長期記憶から active context を組み立てる層として分離される。

## User-Gated Fallback

ライブ用途では、遅い fallback を無断で発火させると UX を壊しやすい。そのため runtime policy の一部として **User-Gated Fallback** を追加で整理した。

主な状態は:

- `PROPOSE_FALLBACK`
- `WAIT_USER_APPROVAL`

主な意図は、SSD recall や FullKV tiered fallback のような重い経路について、AI Vtuber が

> 昔の記憶を探してもいい？

のように確認できるようにすることである。

## max-context-external memory の切り分け

今回の整理で重要だったのは、**max-context-external memory は RelayMEM 側の責務**であり、RelayKV は **active context 内の KV routing** を担当する、という切り分けを明示した点である。

そのため、以下は RelayMEM 側の概念として扱う。

- Sliding Window の外に出た情報の要約
- Summary Memory
- RAG / Vector Memory
- Structured Memory
- Episode Memory
- Profile Memory
- KV Checkpoint / Prefix Cache
- Hybrid Memory Router

一方で RelayKV は、すでに active context に入った範囲の KV をどの tier に置き、どの working set を GPU に残すか、という問題を扱う。

## Profile Memory への名称統一

従来の viewer profile 的な表現は、AI Vtuber 向けには有効でも汎用性が弱い。そのため文書上は **Profile Memory** を基本名称とし、viewer profile は AI Vtuber 用の specialization として扱う方針にした。

これにより、個人アシスタント、長編執筆、AI キャラクタ運用のいずれにも通る説明にしやすくなった。

## 現時点の位置づけ

今回の変更は主に**ターゲット統合と責務境界の明文化**であり、RelayMEM 統合や Open-LLM-VTuber 統合が実装済みだと主張するものではない。

現状の実装中心は引き続き RelayKV の PyTorch prototype path であり、

```text
KV split
→ CPU cold offload
→ blockify
→ metadata
→ scoring
→ retrieval
→ candidate KV
→ working KV
→ attention comparison
```

が main comparison path である。この前提を保ったまま、README / docs / notes の上位説明を RelayStack 方向へ揃えた。
