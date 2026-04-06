# RelayKV Experiment Spec v0.1

## 1. Goal

RelayKV の最初の目的は、**限られたローカル計算資源で長文脈推論を実用化できるか**を検証することです。

主に次を評価します。

- GPU-only KV cache に対して、GPU/CPU tiered KV がどれだけ文脈長を延ばせるか
- block-wise cold KV recall がどれだけ転送量とメモリ圧力を減らせるか
- candidate recall を保ったまま attention quality を維持できるか

---

## 2. Hardware

- **Host OS**: Windows 11
- **Dev / Experiment OS**: Ubuntu on WSL2
- **GPU**: NVIDIA GeForce RTX 3060 12GB
- **CPU**: Intel Core i5-12500
- **RAM**: 32GB

---

## 3. Software

- **Python**: 3.11
- **Framework**: PyTorch
- **Model interface**: Hugging Face Transformers
- **Primary runtime for experiments**: PyTorch + Transformers
- **Environment**: WSL2 Ubuntu
- **Versioning**: `requirements.txt` または `pyproject.toml` で固定

初期段階では、`llama.cpp` や `vLLM` ではなく、**PyTorch + Transformers をベースライン backend** にします。  
理由は、KV の持ち方を自分で制御しやすいからです。

---

## 4. Model Family

### Primary model family

**Qwen2.5-Instruct**

### Planned scale path

- **Stage A**: Qwen2.5-1.5B-Instruct
- **Stage B**: Qwen2.5-3B-Instruct
- **Stage C**: Qwen2.5-7B-Instruct

### Why this family

- 同一系列でサイズ差実験がしやすい
- instruct 系として実用寄り
- 小→中→大で段階的に RelayKV を育てやすい
- 3B / 7B がローカル実用感と研究の両方にちょうど良い

---

## 5. Experimental Phases

### Phase 0: Baseline Profiling

通常の GPU-only KV cache で長文脈推論を計測する。

#### Purpose

- 基準となる memory / latency 曲線を作る
- どの入力長で苦しくなるかを確認する

#### Conditions

- `batch_size = 1`
- `generate_length = fixed short output`
- `context_length = 2k / 4k / 8k / 16k`
- `model = 1.5B`, then `3B`

#### Metrics

- peak GPU memory
- TTFT
- decode latency
- tokens/sec
- output logs

---

## 6. Fixed Evaluation Conditions

- **batch size**: 1
- **sampling**: deterministic 寄りに固定
- **seed**: fixed
- **prompt set**: 固定 JSON
- **generation length**: 短め固定

---

## 7. Metrics

### System metrics

- peak GPU memory
- peak CPU memory
- TTFT
- average decode latency per token
- tokens/sec
- CPU→GPU transfer volume

### Approximation metrics

- recall@k
- attention output difference
- logit difference
- generated output stability

---

## 8. Repository Outputs

```text
results/
├─ metrics.csv
├─ configs/
├─ logs/
├─ plots/
└─ summaries/
```

---

## 9. Initial Success Criteria

- GPU-only より長い context length を扱える
- peak GPU memory を明確に下げられる
- recall-aware block selection が full recall より低転送で動く
- quality degradation が小さい範囲に収まる

---

## 10. Final Decisions

### Official model family

**Qwen2.5-Instruct**

### Starting model

**Qwen2.5-1.5B-Instruct**

### Main evaluation model

**Qwen2.5-3B-Instruct**

### Practical validation model

**Qwen2.5-7B-Instruct**
