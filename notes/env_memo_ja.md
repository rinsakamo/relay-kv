# 環境メモ

- 開発本体は WSL2 でやる
- Windows 側は普段使いと GUI 用
- repo は `/mnt/c/...` ではなく WSL 側に置く
- 最初は PyTorch + Transformers で進める
- いきなり llama.cpp や vLLM には行かない

## 確認したいこと

- `nvidia-smi` が WSL から見えるか
- `torch.cuda.is_available()` が True になるか
- Qwen2.5-1.5B が最初に無理なく動くか
