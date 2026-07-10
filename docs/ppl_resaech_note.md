# PPL 研究ノート (autoresearch/jul10)

目標: Qwen/Qwen3-1.7B の三値量子化モデルで `scripts/run_ppl.py` の PPL を最小化する。

## 現在の最小ppl値

**38.81** - `outputs/qwen3-1.7b-catq-v3` (無変更ベースライン、seed 0)

参考記録: `autoresearch/jul6` の最小値は **28.75**
(`--grad-clip 0.5 --lora-rank 32 --epochs 80 --lr 2e-3`)。今回の実験ではまず
この安定化機構を最小構成で移植し、28.75 未満を更新対象とする。

## 記録

### 2026-07-10 ベースライン確立

- 新規ブランチ `autoresearch/jul10` を `ppl_research` から作成。
- `uv run pytest`: **22 件全 PASS**。
- 初回量子化は `program.md` の許可に従って省略し、既存の量子化済みモデルを評価。
- `uv run python scripts/run_ppl.py --models outputs/qwen3-1.7b-catq-v3`: **PPL 38.81**。
- GPU: NVIDIA GeForce RTX 5090 32 GB。開始時に競合プロセスなし。

### autoresearch/jul6 から引き継ぐ知見

- 最大の改善は勾配クリップによる最適化スパイクの抑制。clip 1.0 で 32.35、
  clip 0.5 で 31.86 まで改善し、シード分散もほぼ消失した。
- 安定化後は LoRA rank 32、epochs 80、LM 学習率 2e-3 の改善がほぼ加法的に働き、
  **28.75** に到達した。
- 既知の失敗: rank 64、lr 4e-3、lora_lr 1e-3、epochs 120、seq_len 1024、
  3-pass、gamma 0.6、サリエンス順量子化、重み付き損失、hard-STE polish。
- 0.6B や低予算 1.7B からフル 1.7B への順位転移は不安定だったため、最終判断は
  1.7B・512 samples・seq_len 2048 で行う。

### 次の探索

- `jul6` の成功に必要な `--grad-clip` を単独で移植する。
- 既知ベスト近傍で未完了だった epochs 100 と clip 0.7 の結果を取り、
  epochs / learning-rate / clipping の相互作用を絞り込む。
