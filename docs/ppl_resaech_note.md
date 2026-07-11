# PPL 研究ノート (autoresearch/jul10)

目標: Qwen/Qwen3-1.7B の三値量子化モデルで `scripts/run_ppl.py` の PPL を最小化する。

## 現在の最小ppl値

**28.75** - `outputs/jul9-17b-clip05-r32-e80-lr2`
(`--grad-clip 0.5 --lora-rank 32 --epochs 80 --lr 2e-3`、seed 0)

`autoresearch/jul6` で生成された既知ベストだが、今回も未変更の `run_ppl.py` で
**28.75** を再確認した。今回の新規出力に限った最小値は 29.41。

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

### 2026-07-11 E1: epochs 100 - 失敗 (29.41)

- 構成: 1.7B、512 samples、seq_len 2048、clip 0.5、LoRA rank 32、
  epochs 100、LM lr 2e-3。seed 0 の校正キャッシュを固定。
- 量子化時間: **387.3 分**。全 196 linear を正常に三値化し、NaN/OOM なし。
- `uv run python scripts/run_ppl.py --models outputs/jul10-17b-best-e100`:
  **PPL 29.41**。既知ベスト e80=28.75 より +0.66 悪化したため不採用。
- pass 1 は全 19 窓で e80 より低い final reconstruction loss。pass 2 も窓 5 の
  一時的な悪化を後続で吸収し、最終窓は e100=267.08 対 e80=269.08 と低かった。
  それでも PPL は悪化したため、**窓 MSE の低下は LM 品質の改善を保証しない**。
- lr 1e-3/rank16 での既知結果(e80=29.38、e120=29.74)と合わせ、epoch 最適点は
  80 付近。lr 2e-3/rank32 でも 100 への延長は過剰最適化になる。
- コード変更なし(CLI のみ)。e80 を維持し、次は e80 固定で clip 0.7 を試す。

### 2026-07-11 E2: grad clip 0.7 - 失敗 (29.08)

- 構成: 1.7B、512 samples、seq_len 2048、clip 0.7、LoRA rank 32、
  epochs 80、LM lr 2e-3。既知ベストから clip 値だけを変更。
- 量子化時間: **311.2 分**。NaN/OOM なし。
- `uv run python scripts/run_ppl.py --models outputs/jul11-17b-clip07-r32-e80-lr2`:
  **PPL 29.08**。clip 0.5 の 28.75 より +0.33 悪化したため不採用。
- pass 1 は全 19 窓で clip 0.5 より低い final reconstruction loss。pass 2 は
  窓 2〜3 だけ悪化し、他は概ね 0.1〜3%改善。最終窓も 268.48 対 269.08 と低い。
  E1 と同様、**再構成 MSE の小幅改善と PPL は逆方向**になった。
- lr 1e-3 での既知 clip sweep(0.25=31.97、0.5=31.86、1.0=32.35)に加え、
  lr 2e-3 でも 0.7 が悪化したため、clip 0.5 を維持する。
- 次は総更新量を再検討する。r32 の既知点(lr 1e-3/e80=29.17、
  lr 2e-3/e80=28.75、lr 2e-3/e100=29.41)を lr×epochs で二次近似すると
  頂点は lr 2e-3 で約 67 epochs。e70 を直接検証する。
