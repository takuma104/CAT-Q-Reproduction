# PPL 研究ノート (autoresearch/jul10)

目標: Qwen/Qwen3-1.7B の三値量子化モデルで `scripts/run_ppl.py` の PPL を最小化する。

## 現在の最小ppl値

**27.75** - `outputs/jul11-17b-b2-clip05-r32-e80-lr2`
(`--batch-size 2 --grad-clip 0.5 --lora-rank 32 --epochs 80 --lr 2e-3`、seed 0)

旧: 28.75 (batch 3) ← 29.17 (lr 1e-3) ← 29.38 (rank 16) ← 38.81 (baseline)

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

### 2026-07-11 E3: epochs 70 - 失敗 (29.57)

- 構成: 1.7B、512 samples、seq_len 2048、clip 0.5、LoRA rank 32、
  epochs 70、LM lr 2e-3。既知ベストから epochs だけを変更。
- 量子化時間: **273.3 分**。NaN/OOM なし。
- `uv run python scripts/run_ppl.py --models outputs/jul11-17b-clip05-r32-e70-lr2`:
  **PPL 29.57**。e80 の 28.75 より +0.82 悪化したため不採用。
- pass 1 は多くの窓で e80 より約 1%高い final loss。pass 2 では窓 5 が約 40%、
  中層が約 3〜8%、最終窓も約 0.7%高く、hard 段階の適応不足が明確だった。
- lr×epochs の二次近似は epochs が ST の `t` 進行も決めることを無視していた。
  e70 は総更新量だけでなく soft/hard 各段階の step 数を同時に削るため外挿不能。
- e70=29.57、e80=28.75、e100=29.41 より epochs 80 を維持する。
  次は epoch 数を固定し、hard 段階の optimizer schedule を検討する。

### E3 後の scheduler 再確認

- CAT-Q 論文 Appendix B は AdamW + linear decay to zero を明記。
- SliderQuant 参照実装も各 window/round で Hugging Face の linear scheduler を作り直し、
  `max_train_steps = epochs * steps_per_epoch` で zero まで減衰する。現行実装と一致。
- lr floor は hard 段階を助ける可能性がある一方、e100 で確認した過剰最適化を強める。
  仕様から外れる割に根拠が弱いため、先に既存 CLI で batch 軸を調べる。
- batch size 4 は 512 samples を割り切り、勾配分散と端数 batch をなくす。
  batch LR scaling により epoch あたりの総更新量は概ね維持される。

### 2026-07-11 E4: batch size 4 - 失敗 (30.44)

- 構成: 1.7B、512 samples、seq_len 2048、batch 4、clip 0.5、LoRA rank 32、
  epochs 80、LM lr 2e-3。既知ベストから batch size だけを変更。
- 量子化時間: **285.1 分**。batch 3 の 310.8 分より 8.3%高速。NaN/OOM なし。
- `uv run python scripts/run_ppl.py --models outputs/jul11-17b-b4-clip05-r32-e80-lr2`:
  **PPL 30.44**。batch 3 の 28.75 より +1.69 悪化したため不採用。
- pass 1 は全窓で batch 3 より高い final loss。pass 2 は窓 2〜4 で約 23〜32%、
  深層でも約 2〜3%高く、最終窓は 276.72 対 269.08 だった。
- linear batch LR scaling は epoch あたりの一次近似の更新量を保つが、optimizer step
  数の減少(171→128 steps/epoch)を補えない。batch 3 を維持する。
- 逆方向の batch 2 は 256 steps/epoch と小さい離散更新になり、epochs/ST 軌道を
  変えずに最適化精度を上げられる可能性があるため次に検証する。

### 2026-07-11 E5: batch size 2 - 成功、新記録 27.75

- 構成: 1.7B、512 samples、seq_len 2048、batch 2、clip 0.5、LoRA rank 32、
  epochs 80、LM lr 2e-3。既知ベストから batch size だけを変更。
- 量子化時間: **373.7 分**。batch 3 の 310.8 分より約 20%低速。NaN/OOM なし。
- `uv run python scripts/run_ppl.py --models outputs/jul11-17b-b2-clip05-r32-e80-lr2`:
  **PPL 27.75**。旧最小 28.75 を **1.00 改善**。
- pass 1 は窓 3 で大きな一時スパイクから回復して batch 3 比約 48%改善し、
  以降も概ね 0〜1%低い final loss。pass 2 は全 19 窓で batch 3 より低く、
  浅層で約 6〜12%、最終窓でも約 2.0%改善(263.59 対 269.08)。
- batch LR scaling で一次近似の総更新量は同じでも、小さい step を多く積む方が
  hard/full 量子化の最適化精度を上げる。batch 曲線は 4=30.44、3=28.75、
  2=**27.75** と単調。次は batch 1 を検証する。
