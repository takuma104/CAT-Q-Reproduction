# PPL 研究ノート (autoresearch/jul14, binary W1)

目標: Qwen/Qwen3-1.7B の**二値量子化**(`--quant-mode binary`、{-1,+1}×グループスケール)モデルで
`scripts/run_ppl.py` の PPL を最小化する。

## 現在の最小ppl値

**52.21** — `outputs/jul14-17b-bin-b1-r32-e80-lr2`
(`--batch-size 1 --lora-rank 32 --epochs 80 --lr 2e-3`、clip 0.5・seed 0)

旧: 70.12 (batch 3) ← 87.15 (ベースライン: デフォルト設定、batch 3、lr 1e-3、rank 4)

## 参考: ternary 実験 (autoresearch/jul6, jul10) からの引き継ぎ知見

ternary では baseline 38.81 → **27.41** まで改善。効いた順に:

1. **勾配クリッピング**(clip 0.5): 最大の改善(38.81→31.86)+シード分散消滅。
   → **binary の baseline には既にデフォルトで入っている**(grad_clip=0.5)
2. **LoRA rank 32**(31.86→29.78@e60、デプロイコストゼロ)
3. **epochs 80**(rank16 で 30.02→29.38、頂点は 80 付近。e70/e100/e120 は悪化)
4. **LM lr 2e-3**(29.17→28.75。4e-3 は悪化、lora_lr は 5e-4 固定が正解)
5. **batch size 2→1**(28.75→27.75→27.41、単調だが時間 +20〜80%)

既知の失敗: rank 64、lr 4e-3、lora_lr 1e-3、epochs 70/100/120、seq_len 1024、
3-pass、gamma 0.6、サリエンス順量子化、重み付き損失(素の MSE 以外全滅)、
hard-STE polish、batch 4、pass 別 batch。

**転移に関する注意**: 0.6B や低予算 1.7B からフル 1.7B への順位転移は不安定
(jul6 で 2 度反転)。最終判断はフル 1.7B(512×2048)で行う。
また ternary→binary の転移も自明ではないので要検証。

## 記録

### 2026-07-14 ベースライン確立

- ブランチ `autoresearch/jul14` を `binary_ppl_research` から作成。
- `uv run pytest`: **36 件全 PASS**。
- 既存の binary 量子化済みモデルを評価(現行コード製、program.md の慣例どおり初回量子化はスキップ):
  - `outputs/qwen3-1.7b-catq-w1`: **PPL 87.15**(README 報告値と一致)
  - 参考: 0.6B binary は 111.6、1.7B ternary v3 は 38.81
- GPU: RTX 5090 32GB、空き。量子化時間の目安: binary 1.7B デフォルト設定 ≈ 212 分
  (ternary 比 ~10% 高速)
- 校正キャッシュ(seed0、512×2048)は `qwen3-1.7b-catq-w1/calibration_ids.pt` を
  新ランに事前配置して再利用する(HF streaming 障害の回避、jul6 の慣例)

### 実験計画

ternary で実証済みのレシピをまず binary に転移する(B1)。効けば batch 軸(B2)、
その後 binary 固有の軸(s₀ シャープネス、μ 初期化、α の L1 最適化など)を探索する。

- B1: `--lora-rank 32 --epochs 80 --lr 2e-3`(clip 0.5・batch 3 はデフォルト)
  — ternary では 38.81→28.75 相当の複合レシピ。予想時間 ~4.8h
- B2: B1 + `--batch-size 1`(ternary では追加 −1.34、時間 +80%)

### 2026-07-14 B1 起動 + binary 固有の改善候補(コード精査より)

- B1 実行中(00:04 開始、~04:45 完了見込み)。校正キャッシュ再利用を確認、
  pass 1 window 1 の loss は正常(0.000181)、GPU 99%
- module.py / transition.py 精査から得た binary 固有の候補:
  1. **μ の初期決定境界**: 現在 μ init = μ₀(グループ平均)。binary の再構成は
     α·sign(W̃−μ) で μ は加算復元されないため、対称レベル {−α,+α} への
     最近傍割当の最適境界は **0**(グループ平均ではない)。binary では
     μ = δ_μ·α₀(μ₀ 項なし)として境界 0 初期化にする案。ソフト段階の開始点も
     ずれなく W̃ に一致する。コード変更小
  2. **s₀ スイープ**(CLI のみ): binary の遷移関数は tanh 1 枚(ternary は 2 枚の重ね)
     で勾配ジオメトリが異なる。ternary では s0=50 はノイズ帯だったが binary は未知
  3. **γ スイープ**(CLI のみ): binary の hard 段階 STE 代理勾配 tanh(30ŵ) は
     0 近傍以外飽和 → soft 段階を延ばす γ 0.9 は ternary 未探索の方向
- α₀ init は mean|W−μ| で、μ=0 なら binary の MSE 最適スケール(XNOR-Net の α*=E|w|)
  と一致 — δ_α init 1 は既に良い初期値

### 2026-07-14 B1: ternary ベストレシピ転移 → ★成功、新記録 70.12(−17.03)

- 構成: 1.7B binary、512×2048、batch 3、clip 0.5、LoRA rank 32、epochs 80、
  LM lr 2e-3(ベースラインからの変更は rank 4→32、e60→80、lr 1e-3→2e-3)。
- 量子化時間: **283.6 分**(ternary 同構成 310.8 分より 8.7% 高速)。NaN/OOM なし。
- `run_ppl.py`: **PPL 70.12**。ベースライン 87.15 から **−17.03(−19.5%)**。
- **ternary で確立した「クリップ下の容量+更新量レシピ」は binary にそのまま転移する**。
  改善率は ternary の同レシピ(38.81→28.75、−26%)よりやや小さいが同方向。
- コード変更なし(CLI のみ)。次: B2(batch 1、ternary では単調改善で最終 −1.34)。

### 2026-07-14 B2 起動: batch 1

- B1 から batch size のみ変更(3→1)。ternary の batch 曲線は 4=30.44、3=28.75、
  2=27.75、1=27.41 と単調。binary でも同傾向を仮定し中間の batch 2 は飛ばす。
- 予想時間: ternary batch1 の 564.3 分 × 0.91 ≈ **515 分(~8.6h)**。04:50 開始。

### 2026-07-14 B1 vs baseline の catq_log 分析

- B1 は pass 2 のほぼ全窓で final loss がベースラインを下回る(深層で −20〜25%)。
  PPL 改善(−19.5%)と整合。
- 両者とも PCSW 末尾 3 窓(深層)は窓内改善が final/initial=0.92〜0.98 でほぼ学習が
  進まない(ternary でも同傾向だった)。深層の absolute loss は ~480 で最大。
- weight 空間の recon_error は B1 の方が 2.2 倍大きい(2.81e6 vs 1.26e6)が PPL は
  大幅に良い — LoRA r32 が W̃ を W から遠ざけるため。**weight 再構成誤差と品質は
  無関係**(ternary の「窓 MSE ≠ PPL」知見の binary 版)。
- STE 飽和の検討: binary の hard 段階代理勾配 tanh(30ŵ) は |ŵ|<~0.1 のみ勾配を持つが、
  これは「符号が不確かな境界重みだけ動かす」正しい機構。ternary も ±Δ 近傍のみで
  実は対称な構造 → s_backward 分離の優先度を下げた。
- binary が ternary に負ける主因は 0 レベル欠落(|ŵ| 小の重みの誤差が ~α に固定)。
  グループ内で α がバランスを取るしかない → **補償容量(LoRA rank)の最適点が
  ternary(r32)より右にシフトしている可能性** → r64 を有望候補に昇格。

### 2026-07-14 B2: batch 1 → ★大成功、新記録 52.21(−17.91)

- 構成: B1 から batch size のみ変更(3→1)。量子化時間: **458.8 分**。NaN/OOM なし。
- `run_ppl.py`: **PPL 52.21**。旧最小 70.12 から **−17.91(−25.5%)**。
- **binary では batch 1 の効果が ternary(−1.34)の 13 倍**。binary は ternary より
  はるかに最適化制約が強い(optimization-limited)ことを示す。
- 解釈仮説: batch 1 は (a) 3 倍のステップ数、(b) batch-LR スケーリングにより 1/3 の
  実効 LR、(c) 大きい勾配ノイズ、の複合。binary は全重みの符号が決定境界に関わるため
  細かい探索の恩恵が大きい(ternary はコード変化が ±Δ 境界近傍のみ)。
- 含意: batch-1 レジームでは lr / epochs の最適点も移動している可能性 → 再スイープ候補。
- コード変更なし(CLI のみ)。

### 次実験キュー(B2 完了後、優先順)

1. B3: `--lora-rank 64`(binary の量子化誤差は ternary の ~2.2 倍 → 補償需要が大きい。
   ternary では r64 は +2.4 悪化だが binary は最適点シフトの仮説。CLI のみ)→ 12:31 起動
2. B4: `--quant-rates 1.0`(単一フルパス、同計算量で窓あたりエポック 2 倍。
   シンプル化の勝利になり得る。ternary 未検証。CLI のみ)
3. B5: μ 決定境界 0 初期化(コード変更小。ternary P8 の「init は学習で吸収」前例
   から期待値は低め)
4. s0 / γ スイープ(ternary で全滅した軸。優先度低)

### 2026-07-14 B2 vs B1 の catq_log 分析: batch 1 の勝因は浅層

- batch1/batch3 の final loss 比: pass1 窓 3-5 で **0.36〜0.61**、pass2 窓 1-4 で
  **0.68〜0.76**、深層(窓 10+)は 0.91〜0.95。**利得は浅層に強く集中**。
- jul10 E6 の「浅層の補正品質が PPL を強く支配」の binary 版を確認。浅層の改善が
  下流に複利で効く。
- 深層窓は依然ほぼ学習が進まず(abs loss ~440)、binary の irreducible error が
  深層に残る。
- batch-1 レジームの含意: 実効 LR が 1/3 になっても勝った → 「小さいステップを
  多く積む」方向がまだ未飽和の可能性 → **e100@b1 を B4 候補に昇格**(ternary の
  e100 失敗は b3/実効 LR 6e-3 での過剰最適化。b1/実効 2e-3 は別レジーム)。
