# PPL 研究ノート (autoresearch/jul14, binary W1)

目標: Qwen/Qwen3-1.7B の**二値量子化**(`--quant-mode binary`、{-1,+1}×グループスケール)モデルで
`scripts/run_ppl.py` の PPL を最小化する。

## 現在の最小ppl値

**70.12** — `outputs/jul14-17b-bin-r32-e80-lr2`
(`--lora-rank 32 --epochs 80 --lr 2e-3`、clip 0.5・batch 3・seed 0)

旧: 87.15 (ベースライン: デフォルト設定 512×60、batch 3、lr 1e-3、rank 4)

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
- 予想時間: ternary batch1 の 564.3 分 × 0.91 ≈ **515 分(~8.6h)**。
