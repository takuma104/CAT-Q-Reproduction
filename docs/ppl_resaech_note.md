# PPL 研究ノート (autoresearch/jul14, binary W1)

目標: Qwen/Qwen3-1.7B の**二値量子化**(`--quant-mode binary`、{-1,+1}×グループスケール)モデルで
`scripts/run_ppl.py` の PPL を最小化する。

## 現在の最小ppl値

**87.15** — `outputs/qwen3-1.7b-catq-w1`(ベースライン: デフォルト設定 512×60、
batch 3、lr 1e-3、LoRA rank 4、grad_clip 0.5、seed 0)

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
