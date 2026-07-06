# CAT-Q 再現実装

[CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs](docs/papers/cat-q-paper.md) (arXiv 2606.26650) の再現実装です。事前学習済み LLM の重みを PTQ で三値 ({-1, 0, +1} × グループスケール、1.58-bit) に量子化します。

CAT-Q が依拠する [SliderQuant](docs/papers/slider-quant-paper.md) のフレームワーク(スライディングウィンドウ・二重ストリーム学習・LoRA 補償・イントラレイヤースライディング)は、論文と `docs/reference_impl/` の公式実装に準拠しています。

## 現在の再現状況

Qwen3-1.7B W1.58A16、zero-shot 5 タスク(paper プロトコル: PIQA/ARC/HS=acc_norm, WG=acc):

| Model | PIQA | ARC-e | ARC-c | HS | WG | Avg | C4-val PPL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B FP(本評価) | 72.63 | 70.08 | 43.17 | 60.32 | 61.09 | **61.46** | - |
| + CAT-Q(論文値) | 68.40 | 55.88 | 28.16 | 47.77 | 54.83 | **51.01** | - |
| + CAT-Q(本実装 v3) | 61.97 | 41.33 | 24.83 | 36.92 | 52.01 | **43.41** | 38.8 |

FP 評価は論文 Table 1 と一致(61.46 vs 61.42)。CAT-Q は **部分的再現(ギャップ −7.6pt)** で、経緯・診断・残ギャップの仮説は [docs/reproduction-report.md](docs/reproduction-report.md) を参照してください。量子化時間は RTX 5090 ×1 で 0.6B 約 104 分、1.7B 約 234 分です。

## セットアップ

Python 3.13 / uv / NVIDIA GPU(bf16 対応、0.6B〜1.7B なら 24GB 以上推奨)。

```bash
uv sync
uv run pytest   # 単体テスト 22 件
```

モデルとキャリブレーションデータ (allenai/c4) は初回実行時に Hugging Face Hub から自動ダウンロードされます。

## 使い方

### 量子化

```bash
# 論文設定 (512 サンプル × 2048 トークン、60 エポック) で Qwen3-0.6B を量子化
uv run python scripts/run_quantize.py --output-dir outputs/qwen3-0.6b-catq-v3

# Qwen3-1.7B (断片化対策の環境変数を推奨)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python scripts/run_quantize.py \
  --model-name Qwen/Qwen3-1.7B --output-dir outputs/qwen3-1.7b-catq-v3

# スモークテスト (数分)
uv run python scripts/run_quantize.py \
  --num-calib-samples 32 --epochs 4 --output-dir outputs/smoke
```

出力ディレクトリには fake-quant 済みモデル(`save_pretrained` 形式、そのまま `transformers` でロード可能)、トークナイザ、キャリブレーションデータのキャッシュ (`calibration_ids.pt`)、窓ごとの最適化ログ (`catq_log.json`) が保存されます。

主なオプション(デフォルトは論文 Table A + SliderQuant W2A16 リファレンス設定):

| オプション | デフォルト | 意味 |
| --- | --- | --- |
| `--num-calib-samples` / `--seq-len` | 512 / 2048 | C4 キャリブレーションのサンプル数・長さ |
| `--epochs` | 60 | 総エポック(quant_rate パス数で等分) |
| `--group-size` | 128 | 三値化グループサイズ |
| `--s0` / `--gamma` | 30 / 0.8 | ST の最終シャープネス・ソフト段階比率 |
| `--window-size` / `--stride` / `--fill-window-size` | 4 / 2 / 4 | スライディングウィンドウ設定 |
| `--quant-rates` | 0.5 1.0 | イントラレイヤースライディングのパス |
| `--lora-rank` / `--lora-lr` | 4 / 5e-4 | LoRA 補償 |
| `--cs` | off | チャネルスケーリング(三値では有害と実証済み、通常オフのまま) |

### 評価

```bash
# zero-shot 5 タスク (lm-evaluation-harness)
uv run python scripts/run_eval.py --model outputs/qwen3-0.6b-catq-v3 \
  --label "CAT-Q 0.6B" --output outputs/eval.json

# C4 validation perplexity (MC 精度より鋭敏な品質診断)
uv run python scripts/run_ppl.py --models Qwen/Qwen3-0.6B outputs/qwen3-0.6b-catq-v3
```

`run_eval.py` は acc と acc_norm の両方を JSON に保存します。論文と比較する際は paper プロトコル(PIQA/ARC/HS=acc_norm、WG=acc)で集計してください。

## ファイル構成

```
src/catq/
  config.py       # CATQConfig: 全ハイパーパラメータ (論文 Table A + リファレンス W2A16 設定)
  transition.py   # 遷移関数 f(W;s,Δ) (Eq.5)、ハード三値化 Q (Eq.2)、STE
  ternary.py      # グループ分割 (g=128) と μ₀/α₀ 統計
  module.py       # CATQLinear: LM の学習因子 δ_μ/δ_α/δ_Δ + ST + LoRA + 部分量子化
  calibration.py  # C4 からのキャリブレーションデータ生成 (シード固定・キャッシュ)
  slider.py       # スライディングウィンドウ最適化 (二重ストリーム・2パス・一括焼き込み)
  quantize.py     # エントリポイント: ロード → 量子化 → 保存
scripts/
  run_quantize.py # 量子化 CLI
  run_eval.py     # lm-eval による zero-shot 評価
  run_ppl.py      # C4-val PPL 診断
tests/            # 単体テスト (遷移関数の性質、グループ化、モジュール挙動、E2E スモーク)
docs/
  papers/         # CAT-Q / SliderQuant / OmniQuant 論文 (markdown)
  reference_impl/ # SliderQuant / OmniQuant 公式実装 (参照用)
  plans/          # 実装プラン
  reproduction-report.md  # 再現実験レポート (結果・診断・アブレーション・残ギャップ仮説)
outputs/          # 量子化モデル・評価結果 (git 管理外)
```

## アルゴリズム概要

1 デコーダ層内の対象 linear(q/k/v/o/gate/up/down)を `CATQLinear` に置き換え、窓単位の出力再構成でグループごとの学習因子を最適化します。

- **LM (Learnable Modulation)**: グループごとに Ŵ=(W−μ)/α と変換(μ=μ₀+δ_μ·α₀、α=δ_α·α₀、Δ=δ_Δ·0.5)。δ の 3 因子のみ学習し、再構成は W≈αT(μ なし)。
- **ST (Softened Ternarization)**: 時刻 t に応じ恒等写像 → 微分可能三値化 f(Ŵ; (t/γ)s₀, Δ) → ハード三値化(STE)へ遷移。t は**層の参加期間全体**(全パス×全参加窓)で単調に進み、最終窓で t=1。
- **SliderQuant フレームワーク**: 拡張窓(浅層 4)→ 固定窓 {4, stride 2} → 収縮窓(深層 4)のスケジュール。FP ストリームと量子化ストリームを別々に前進させ、「量子化ストリームを入力された student が FP ストリームの教師出力を再現する」損失で蓄積誤差を補正。quant_rate {0.5→1.0} の 2 パスで全スケジュールを回し、最後に全層を一括でハード三値化して焼き込み。LoRA (r=4) は量子化前の重みに吸収される補償項。

## 論文から読み取れず解釈・実証した主な点

詳細は [docs/reproduction-report.md](docs/reproduction-report.md):

- ST の t を窓ごとに再アニールすると解が破壊される(PPL 爆発)→ 層の生涯で単調に進める
- チャネルスケーリングは三値量子化で有害(アブレーションで特定。リファレンスの W2A16 も `lora_only` で不使用)
- ターゲットは Eq.7 の字面(同一 X)ではなく二重ストリーム(公式実装準拠)。これが PPL 3 倍改善の主因
- 損失は素の MSE が最良(token-RMS 正規化は逆効果)
