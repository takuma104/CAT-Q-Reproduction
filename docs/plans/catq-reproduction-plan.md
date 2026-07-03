# CAT-Q 再現実装プラン (Qwen3-0.6B / RTX 5090 x1)

## Context

[docs/paper.md](docs/paper.md) の CAT-Q (arXiv 2606.26650, "Cost-efficient and Accurate Ternary Quantization for LLMs") の再現実装を行う。CAT-Q は PTQ ベースの三値化(1.58-bit)手法で、2つのコア要素からなる:

1. **Learnable Modulation (LM)**: グループ単位(g=128)で学習可能な3因子 δ_μ, δ_α, δ_Δ により重み分布としきい値を変調
2. **Softened Ternarization (ST)**: 滑らかな遷移関数 f(W;s,Δ) = [tanh(s(W−Δ)) + tanh(s(W+Δ))] / (2·tanh(s)) により、恒等写像→微分可能三値化→ハード三値化の2段階リレーで安定収束

これらを **スライディングレイヤー出力再構成**(Eq.7: argmin ||F(W,X) − F(A·T,X)||²)に組み込む。

- 対象モデル: **Qwen3-0.6B**(パイプライン確立用。論文最小は1.7Bなので、動作確認後 1.7B で論文 Table 1 と比較可能)
- 環境: RTX 5090 (32GB) x1、Python 3.13 / uv / torch 2.12 / transformers 5.12
- 公式コード (github.com/IntelChina-AI/BitTern) は参照せず、論文記載のみから実装

### ユーザー確認済みの裁量事項
- スライディングウィンドウは **l=2 デフォルト、1〜4 で設定可能**(SliderQuant のデフォルト値は論文から読み取れないため)
- 評価は **lm-evaluation-harness** を使用
- 最初のマイルストーンは **W1.58A16 のみ**(activation 量子化は後回し)

## 論文から読み取ったアルゴリズム仕様

### ハイパーパラメータ (Appendix B.1, Table A)
| 項目 | 値 |
| --- | --- |
| キャリブレーションセット | C4 (allenai/c4, en) からランダム 512 サンプル × 2048 トークン |
| グループサイズ g | 128 |
| Δ₀ | 0.5 |
| s₀ | 30 |
| γ | 0.8 |
| バッチサイズ | 3 |
| Optimizer | AdamW |
| Epochs m | 60 |
| LM の学習率 | 1e-3、線形減衰で 0 へ |

### LM (Eq.3, 4)
グループごと(重みを flatten して 128 要素ずつ非重複分割、Appendix B.2):
- μ₀ = mean(W)、α₀ = mean(|W − μ₀|) — W は凍結なので**初期化時に一度だけ計算**
- Ŵ = (W − μ)/α、ここで μ = μ₀ + δ_μ·α₀、α = δ_α·α₀
- Δ = δ_Δ·Δ₀ (Δ₀ = 0.5)
- 制約: −1 < δ_μ < 1、δ_α > 0、δ_Δ > 0 (init: δ_μ=0, δ_α=1, δ_Δ=1)
- 再構成は **W ≈ α·T(μ なし)** — disentangled 戦略(Table 10 で優位性確認済み、ハードウェアフレンドリー性維持)

### ST (Eq.5, 6)
正規化時刻 t(現在エポック / 総エポック m)に応じて:
- t = 0: T = W(恒等)
- 0 < t ≤ γ: T = f(Ŵ; (t/γ)·s₀, δ_Δ·Δ₀) — 微分可能三値化(シャープネスをエポックごとに増加)
- γ < t ≤ 1: T = Q(Ŵ; δ_Δ·Δ₀) — ハード三値化。勾配は「第1段階最後の反復の勾配を利用」→ **backward は f(·; s₀, Δ) を代理勾配とする STE 風実装**と解釈

### スライディングレイヤー最適化 (§2.4)
- デコーダ層の窓 [i, i+l−1] 単位で出力再構成: 窓の入力 X は**量子化済み上流の出力**(逐次伝播)、ターゲットは同じ X に対する FP 重みでの窓出力(Eq.7 は両者に同一 X を使用)
- 窓ごとに δ_μ, δ_α, δ_Δ を AdamW で 60 エポック最適化 → 窓先頭層を確定(ハード三値化して重み焼き込み)→ 窓を 1 層スライド
- 量子化対象: デコーダ層内の全 nn.Linear(q/k/v/o_proj, gate/up/down_proj)。embedding・lm_head・norm は FP のまま(BitNet 系と同じ慣行)

## 実装ステップ

### 1. プロジェクト準備
- `uv add datasets lm-eval accelerate` (+ dev: pytest)
- pyproject.toml に `[tool.pytest.ini_options]` 等を整備

### 2. コア実装 `src/catq/`
- `config.py` — `CATQConfig` dataclass(上記ハイパラ全部 + window_size=2, モデル名, 出力先。全て型ヒント付き)
- `transition.py` — 遷移関数 f(W;s,Δ)(Eq.5)とハード三値化 Q(W;Δ)(Eq.2)、STE 代理勾配(torch.autograd.Function または detach トリック)
- `ternary.py` — グループ分割ユーティリティ(flatten→[n_groups,128]、端数はグループ縮小で対応)、μ₀/α₀ 統計、α·T の焼き込み(fake-quant 重み生成)
- `module.py` — `CATQLinear`: nn.Linear をラップし、凍結 W + グループ統計バッファ + 学習パラメータ(raw パラメータに tanh/softplus をかけて制約を満たす)を保持。forward で時刻 t に応じた Eq.6 の実効重み α·T を用いて F.linear
- `calibration.py` — C4 streaming から 512×2048 トークンのキャリブレーションデータ生成(シード固定)、トークナイズ、キャッシュ
- `slider.py` — スライディングウィンドウループ:
  1. 埋め込み層まで forward して層0への入力(hidden states, position 情報等)を全サンプル分キャッシュ(0.6B なら 512×2048×1024×2B ≈ 2GB、GPU に載る。載らない場合は CPU オフロード)
  2. 各窓: FP ターゲットを no_grad で計算 → CATQLinear に差し替え → 60 エポック最適化(L2 損失、AdamW lr=1e-3 線形減衰、バッチ3)→ 先頭層をハード三値化で確定・焼き込み → キャッシュを確定済み層の出力で更新して次の窓へ
- `quantize.py` — エントリポイント: モデルロード → パイプライン実行 → fake-quant 済みモデルを save_pretrained + 三値化統計(ゼロ率など)を JSON 出力

### 3. スクリプト `scripts/`
- `run_quantize.py` — CLI(argparse: モデル名、サンプル数、エポック数、ウィンドウサイズ等を上書き可能)
- `run_eval.py` — lm_eval Python API で PIQA / ARC-e / ARC-c / HellaSwag / WinoGrande の zero-shot acc を測定、FP ベースラインと量子化モデルの比較表を出力

### 4. テスト `tests/`
- `test_transition.py` — f の性質検証: 原点対称、s→小 で恒等近似、s=30 で出力が {−1,0,1} 近傍、微分可能性、Q との一致(s→∞ 極限)
- `test_ternary.py` — グループ分割/統計/焼き込みの正当性(往復チェック、端数グループ)
- `test_module.py` — CATQLinear: t=0 で FP forward と一致、ハード段階で重みが {−α,0,α} のみ、勾配が学習パラメータに流れること
- `test_slider.py` — 2〜3層のダミー Transformer で E2E スモーク(損失が減少すること)

### 5. 実行と検証
1. `pytest` 全通過 → git commit
2. **スモークラン**: Qwen3-0.6B、32 サンプル × 5 エポックで全パイプライン疎通確認(数分)
3. **本番ラン**: Qwen3-0.6B、512 サンプル × 60 エポック(論文設定)。窓ごとの再構成損失ログを記録
4. lm-eval で FP (W16A16) と CAT-Q (W1.58A16) を評価、比較表を作成
5. 結果を docs/ にレポートとしてまとめ、git commit

### 期待される結果の目安
Qwen3-0.6B は論文に無いが、Qwen3-1.7B の傾向(W16A16 avg 61.42 → W1.58A16 avg 51.01、約10pt低下)から、0.6B ではそれ以上の相対劣化が予想される。パイプライン検証後、必要なら 1.7B(bf16 で 3.4GB、32GB に余裕で収まる)で論文 Table 1 との直接比較を行う。

## 検証方法
- 単体テスト: `uv run pytest`
- 数値検証: 遷移関数の形状を Appendix C の記述(s=4.95 で出力が [−1,1] に収まる等)と突き合わせ
- E2E: スモークラン → 本番ラン → lm-eval のスコアが「FP > CAT-Q だが崩壊しない(ランダム率 ~40% を大きく上回る)」ことを確認。ablation(Table 5)のベースライン(LM/ST なし = avg 40.16 @4B)より良いことが健全性の目安

## リスク / 論文から読み取れず解釈した点(実装時にコメントで明記)
- SliderQuant のウィンドウ詳細(サイズ・ストライド・ターゲットの定義)→ l=2 / ストライド1 / 同一 X でターゲット計算、と解釈
- ハード段階の勾配伝播の正確な仕様 → f(·;s₀,Δ) を代理勾配とする STE と解釈
- 「60 エポック」が窓ごとか全体か → 窓ごとと解釈(各窓の最適化が独立なため)
- lm-eval と transformers 5.x の互換性は導入時に確認し、問題があれば HFLM ラッパーを自作
