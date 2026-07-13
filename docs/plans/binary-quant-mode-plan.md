# 二値化 (W1, {-1,+1}) モード追加プラン

## Context

本リポジトリは CAT-Q(三値 {-1,0,+1} × グループスケール、1.58-bit PTQ)の再現実装。ここに **二値化 ({-1,+1} × グループスケール、W1) モード** を追加する。三値の Δ→0 極限として同じ LM+ST+SliderQuant フレームワークに乗せることで、既存パイプライン(スライディングウィンドウ・二重ストリーム・LoRA 補償)を無改修で流用できる。論文外の新規実験である。

**スコープ**: 実装 + 単体テスト + スモーク量子化での動作確認まで。0.6B フル実験・評価・README への結果追記は後続タスク。

## 数式定義

三値の Δ→0 極限として定義(既存 Eq. 2/5/6 と平行):

- **Soft binarize**: `f_bin(W; s) = tanh(s·W) / tanh(s)` — `smooth_transition(w, s, 0)` と数学的に同値。s→0 で恒等、s→∞ で sign。既存 ST スケジュール(t≤γ で s=(t/γ)s₀)をそのまま使用
- **Hard binarize**: `Q_bin(W) = +1 if W ≥ 0 else −1` — `torch.sign(0)=0` のため `torch.where(w >= 0, 1, -1)` でタイブレークを +1 に固定(`hard_ternarize(w, 0)` は w=0 で 0 を返すので流用不可。専用関数を追加する決定的理由)
- **STE**: forward = Q_bin、backward = ∂f_bin(·; s₀)(既存 `hard_ternarize_ste` と同じ `soft + (hard − soft).detach()` パターン)
- **LM 因子**: δ_μ は残す(sign(Ŵ)=sign(W̃−μ) なので μ が sign の決定境界として機能)。δ_α も残す(振幅 + soft 段階のスロープ)。**δ_Δ は作らない**(二値に閾値は存在しない)。再構成は三値と同じく α·B(μ は足し戻さない)→ 各グループ純粋な {−α, +α}

## 実装ステップ

### Step 0: git worktree 作成(CLAUDE.md 規約)

```bash
git worktree add ../CAT-Q-binary-mode -b feature/binary-quant-mode
```

以降すべて worktree 内で作業し、テスト通過ごとに commit。

### Step 1: `src/catq/transition.py` — 二値化プリミティブ 3 関数追加

既存 3 関数の直後に、同じ docstring スタイル(論文 Eq 参照付き)で:

```python
def smooth_binarize(w: torch.Tensor, s: float) -> torch.Tensor:
    """f_bin(W; s) = tanh(s W) / tanh(s): the delta -> 0 limit of Eq. 5."""
    return torch.tanh(s * w) / torch.tanh(torch.as_tensor(s, dtype=w.dtype, device=w.device))

def hard_binarize(w: torch.Tensor) -> torch.Tensor:
    """Q_bin(W) = sign(W) with the tie at 0 broken toward +1."""
    return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))

def hard_binarize_ste(w: torch.Tensor, s0: float) -> torch.Tensor:
    soft = smooth_binarize(w, s0)
    return soft + (hard_binarize(w) - soft).detach()
```

### Step 2: `tests/test_transition.py` — binary プリミティブのテスト

既存パターン(tests/test_transition.py)の binary 版:
- `smooth_binarize(w,s) ≈ smooth_transition(w,s,0)`(float64 で数学的同値性)
- 奇関数性 / s→小で恒等 / s=30 で sign 近似 / 微分可能性
- hard 値: `[-2,-0.5,0,0.5,2] → [-1,-1,1,1,1]`(**0→+1 タイブレーク固定**)
- STE forward=hard, backward=soft の勾配一致

→ `uv run pytest tests/test_transition.py` → commit

### Step 3: `src/catq/config.py` + `src/catq/module.py` — quant_mode 導入

**config.py** (Quantization セクション L25-28):
```python
# "ternary": {-1,0,+1} x group scale (paper). "binary": {-1,+1} x group
# scale (W1), the delta -> 0 limit; delta0 / rho_delta are unused.
quant_mode: str = "ternary"
```

**module.py** (`CATQLinear`):
1. `__init__` に `quant_mode: str = "ternary"` 追加、冒頭で `("ternary","binary")` 以外は `ValueError`
2. `rho_delta` を条件化 (L78): binary では `self.rho_delta: nn.Parameter | None = None`
3. `_modulated` (L147-155): 戻り値 `tuple[Tensor, Tensor, Tensor | None]`。`delta = self.delta_delta * self.delta0 if self.quant_mode == "ternary" else None`
4. `effective_weight` (L163-168): soft/hard 両分岐でモード dispatch(`smooth_binarize` / `hard_binarize_ste` vs 既存)。ローカル変数 `ternary` → `code` に改名
5. `finalize` (L194): `hard_binarize(w_hat)` / `hard_ternarize(w_hat, delta)` を dispatch。stats に `positive_fraction`(+1 の割合)を**両モード共通**で追加(binary の符号バランス診断。`zero_fraction` は残す — binary では常に 0.0 で事実として正しく、`quantize.py` の集計が無改修で動く)
6. モジュール/クラス docstring を両モード対応に更新

### Step 4: `tests/test_module.py` — binary 版テスト

`make_module(quant_mode="binary")`(既存 `make_module` は kwargs 透過)で:
- t=0 で FP 一致 / t=1.0 で各グループ unique ≤ 2 **かつ 0 を含まず** {−α,+α}
- soft 段階 (t=0.4) で有限・逆伝播可能(勾配チェック対象から rho_delta を除外)
- `module.rho_delta is None` かつ `named_parameters()` に現れない
- finalize が hard 段階と一致、`zero_fraction == 0.0`、`0 ≤ positive_fraction ≤ 1`
- 不正な quant_mode で `ValueError`
- 既存 ternary の finalize テストに `positive_fraction` 存在チェック追加

→ `uv run pytest tests/test_module.py tests/test_transition.py` → commit

### Step 5: `src/catq/slider.py` + `src/catq/quantize.py` — パイプライン接続

**slider.py**:
1. `_wrap_layer` (L150-158): `CATQLinear(..., quant_mode=cfg.quant_mode)` を追加
2. `_window_params` (**L179 — 唯一の破壊的接点**):
   ```python
   factors += [m.rho_mu, m.rho_alpha]
   if m.rho_delta is not None:
       factors.append(m.rho_delta)
   ```
   ※忘れると binary 実行時に optimizer 構築で即例外
3. docstring の "ternary" 表現を両モード対応に

**quantize.py** (L56-60): report に `mean_positive_fraction` を追加(`mean_zero_fraction` と同パターン)。`quant_mode` は `dataclasses.asdict(config)` 経由で自動ログされる。

### Step 6: `tests/test_slider.py` — E2E smoke の binary 対応

`test_sliding_window_end_to_end` を `@pytest.mark.parametrize("quant_mode", ["ternary", "binary"])` 化:
- ternary: 既存どおり unique ≤ 3
- binary: unique ≤ 2 かつ `(values != 0).all()`
- 共通: 有限 logits、loss 改善、layer_stats 件数

→ `uv run pytest`(全件、ternary の既存挙動が無変化であることを含む)→ commit

### Step 7: `scripts/run_quantize.py` + `src/catq/__init__.py`

- `--quant-mode` 引数追加: `choices=["ternary", "binary"], default=defaults.quant_mode`、`CATQConfig(..., quant_mode=args.quant_mode)`
- `__init__.py`: `smooth_binarize, hard_binarize` を export(既存の ternary 関数と対称に)

### Step 8: `README.md` 更新

- 冒頭: binary モードのサポートを追記(**再現状況テーブルは三値のもの、binary は論文外の新規実験**と明記)
- オプション表に `--quant-mode` 行追加
- 使い方に binary 実行例、アルゴリズム概要に W1 の数式 1 行、テスト件数更新
- このプランを `docs/plans/` へ移動(CLAUDE.md 規約)

## 検証

1. `uv run pytest` 全件パス(ternary 回帰なし)
2. binary スモーク:
   ```bash
   uv run python scripts/run_quantize.py --quant-mode binary \
     --num-calib-samples 32 --epochs 4 --output-dir outputs/smoke-binary
   ```
   確認: `catq_log.json` の `config.quant_mode == "binary"`、`mean_zero_fraction == 0.0`、`mean_positive_fraction ≈ 0.5`、window loss が有限で減少
3. 保存モデルの重み spot check(グループごと unique ≤ 2、0 なし)

## リスク・注意点

- s₀=30 / γ=0.8 / lr は三値チューニングのまま。binary の品質チューニングはスコープ外
- `delta0` は binary で未使用(config に残す、コメントで明記)
- 0.6B フル実験(~104 分)+ 評価 + README 結果追記は後続タスク

## 変更ファイル一覧

| ファイル | 変更 |
| --- | --- |
| `src/catq/transition.py` | `smooth_binarize` / `hard_binarize` / `hard_binarize_ste` 追加 |
| `src/catq/config.py` | `quant_mode: str = "ternary"` 追加 |
| `src/catq/module.py` | quant_mode dispatch、rho_delta 条件化、`positive_fraction` 統計 |
| `src/catq/slider.py` | `_wrap_layer` 伝播、`_window_params` の None フィルタ (L179) |
| `src/catq/quantize.py` | `mean_positive_fraction` 集計 |
| `scripts/run_quantize.py` | `--quant-mode` CLI |
| `src/catq/__init__.py` | binary 関数 export |
| `tests/test_transition.py` / `test_module.py` / `test_slider.py` | binary 版テスト追加 |
| `README.md` | binary モード記載 |
