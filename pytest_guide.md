# pytest 実践ガイド（機械学習エンジニア向け）

Python には慣れているが pytest はこれから、という人向けの実践ガイド。
pytest の使い方そのものは一般的な題材で説明し、**具体的な実装例は機械学習（ML）の場面**を使う。

前提パッケージ（この `pyproject.toml` にすでに入っている）:

```
pytest        # 本体
pytest-cov    # カバレッジ計測
pytest-mock   # モック（mocker フィクスチャ）
```

インストールは `uv sync`。以降、コマンドは `uv run pytest ...` を想定する
（環境を有効化済みなら単に `pytest ...`）。

---

## 目次

1. [pytest の考え方と最初のテスト](#1-pytest-の考え方と最初のテスト)
2. [テストの実行とよく使うオプション](#2-テストの実行とよく使うオプション)
3. [assert の書き方（浮動小数・配列の比較）](#3-assert-の書き方浮動小数配列の比較)
4. [クラスでテストをまとめる](#4-クラスでテストをまとめる)
5. [前処理・後処理（xunit スタイル）](#5-前処理後処理xunit-スタイル)
6. [フィクスチャ（fixture）](#6-フィクスチャfixture)
7. [フィクスチャのスコープと autouse](#7-フィクスチャのスコープと-autouse)
8. [conftest.py で共有する](#8-conftestpy-で共有する)
9. [パラメータ化テスト（parametrize）](#9-パラメータ化テストparametrize)
10. [マーカー（skip / skipif / xfail / 自作マーカー）](#10-マーカーskip--skipif--xfail--自作マーカー)
11. [例外のテスト（pytest.raises）](#11-例外のテストpytestraises)
12. [標準出力のテスト（capsys）](#12-標準出力のテストcapsys)
13. [一時ファイル・ディレクトリ（tmp_path）](#13-一時ファイルディレクトリtmp_path)
14. [モック（pytest-mock）](#14-モックpytest-mock)
15. [スナップショットテスト](#15-スナップショットテスト)
16. [時刻の固定（freeze_time）](#16-時刻の固定freeze_time)
17. [カバレッジ（pytest-cov）](#17-カバレッジpytest-cov)
18. [ML プロジェクトでのテスト設計の勘所](#18-ml-プロジェクトでのテスト設計の勘所)
19. [チートシート](#19-チートシート)

---

## 1. pytest の考え方と最初のテスト

pytest のルールは驚くほど少ない。

- ファイル名は `test_*.py`（または `*_test.py`）
- 関数名は `test_` で始める
- 検証は `assert` 文で書く。`assert` が `False` なら失敗、`True`（例外なし）なら成功

まず「テスト対象のコード」を用意する。ML でよく書く前処理関数を例にする。

```python
# ml/preprocessing.py
import numpy as np

def min_max_scale(x: np.ndarray) -> np.ndarray:
    """特徴量を [0, 1] に正規化する。"""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi == lo:              # 全要素が同じ値なら 0 を返す
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
```

これに対するテスト:

```python
# tests/test_preprocessing.py
import numpy as np
from ml.preprocessing import min_max_scale

def test_min_max_scale_range():
    scaled = min_max_scale([0, 5, 10])
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0

def test_min_max_scale_constant_input():
    # 全部同じ値のときはゼロ除算にならず 0 が返る
    scaled = min_max_scale([3, 3, 3])
    assert (scaled == 0).all()
```

実行:

```bash
uv run pytest tests/test_preprocessing.py
```

pytest は「`assert` が通れば合格」というだけ。特別なアサーションメソッド
（`assertEqual` など）を覚える必要はなく、Python の式をそのまま書けるのが強み。

> **ポイント**：テスト関数は「準備（Arrange）→ 実行（Act）→ 検証（Assert）」の3段で書くと読みやすい。

---

## 2. テストの実行とよく使うオプション

```bash
uv run pytest                      # カレント以下の全テストを自動収集して実行
uv run pytest tests/               # ディレクトリ指定
uv run pytest tests/test_x.py      # ファイル指定
uv run pytest tests/test_x.py::test_min_max_scale_range   # 関数を1つだけ実行
```

覚えておくと捗るオプション:

| オプション | 意味 |
|---|---|
| `-v` | 詳細表示（どの関数が pass/fail したか関数名で見える） |
| `-s` | `print` の出力を表示する（通常は捕捉されて隠れる） |
| `-x` | 最初の失敗で即停止 |
| `--maxfail=2` | 失敗が2件出たら停止 |
| `-k "scale and not constant"` | 名前で絞り込み（式で書ける） |
| `-q` | 出力を簡潔に |
| `--lf` | 前回失敗したテストだけ再実行（last-failed） |
| `--ff` | 前回失敗を先に実行（fail-first） |

```bash
uv run pytest -v -s               # 詳細＋print表示、開発中の定番
uv run pytest -k "scale"          # 名前に scale を含むテストだけ
```

`print` デバッグしたいのに何も出ない、というのは pytest 初心者が必ずハマる点。
**`-s` を付ける**と出る。

---

## 3. assert の書き方（浮動小数・配列の比較）

ML では浮動小数や numpy 配列を比較する場面が多い。`==` の直接比較は誤差で落ちるので注意。

### 浮動小数：`pytest.approx`

```python
import pytest

def test_float_equality():
    assert 0.1 + 0.2 == pytest.approx(0.3)         # 誤差を許容して比較
    assert 0.1 + 0.2 != 0.3                          # 素の == は False になる

def test_accuracy_value():
    acc = 0.9999999
    assert acc == pytest.approx(1.0, abs=1e-3)       # 絶対誤差 1e-3 まで許容
```

### numpy 配列：`np.testing`

```python
import numpy as np
from ml.preprocessing import min_max_scale

def test_scale_exact_values():
    result = min_max_scale([0, 5, 10])
    expected = np.array([0.0, 0.5, 1.0])
    # 配列全体を誤差込みで比較（要素ごと）
    np.testing.assert_allclose(result, expected, rtol=1e-7)

def test_shape():
    result = min_max_scale(np.zeros((3, 4)))
    assert result.shape == (3, 4)     # 形状は素の == でOK
```

> **なぜ `assert (a == b).all()` を避けるか**：誤差で1要素でも違うと落ちる上、
> 失敗時に「どこがどれだけ違うか」が分からない。`np.testing.assert_allclose` は
> 差分を分かりやすく表示してくれる。

pytest は `assert` の式を解析して失敗理由を自動で詳しく表示する（アサーション書き換え機能）。
たとえば `assert scaled.max() == 1.0` が落ちると、`scaled.max()` の実際の値まで出る。

---

## 4. クラスでテストをまとめる

関連するテストを1つのクラスにまとめられる。**クラス名は `Test` で始める**
（`__init__` は書かない）。

```python
# tests/test_metrics.py
import numpy as np
from ml.metrics import accuracy

class TestAccuracy:
    def test_all_correct(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        assert accuracy(y_true, y_pred) == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        assert accuracy(y_true, y_pred) == 0.0
```

対象コード:

```python
# ml/metrics.py
import numpy as np

def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())
```

クラスにまとめる利点は、後述の「前処理・後処理」や「フィクスチャ」を
クラス単位で共有できること。

---

## 5. 前処理・後処理（xunit スタイル）

テストの前後に共通処理を差し込む古典的な方法。関数名が固定で決まっている。

### 関数レベル

```python
# tests/test_setup_function.py
def setup_function(function):
    print("＝＝ テスト開始前 ＝＝")

def teardown_function(function):
    print("＝＝ テスト終了後 ＝＝")

def test_a():
    print("test_a 本体")

def test_b():
    print("test_b 本体")
```

`uv run pytest -s tests/test_setup_function.py` で実行すると、各テストの前後で
setup/teardown が呼ばれるのが分かる。

### メソッド／クラス／モジュール レベル

| 関数名 | 呼ばれるタイミング |
|---|---|
| `setup_module` / `teardown_module` | モジュール（ファイル）全体で1回 |
| `setup_function` / `teardown_function` | 各テスト**関数**の前後 |
| `setup_class` / `teardown_class` | 各テスト**クラス**で1回（`@classmethod`） |
| `setup_method` / `teardown_method` | クラス内の各**メソッド**の前後 |

```python
# tests/test_setup_class.py
class TestModel:
    @classmethod
    def setup_class(cls):
        print("重いモデルを1回だけロード")   # クラスで1回

    @classmethod
    def teardown_class(cls):
        print("後片付け")

    def setup_method(self, method):
        print("各テストの前")

    def test_predict_shape(self):
        print("推論テスト1")

    def test_predict_range(self):
        print("推論テスト2")
```

この xunit スタイルは今も使えるが、pytest では次章の**フィクスチャ**の方が柔軟で推奨される。

---

## 6. フィクスチャ（fixture）

フィクスチャは「テストに渡す準備済みの値／リソース」を作る仕組み。
`@pytest.fixture` を付けた関数を定義し、**テスト関数の引数名に同じ名前を書く**と、
pytest がその戻り値を渡してくれる（依存性注入）。

```python
# tests/test_with_fixture.py
import numpy as np
import pytest
from ml.metrics import accuracy

@pytest.fixture
def sample_labels():
    # テストで使うデータを準備して返す
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    return y_true, y_pred

def test_accuracy_with_fixture(sample_labels):
    y_true, y_pred = sample_labels          # 引数名 = フィクスチャ名
    assert accuracy(y_true, y_pred) == pytest.approx(0.8)
```

同じデータを複数のテストで使い回せるので、重複が消える。

### 後処理を書く：`yield`

前処理と後処理の両方が必要なら、`return` の代わりに `yield` を使う。
`yield` の**前が前処理、後ろが後処理**。

```python
import pytest

@pytest.fixture
def temp_experiment_dir(tmp_path):
    # --- 前処理 ---
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    print("実験ディレクトリ作成")

    yield run_dir                # ← テストにこの値が渡る

    # --- 後処理（テスト終了後に必ず実行される）---
    print("実験ディレクトリ後始末")

def test_save_checkpoint(temp_experiment_dir):
    ckpt = temp_experiment_dir / "model.txt"
    ckpt.write_text("weights")
    assert ckpt.exists()
```

`yield` の後ろはテストが失敗しても必ず実行されるので、リソース解放
（DB切断、ファイル削除、GPUメモリ解放など）に向く。

> **補足**：フィクスチャを「値」として使わず前後処理だけしたいなら、
> `yield 何か` ではなく単に `yield` と書けばよい。

### 旧スタイル：`request.addfinalizer`

`yield` と等価な古い書き方。既存コードで見かける。

```python
@pytest.fixture
def setup_processing(request):
    print("前処理")
    def teardown():
        print("後処理")
    request.addfinalizer(teardown)     # yield の後ろに書くのと同じ
```

新規に書くなら `yield` の方が読みやすい。

---

## 7. フィクスチャのスコープと autouse

### スコープ：どの粒度で1回作るか

重いリソース（学習済みモデル、DB接続）を毎テスト作り直すのは無駄。
`scope` を指定すると生成回数を制御できる。

```python
@pytest.fixture(scope="module")     # このファイルで1回だけ生成
def trained_model():
    print("モデルをロード（重い処理）")
    model = load_heavy_model()
    return model
```

| scope | 生成のタイミング |
|---|---|
| `function`（既定） | テスト関数ごと |
| `class` | テストクラスごと |
| `module` | ファイルごと |
| `package` | パッケージごと |
| `session` | テスト全体で1回 |

ML では「学習済みモデルのロード」「サンプルデータセットの読み込み」を
`module` や `session` にすると、テスト時間が大きく縮む。

```python
# tests/test_model.py
import numpy as np
import pytest

@pytest.fixture(scope="module")
def model():
    print("\n== モデルをロード（1回だけ）==")
    # 本物の代わりに「入力の平均を返すだけ」のダミー推論器
    class DummyModel:
        def predict(self, x):
            return np.asarray(x).mean(axis=1)
    return DummyModel()

def test_predict_shape(model):
    x = np.random.rand(8, 4)
    y = model.predict(x)
    assert y.shape == (8,)

def test_predict_deterministic(model):
    x = np.ones((3, 4))
    y = model.predict(x)
    np.testing.assert_allclose(y, [1.0, 1.0, 1.0])
```

`-s` を付けて実行すると、"モデルをロード" が2つのテストで**1回だけ**出るのが確認できる。

### autouse：明示的に指定しなくても全部に適用

`autouse=True` にすると、引数に書かなくても対象範囲の全テストで自動実行される。

```python
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def fix_random_seed():
    # 全テストで乱数シードを固定（再現性の担保）
    np.random.seed(0)
    yield

def test_random_reproducible():
    a = np.random.rand(3)
    np.random.seed(0)
    b = np.random.rand(3)
    np.testing.assert_allclose(a, b)
```

「毎テスト前にシードを固定」「ログレベルを設定」などの共通処理に便利。
乱数を使う ML テストでは、シード固定の autouse フィクスチャは定番。

---

## 8. conftest.py で共有する

複数のテストファイルで同じフィクスチャを使いたいときは、`conftest.py` に置く。
**import 不要**で、同じディレクトリ以下の全テストから自動的に見える。

```python
# tests/conftest.py
import numpy as np
import pytest

@pytest.fixture(scope="session")
def sample_dataset():
    """全テスト共通のダミーデータセット。"""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y
```

```python
# tests/test_training.py
def test_dataset_shape(sample_dataset):   # import せずに使える
    X, y = sample_dataset
    assert X.shape == (100, 4)
    assert set(y.tolist()) <= {0, 1}
```

`conftest.py` はマーカー登録や共通設定の置き場所としても使う。

---

## 9. パラメータ化テスト（parametrize）

同じテストを入力違いで何度も回したいときに使う。ML の
「複数の入力パターン」「複数のハイパーパラメータ」の検証に最適。

```python
# tests/test_parametrize.py
import numpy as np
import pytest
from ml.metrics import accuracy

@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 1, 1, 0], [0, 1, 1, 0], 1.0),      # 全問正解
        ([0, 1, 1, 0], [1, 0, 0, 1], 0.0),      # 全問不正解
        ([0, 1, 1, 0], [0, 1, 0, 0], 0.75),     # 3/4 正解
    ],
)
def test_accuracy_cases(y_true, y_pred, expected):
    assert accuracy(np.array(y_true), np.array(y_pred)) == pytest.approx(expected)
```

これで**3つの独立したテスト**として実行・集計される（1つ落ちても他は動く）。
`-v` で見ると `test_accuracy_cases[y_true0-...]` のように1件ずつ表示される。

### 複数の parametrize を掛け合わせる

デコレータを重ねると直積（全組み合わせ）になる。ハイパラのグリッド検証に便利。

```python
@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("n_features", [4, 16])
def test_forward_shape(batch_size, n_features):
    x = np.random.rand(batch_size, n_features)
    y = x.mean(axis=1)                      # ダミーの forward
    assert y.shape == (batch_size,)         # 3 × 2 = 6 通り実行される
```

### 変数にまとめて渡す（資料の書き方）

パラメータのリストを変数に切り出しても同じ。

```python
input_data = [
    ("私の名前は佐藤です。", "日本語", "英語", "My name is Sato."),
    ("こんにちは", "日本語", "英語", "Hello"),
]

@pytest.mark.parametrize("text, src, dst, expected", input_data)
def test_translate(text, src, dst, expected):
    ...
```

### `ids` で名前を付ける

失敗時に読みやすくなる。

```python
@pytest.mark.parametrize(
    "lr", [0.1, 0.01, 0.001],
    ids=["lr_high", "lr_mid", "lr_low"],
)
def test_learning_rate(lr):
    assert 0 < lr < 1
```

---

## 10. マーカー（skip / skipif / xfail / 自作マーカー）

マーカーはテストに「タグ」や「振る舞い」を付ける仕組み。`@pytest.mark.xxx` で付ける。

### 組み込みマーカー

```python
import sys
import pytest

@pytest.mark.skip(reason="前処理APIの仕様変更待ち")
def test_not_ready():
    ...

@pytest.mark.skipif(sys.platform == "win32", reason="CUDAが無い環境ではスキップ")
def test_gpu_inference():
    ...

@pytest.mark.xfail(reason="既知のバグ。修正予定")
def test_known_bug():
    assert buggy_function() == 42     # 失敗しても xfail 扱いで赤にならない
```

- `skip`：無条件でスキップ
- `skipif`：条件を満たすときだけスキップ（環境依存テストで多用）
- `xfail`：失敗する想定。失敗→`xfail`、たまたま成功→`xpass` として区別される

GPU が要るテストを CPU-only の CI でスキップ、といった使い方が ML では頻出。

### 自作マーカー

`@pytest.mark.任意の名前` でタグを付け、実行時に絞り込める。

```python
# tests/test_custom_marker.py
import pytest

@pytest.mark.slow
def test_full_training():
    ...              # 時間のかかる学習テスト

@pytest.mark.fast
def test_single_forward():
    ...
```

自作マーカーは**登録が必要**（未登録だと警告が出る）。`pytest.ini` か
`pyproject.toml` に書く。

```ini
# pytest.ini
[pytest]
markers =
    slow: 時間のかかるテスト
    fast: すぐ終わるテスト
```

```toml
# pyproject.toml に書く場合
[tool.pytest.ini_options]
markers = [
    "slow: 時間のかかるテスト",
    "fast: すぐ終わるテスト",
]
```

実行時に `-m` で絞り込む（論理式が書ける）:

```bash
uv run pytest -m slow                 # slow だけ
uv run pytest -m "not slow"           # slow 以外（CI の高速チェック向け）
uv run pytest -m "fast and not slow"  # 両条件
uv run pytest -m "slow or fast"       # どちらか
```

「普段は `-m "not slow"` で高速に、夜間 CI で全部」といった運用ができる。

---

## 11. 例外のテスト（pytest.raises）

「この入力なら例外を投げるべき」を検証する。`with pytest.raises(例外):` の
ブロック内で**その例外が出れば合格**、出なければ失敗。

```python
# tests/test_exceptions.py
import numpy as np
import pytest
from ml.metrics import accuracy

def test_shape_mismatch_raises():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1])          # 長さが違う → 比較で壊れるべき
    with pytest.raises(ValueError):
        check_same_length(y_true, y_pred)
```

対象コード側で明示的に投げる例:

```python
# ml/metrics.py（追記）
def check_same_length(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
```

### 複数の例外を許容

```python
with pytest.raises((ValueError, TypeError)):
    ...
```

### 例外オブジェクトを受け取ってメッセージも検証

```python
def test_error_message():
    with pytest.raises(ValueError) as exc_info:
        check_same_length([0, 1, 1], [0, 1])
    assert "mismatch" in str(exc_info.value)      # メッセージ内容まで確認
```

`match=` を使うと正規表現でメッセージを1行で検証できる:

```python
with pytest.raises(ValueError, match="mismatch"):
    check_same_length([0, 1, 1], [0, 1])
```

---

## 12. 標準出力のテスト（capsys）

`print` などの標準出力／標準エラー出力を捕捉して検証する組み込みフィクスチャ。
テスト関数の引数に `capsys` を書き、`capsys.readouterr()` で取り出す。

```python
# tests/test_capsys.py
def train_report(epoch, loss):
    print(f"epoch={epoch} loss={loss:.3f}")

def test_train_report(capsys):
    train_report(1, 0.1234)
    out, err = capsys.readouterr()      # out=標準出力, err=標準エラー
    assert out == "epoch=1 loss=0.123\n"
    assert err == ""
```

学習ループのログ出力やCLIの表示内容を検証したいときに使う。
（`-s` と違い、`capsys` は**検証のために捕捉**する。両立はしない点に注意。）

---

## 13. 一時ファイル・ディレクトリ（tmp_path）

テストでファイルを作りたいとき、実プロジェクトを汚さないための組み込みフィクスチャ。
`tmp_path` にはテストごとに使い捨ての一時ディレクトリ（`pathlib.Path`）が入る。

```python
# tests/test_tmp_path.py
import numpy as np

def save_predictions(path, preds):
    path.write_text("\n".join(map(str, preds)))

def test_save_predictions(tmp_path):
    out_file = tmp_path / "preds.txt"          # / でパス連結
    save_predictions(out_file, [0, 1, 1, 0])
    assert out_file.exists()
    assert out_file.read_text() == "0\n1\n1\n0"
```

`pathlib.Path` なので `tmp_path / "sub" / "file.npy"` のように書ける。
モデルのチェックポイント保存・読み込みや、前処理結果のファイル出力テストに使う。

```python
def test_checkpoint_roundtrip(tmp_path):
    import numpy as np
    ckpt = tmp_path / "weights.npy"
    w = np.array([1.0, 2.0, 3.0])
    np.save(ckpt, w)                            # 保存
    loaded = np.load(ckpt)                       # 読み込み
    np.testing.assert_allclose(loaded, w)        # 往復して一致するか
```

テストが終われば pytest が自動で片付ける（数回分は残して古いものから削除）。

---

## 14. モック（pytest-mock）

外部依存（API 呼び出し、GPU、DB、重いモデル）を**偽物に差し替える**のがモック。
`pytest-mock` を入れると `mocker` フィクスチャが使える
（標準ライブラリ `unittest.mock` の薄いラッパーで、後始末が自動）。

ML で最重要なのは「**外部APIや重い処理を呼ばずにロジックだけテストする**」こと。
資料の翻訳クラスを例にする。

```python
# translator.py
from googletrans import Translator

class GoogleTranslator:
    def __init__(self):
        self.translator = Translator()

    def get_language_id(self, language_name):
        languages = {"日本語": "ja", "英語": "en", "フランス語": "fr"}
        return languages[language_name]      # 未登録なら KeyError

    def convert(self, text, src_name, dst_name):
        src = self.get_language_id(src_name)
        dst = self.get_language_id(dst_name)
        return self.translator.translate(text, src=src, dest=dst).text
```

### 戻り値を固定する：`return_value`

`convert` は内部で外部翻訳APIを叩く。テストでは叩かせたくない。

```python
# tests/test_mock.py
from translator import GoogleTranslator

def test_convert_mocked(mocker):
    # GoogleTranslator.convert を、常に "hello world!" を返す偽物に差し替え
    mocker.patch(
        "translator.GoogleTranslator.convert",
        return_value="hello world!",
    )
    trans = GoogleTranslator()
    result = trans.convert("私の名前は佐藤です。", "日本語", "英語")
    assert result == "hello world!"       # 実際のAPIは呼ばれない
```

`mocker.patch("差し替える場所の文字列パス", return_value=...)` が基本形。
**差し替えるパスは「使われる場所」を指す**のが原則（import の仕方で変わる）。

### 入力に応じて戻り値を変える：`side_effect`（関数）

```python
def test_side_effect_func(mocker):
    def fake_lang_id(name):
        return {"日本語": "ja", "英語": "en"}.get(name, "xx")

    mocker.patch(
        "translator.GoogleTranslator.get_language_id",
        side_effect=fake_lang_id,          # 呼ばれるたびにこの関数が実行される
    )
    trans = GoogleTranslator()
    assert trans.get_language_id("日本語") == "ja"
    assert trans.get_language_id("英語") == "en"
```

### 例外を起こさせる：`side_effect`（例外）

異常系（APIが落ちたときの挙動）をテストできる。

```python
import pytest

def test_side_effect_exception(mocker):
    mock_obj = mocker.patch("translator.GoogleTranslator.get_language_id")
    mock_obj.side_effect = Exception("ConvertException")

    trans = GoogleTranslator()
    with pytest.raises(Exception) as exc_info:
        trans.convert("私の名前は佐藤です。", "日本語", "英語")
    assert exc_info.value.args[0] == "ConvertException"
```

### 「正しい引数で呼ばれたか」を検証する

モックは戻り値だけでなく**呼ばれ方**も記録している。前処理が正しい引数で
モデルを呼んだか、などを確認できる。

```python
def test_call_args(mocker):
    mock_obj = mocker.patch("translator.GoogleTranslator.get_language_id",
                            side_effect=lambda name: "ja")
    trans = GoogleTranslator()
    trans.convert("テスト", "日本語", "英語")

    assert mock_obj.call_count == 2                      # 2回呼ばれた
    mock_obj.assert_any_call("日本語")                    # この引数で呼ばれた
    mock_obj.assert_any_call("英語")
    # 呼び出し履歴の生データ
    first_call = mock_obj.call_args_list[0]
    assert first_call.args[0] == "日本語"
```

よく使うメソッド:

| 属性/メソッド | 意味 |
|---|---|
| `mock.return_value = v` | 戻り値を固定 |
| `mock.side_effect = f` | 呼び出しごとに `f` を実行（関数 or 例外 or リスト） |
| `mock.call_count` | 呼ばれた回数 |
| `mock.call_args` | 最後の呼び出し引数 |
| `mock.call_args_list` | 全呼び出しの引数リスト |
| `mock.assert_called_once()` | ちょうど1回呼ばれたか |
| `mock.assert_any_call(...)` | その引数で呼ばれたことがあるか |

> **ML での典型例**：`torch.load` / `joblib.load` をモックしてディスクI/Oを回避、
> 外部推論APIをモックしてネットワークなしでパイプラインをテスト、
> `time.sleep` をモックしてリトライ処理を高速にテスト、など。

---

## 15. スナップショットテスト

出力が複雑で「期待値を手で書くのが大変」なとき、**初回の出力を保存しておき、
次回以降それと一致するか**を比較する手法。モデルの設定辞書や、前処理後の
特徴量サマリなどの回帰テストに向く。

`syrupy`（pytest-syrupy）を使うのが今の主流。`snapshot` フィクスチャを使う。

```bash
uv add --dev syrupy
```

```python
# tests/test_snapshot.py
def build_model_config(lr, n_layers):
    return {
        "optimizer": "adam",
        "lr": lr,
        "n_layers": n_layers,
        "activation": "relu",
    }

def test_model_config(snapshot):
    config = build_model_config(0.01, 3)
    assert config == snapshot         # 初回は保存、次回以降は比較
```

- 初回実行：`uv run pytest --snapshot-update` で期待値ファイルを生成
- 以降：`uv run pytest` で保存済みスナップショットと比較。差分があれば失敗
- 意図的に出力を変えたとき：再度 `--snapshot-update` で更新

> **重要（資料より）**：スナップショットの保存ディレクトリ（`__snapshots__` など）は
> **必ず git 管理に含める**。含めないと、clone した環境では「前回の記録が無い＝比較対象なし」
> となり、コードが壊れていてもテストが通ってしまう。

（資料の `snapshottest` は古いライブラリで新しめの Python では動きにくい。
新規なら `syrupy` を推奨。考え方は同じ。）

---

## 16. 時刻の固定（freeze_time）

「現在時刻」に依存するコードは、そのままだとテストのたびに結果が変わる。
`pytest-freezer`（または `freezegun`）で時刻を固定できる。

```bash
uv add --dev pytest-freezer
```

```python
# tests/test_time.py
import datetime
import pytest

def make_run_id():
    # 実行時刻からラン ID を作る（実験管理でよくある）
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

@pytest.mark.freeze_time("2026-07-07 09:00:00")
def test_make_run_id():
    assert make_run_id() == "run_20260707_090000"    # 時刻が固定されるので決定的
```

`@pytest.mark.freeze_time(...)` を付けた関数の中だけ、`datetime.now()` などが
指定時刻を返す。ログのタイムスタンプ、TTL、スケジューラなどのテストに使う。

---

## 17. カバレッジ（pytest-cov）

テストがコードのどれだけを実行したか（カバー率）を測る。`pytest-cov` は導入済み。

```bash
uv run pytest --cov=ml                       # ml パッケージのカバレッジを表示
uv run pytest --cov=ml --cov-report=term-missing   # 未実行の行番号も表示
uv run pytest --cov=ml --cov-report=html     # htmlcov/ にHTMLレポート生成
```

`term-missing` は「どの行がテストされていないか」を行番号で教えてくれるので、
テストの穴を埋めるのに便利。

```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
ml/metrics.py               8      1    88%   14
ml/preprocessing.py         7      0   100%
```

> **注意**：カバレッジ100%は「全行が1回実行された」だけで、正しさの証明ではない。
> ML では特に、境界値・異常系・shape違いなど**意味のあるケース**を測る方が大事。
> 数値目標より「重要なロジックが検証されているか」を見る。

`pyproject.toml` にデフォルトを書いておける:

```toml
[tool.pytest.ini_options]
addopts = "--cov=ml --cov-report=term-missing"
testpaths = ["tests"]
```

---

## 18. ML プロジェクトでのテスト設計の勘所

ML コードは「正解が確率的・近似的」なので、一般的なソフトウェアと少しコツが違う。

**1. 何をテストするか — 決定的な部分を切り出す**
モデルの精度そのものは環境で揺れる。テストしやすいのは:
- 前処理・特徴量エンジニアリング（入力→出力が決定的）
- 評価指標の計算（`accuracy`, `f1` などの実装）
- データのバリデーション（欠損・shape・型・値域）
- I/O（保存→読み込みの往復一致）

**2. shape と dtype を必ず確認する**
バグの多くは「次元がずれる」「int と float が混ざる」で起きる。

```python
def test_output_contract(model):
    x = np.random.rand(16, 8)
    y = model.predict(x)
    assert y.shape == (16,)             # 期待する形
    assert y.dtype == np.float64        # 期待する型
    assert not np.isnan(y).any()        # NaN が混じっていない
```

**3. 数値は必ず誤差込みで比較する**（`pytest.approx` / `np.testing.assert_allclose`）

**4. 乱数はシードを固定する**（autouse フィクスチャが便利、[7章](#7-フィクスチャのスコープと-autouse)）

**5. 重い・外部依存は分離する**
- 重いモデルのロードは `scope="module"` 以上のフィクスチャで1回に
- 外部API・GPU・巨大データはモック（[14章](#14-モックpytest-mock)）や `skipif`（[10章](#10-マーカーskip--skipif--xfail--自作マーカー)）で切り離す
- 学習まで走る重いテストは `@pytest.mark.slow` を付け、普段は `-m "not slow"`

**6. メタモルフィックテスト（性質ベース）**
正解値が分からなくても「成り立つべき性質」は書ける。

```python
def test_scale_is_monotonic():
    # 正規化しても大小関係は保たれるべき
    x = [3, 1, 4, 1, 5, 9]
    scaled = min_max_scale(x)
    assert list(np.argsort(x)) == list(np.argsort(scaled))

def test_shuffle_invariance(model):
    # 行をシャッフルしても各行の予測は変わらないはず（順序非依存）
    x = np.random.rand(5, 4)
    idx = [3, 0, 4, 1, 2]
    np.testing.assert_allclose(
        model.predict(x)[idx],
        model.predict(x[idx]),
    )
```

推奨ディレクトリ構成:

```
project/
├── ml/                     # 本体コード
│   ├── preprocessing.py
│   └── metrics.py
├── tests/
│   ├── conftest.py         # 共通フィクスチャ（データ、シード固定）
│   ├── test_preprocessing.py
│   └── test_metrics.py
└── pyproject.toml          # [tool.pytest.ini_options] に設定
```

---

## 19. チートシート

### 実行コマンド

```bash
uv run pytest                         # 全実行
uv run pytest -v -s                   # 詳細 + print 表示
uv run pytest -x                      # 最初の失敗で停止
uv run pytest -k "scale"              # 名前で絞り込み
uv run pytest -m "not slow"           # マーカーで絞り込み
uv run pytest --lf                    # 前回失敗のみ
uv run pytest path::TestClass::test_x # 1つだけ実行
uv run pytest --cov=ml --cov-report=term-missing   # カバレッジ
```

### アサーション

```python
assert x == y                                   # 基本
assert a == pytest.approx(b)                    # 浮動小数
np.testing.assert_allclose(arr1, arr2)          # numpy 配列
with pytest.raises(ValueError):                 # 例外
    ...
with pytest.raises(ValueError, match="msg"):    # 例外メッセージ
    ...
```

### フィクスチャ

```python
@pytest.fixture                       # 値を渡す
@pytest.fixture(scope="module")       # 生成粒度（function/class/module/package/session）
@pytest.fixture(autouse=True)         # 自動適用
def f():
    ...        # 前処理
    yield v    # v をテストに渡す
    ...        # 後処理

# 組み込みフィクスチャ
tmp_path      # 一時ディレクトリ（pathlib.Path）
capsys        # 標準出力/エラーの捕捉
mocker        # モック（pytest-mock）
```

### マーカー

```python
@pytest.mark.skip(reason="...")
@pytest.mark.skipif(cond, reason="...")
@pytest.mark.xfail(reason="...")
@pytest.mark.parametrize("a, b", [(1, 2), (3, 4)])
@pytest.mark.slow                     # 自作（要登録）
```

### モック（mocker）

```python
mocker.patch("mod.Cls.method", return_value=v)
mocker.patch("mod.Cls.method", side_effect=func_or_exc)
m = mocker.patch("mod.func")
m.assert_called_once()
m.assert_any_call(arg)
m.call_count, m.call_args_list
```

---

これで pytest の基本から ML 現場で使う実践パターンまでを一通りカバーした。
まずは `min_max_scale` や `accuracy` のような小さな決定的関数にテストを書くところから
始めると、テストの効果を実感しやすい。
