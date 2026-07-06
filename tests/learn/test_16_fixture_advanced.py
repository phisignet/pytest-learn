"""レッスン16: フィクスチャの応用（パラメータ化・ファクトリ・tmp_path_factory）

【ポイント】
1. パラメータ化フィクスチャ: @pytest.fixture(params=[...])
   - フィクスチャ自体を複数パターンで回す。そのフィクスチャを使う
     「全テスト」が params の数だけ実行される
   - parametrize（レッスン07）との違い:
       parametrize … そのテスト1つの入力を変える
       params=     … フィクスチャを使う全テストをまとめて変える
     （例:「どの dtype でも前処理が動く」を全テストに一括適用）
2. フィクスチャファクトリ: フィクスチャから「関数」を返す
   - テストごとに引数を変えてデータを作りたいときの定石
3. tmp_path_factory: session スコープでも使える一時ディレクトリ
   - tmp_path は function スコープ専用。session フィクスチャの中では
     tmp_path_factory.mktemp(...) を使う

【試すコマンド】
    uv run pytest tests/learn/test_16_fixture_advanced.py -v
    # dtype ごとにテストが増えていることを -v で確認する
"""
import numpy as np
import pytest

from ml.preprocessing import min_max_scale


# ---- 1. パラメータ化フィクスチャ -------------------------------------------
@pytest.fixture(params=[np.float32, np.float64, np.int64],
                ids=["f32", "f64", "i64"])
def typed_array(request):
    """dtype 違いの入力を作る。request.param に params の各値が入る。"""
    return np.array([0, 5, 10], dtype=request.param)


def test_scale_range_any_dtype(typed_array):
    # このテストは dtype 3種 × 1 = 3回実行される
    scaled = min_max_scale(typed_array)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


def test_scale_output_is_float_any_dtype(typed_array):
    # 入力が int でも出力は float であること（契約）も3回検証される
    assert min_max_scale(typed_array).dtype == np.float64


# ---- 2. フィクスチャファクトリ ----------------------------------------------
@pytest.fixture
def make_dataset():
    """「データセットを作る関数」を返すフィクスチャ。

    テスト側で好きなサイズ・クラス数を指定して呼べる。
    生成したものの後始末が要る場合は、リストに溜めて yield 後に片付ける。
    """
    def _make(n_samples: int = 10, n_features: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features))
        y = rng.integers(0, 2, size=n_samples)
        return X, y

    return _make


def test_small_dataset(make_dataset):
    X, y = make_dataset(n_samples=5)              # テストごとに引数を変えられる
    assert X.shape == (5, 4)


def test_wide_dataset(make_dataset):
    X, y = make_dataset(n_samples=8, n_features=100)
    assert X.shape == (8, 100)


# ---- 3. tmp_path_factory: session スコープの一時ディレクトリ ---------------
@pytest.fixture(scope="session")
def shared_model_dir(tmp_path_factory):
    """全テストで共有する一時ディレクトリ（session では tmp_path が使えないため）。

    実務では「重い前処理済みデータを1回だけ書き出し、全テストで読む」用途など。
    """
    d = tmp_path_factory.mktemp("models")
    (d / "weights.npy").write_bytes(b"")          # ここで1回だけ準備する想定
    np.save(d / "weights.npy", np.array([1.0, 2.0]))
    return d


def test_shared_dir_readable(shared_model_dir):
    w = np.load(shared_model_dir / "weights.npy")
    np.testing.assert_allclose(w, [1.0, 2.0])


def test_shared_dir_is_same_instance(shared_model_dir):
    # session スコープなので、前のテストと同じディレクトリが渡ってくる
    assert (shared_model_dir / "weights.npy").exists()
