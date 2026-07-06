"""レッスン06: フィクスチャのスコープと autouse

【ポイント】
- scope で「どの粒度で1回だけ作るか」を指定できる
    function(既定) / class / module / package / session
  重いモデルのロードや大きなデータの読み込みは module や session にすると
  生成が1回で済み、テスト時間が大きく縮む
- autouse=True にすると、引数に書かなくても対象範囲の全テストで自動実行される
  （このプロジェクトでは conftest.py のシード固定フィクスチャが autouse の例）

【試すコマンド】
    uv run pytest tests/learn/test_06_fixture_scope_autouse.py -v -s
    # "モデルをロード" が2つのテストで1回だけ出ることを確認する
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def model():
    """module スコープ: このファイルで1回だけ生成される。"""
    print("\n[fixture] == モデルをロード（重い処理を1回だけ）==")

    class DummyModel:
        def predict(self, x):
            return np.asarray(x).mean(axis=1)     # 各行の平均を返すだけ

    return DummyModel()


def test_predict_shape(model):
    x = np.random.rand(8, 4)
    y = model.predict(x)
    assert y.shape == (8,)


def test_predict_deterministic(model):
    x = np.ones((3, 4))
    y = model.predict(x)
    np.testing.assert_allclose(y, [1.0, 1.0, 1.0])


def test_seed_is_fixed_by_autouse():
    """conftest.py の autouse フィクスチャでシードが固定されている確認。

    毎テスト前に np.random.seed(0) が呼ばれるので、この値は常に同じになる。
    """
    a = np.random.rand(3)
    np.random.seed(0)
    b = np.random.rand(3)
    np.testing.assert_allclose(a, b)
