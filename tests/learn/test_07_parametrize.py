"""レッスン07: パラメータ化テスト（parametrize）

【ポイント】
- @pytest.mark.parametrize で、同じテストを入力違いで何度も回せる
- それぞれ独立したテストとして集計される（1つ落ちても他は動く）
- デコレータを重ねると全組み合わせ（直積）になる → ハイパラのグリッド検証に便利
- ids= で各ケースに読みやすい名前を付けられる

【試すコマンド】
    uv run pytest tests/learn/test_07_parametrize.py -v
    # test_accuracy_cases[...] のように1件ずつ表示される
"""
import numpy as np
import pytest

from ml.metrics import accuracy


# 変数に切り出しても、直接書いても同じ
cases = [
    ([0, 1, 1, 0], [0, 1, 1, 0], 1.0),          # 全問正解
    ([0, 1, 1, 0], [1, 0, 0, 1], 0.0),          # 全問不正解
    ([0, 1, 1, 0], [0, 1, 0, 0], 0.75),         # 3/4 正解
]


@pytest.mark.parametrize("y_true, y_pred, expected", cases)
def test_accuracy_cases(y_true, y_pred, expected):
    assert accuracy(np.array(y_true), np.array(y_pred)) == pytest.approx(expected)


# デコレータを重ねる → 3 × 2 = 6 通り実行される（ハイパラのグリッド検証）
@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("n_features", [4, 16])
def test_forward_shape(batch_size, n_features):
    x = np.random.rand(batch_size, n_features)
    y = x.mean(axis=1)                           # ダミーの forward
    assert y.shape == (batch_size,)


# ids で名前を付けると、失敗時に読みやすい
@pytest.mark.parametrize(
    "lr", [0.1, 0.01, 0.001],
    ids=["lr_high", "lr_mid", "lr_low"],
)
def test_learning_rate_range(lr):
    assert 0 < lr < 1
