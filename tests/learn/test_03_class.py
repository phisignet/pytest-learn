"""レッスン03: クラスでテストをまとめる

【ポイント】
- 関連するテストは Test で始まるクラスにまとめられる（__init__ は書かない）
- 各テストメソッドは self を取り、名前は test_ で始める
- まとめておくと、後のレッスンで前処理・フィクスチャをクラス単位で共有できる

【試すコマンド】
    uv run pytest tests/learn/test_03_class.py -v
"""
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

    def test_partial(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])          # 3/4 正解
        assert accuracy(y_true, y_pred) == 0.75
