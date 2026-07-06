"""推論器（モックの練習用テスト対象コード）。

load_weights() は「本来はディスクや外部ストレージからモデルを読む
重い / 外部依存の処理」に見立てている。テストではここをモックで
差し替えて、実際の I/O を起こさずにロジックだけを検証する。
"""
import numpy as np


def load_weights(path) -> np.ndarray:
    """モデルの重みをファイルから読み込む（重い外部依存の想定）。"""
    return np.load(path)


class Predictor:
    """線形モデルの推論器。"""

    def __init__(self, weights_path):
        # __init__ の中で外部依存を呼んでいる点がミソ
        self.w = load_weights(weights_path)

    def predict(self, x) -> np.ndarray:
        """x @ w を返す。x: (n_samples, n_features)。"""
        return np.asarray(x, dtype=float) @ self.w
