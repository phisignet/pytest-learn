"""モデルの学習・推論・永続化。

テスト観点:
- 未学習で predict したら明確なエラーになるか（フェイルファスト）
- 出力の shape / 値域（確率なら [0, 1]）
- save → load の往復で予測が一致するか（roundtrip）
- seed 固定で再現するか
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class NotFittedError(RuntimeError):
    """未学習のモデルで推論しようとしたときに投げる例外。"""


class ChurnModel:
    """ロジスティック回帰によるチャーン予測モデル。"""

    def __init__(self, C: float = 1.0, seed: int = 42):
        self.clf = LogisticRegression(C=C, random_state=seed, max_iter=1000)
        self._fitted = False

    def fit(self, X, y) -> "ChurnModel":
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """チャーン確率（正例クラスの確率）を (n_samples,) で返す。"""
        if not self._fitted:
            raise NotFittedError("model is not fitted; call fit() first")
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "ChurnModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(model).__name__}")
        return model
