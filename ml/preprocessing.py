"""前処理まわりの関数（テスト対象コード）。

入力から出力が決まる「決定的」な関数は、機械学習コードの中でも
とくにテストしやすい部分。まずはここから練習するとよい。
"""
import numpy as np


def min_max_scale(x) -> np.ndarray:
    """特徴量を [0, 1] に正規化する（Min-Max 正規化）。

    全要素が同じ値のときはゼロ除算になるので、その場合は 0 を返す。
    """
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def train_test_split(x, test_ratio=0.2):
    """配列を先頭から (train, test) に分割する簡易版。

    test_ratio が [0, 1] の範囲外なら ValueError を投げる。
    """
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError(f"test_ratio must be in [0, 1], got {test_ratio}")
    x = np.asarray(x)
    n_test = int(len(x) * test_ratio)
    return x[n_test:], x[:n_test]
