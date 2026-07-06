"""評価指標（テスト対象コード）。

評価指標の実装は「入力に対する正解が計算できる」ので、
数値テストの題材にちょうどよい。
"""
import numpy as np


def check_same_length(y_true, y_pred) -> None:
    """2つの配列の長さが違えば ValueError を投げる。"""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: {len(y_true)} != {len(y_pred)}"
        )


def accuracy(y_true, y_pred) -> float:
    """正解率を返す。"""
    check_same_length(y_true, y_pred)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())
