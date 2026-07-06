"""評価レポート。

指標計算は「小さな入力なら手で正解が計算できる」ので、
既知ケースを parametrize で網羅するのがテストの定石。
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluation_report(y_true, y_proba, threshold: float = 0.5) -> dict:
    """確率予測をしきい値で二値化し、主要指標をまとめた辞書を返す。"""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "threshold": threshold,
        "n_samples": int(len(y_true)),
    }
