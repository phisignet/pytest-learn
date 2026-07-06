"""評価層のテスト: 手計算できる既知ケースで指標の実装を検証する。

実務での狙い:
- 指標の実装ミス（precision と recall の取り違え等）は静かに壊れるので、
  小さな手計算ケースで固定しておく
- レポートのキー構成（下流のダッシュボード等が依存する契約）を守る
"""
import pytest

from ml.churn.evaluate import evaluation_report


def test_perfect_predictions():
    report = evaluation_report(
        y_true=[0, 0, 1, 1],
        y_proba=[0.1, 0.2, 0.9, 0.8],
    )
    assert report["accuracy"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["auc"] == 1.0


def test_known_hand_computed_case():
    # y_true:  [1,   1,   0,   0  ]
    # y_pred:  [1,   0,   1,   0  ]  (threshold 0.5)
    # TP=1, FN=1, FP=1, TN=1 → accuracy=0.5, precision=0.5, recall=0.5
    report = evaluation_report(
        y_true=[1, 1, 0, 0],
        y_proba=[0.9, 0.1, 0.8, 0.2],
    )
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(0.5)


def test_report_keys_contract():
    # レポートのキーは下流（ダッシュボード・ログ基盤）との契約
    report = evaluation_report([0, 1], [0.3, 0.7])
    assert set(report.keys()) == {
        "accuracy", "precision", "recall", "auc", "threshold", "n_samples",
    }
    assert report["n_samples"] == 2


@pytest.mark.parametrize(
    "threshold, expected_recall",
    [
        (0.1, 1.0),    # しきい値を下げると正例を拾いやすい → recall 高
        (0.95, 0.0),   # しきい値を上げると正例を逃す → recall 低
    ],
)
def test_threshold_changes_recall(threshold, expected_recall):
    report = evaluation_report(
        y_true=[1, 1, 0, 0],
        y_proba=[0.6, 0.7, 0.2, 0.3],
        threshold=threshold,
    )
    assert report["recall"] == pytest.approx(expected_recall)


def test_all_negative_predictions_do_not_crash():
    # 正例を1つも予測しないと precision は 0/0 になりがち → zero_division=0 の検証
    report = evaluation_report([0, 1], [0.1, 0.2])
    assert report["precision"] == 0.0
