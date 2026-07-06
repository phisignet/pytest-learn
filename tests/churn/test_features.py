"""特徴量層のテスト。

実務での狙い:
- 出力カラムの「契約」（名前・順序）を守る — モデルはこの順序に依存する
- NaN / inf を絶対に下流へ流さない
- 決定性（同じ入力 → 同じ出力）
- 境界値（tenure=0、is_new_customer のしきい値前後）
"""
import numpy as np
import pandas as pd
import pytest

from ml.churn.features import FEATURE_COLUMNS, build_features


def _one_row(tenure=12, charges=50.0, contract="one-year"):
    return pd.DataFrame(
        {
            "tenure_months": [tenure],
            "monthly_charges": [charges],
            "contract_type": [contract],
            "churned": [0],
        }
    )


def test_column_contract(raw_df):
    # カラムの名前と順序が契約どおり（モデルの入力仕様そのもの）
    features = build_features(raw_df)
    assert list(features.columns) == FEATURE_COLUMNS


def test_no_nan_or_inf(raw_df):
    features = build_features(raw_df)
    values = features.to_numpy(dtype=float)
    assert not np.isnan(values).any()
    assert np.isfinite(values).all()


def test_deterministic(raw_df):
    # 同じ入力から2回作って完全一致（乱数や状態に依存していない）
    f1 = build_features(raw_df)
    f2 = build_features(raw_df)
    pd.testing.assert_frame_equal(f1, f2)


def test_tenure_zero_is_safe():
    # tenure=0 でもゼロ除算にならない（+1 補正の検証）
    features = build_features(_one_row(tenure=0, charges=80.0))
    assert features["charges_per_tenure"].iloc[0] == pytest.approx(80.0)


@pytest.mark.parametrize(
    "tenure, expected",
    [(0, 1.0), (5, 1.0), (6, 0.0), (72, 0.0)],
    ids=["brand_new", "just_below", "boundary", "long_term"],
)
def test_is_new_customer_boundary(tenure, expected):
    # しきい値（6か月）の境界を両側から確認する
    features = build_features(_one_row(tenure=tenure))
    assert features["is_new_customer"].iloc[0] == expected


def test_one_hot_sums_to_one(raw_df):
    # one-hot は各行でちょうど1つだけ 1 になる
    features = build_features(raw_df)
    onehot = features[[c for c in FEATURE_COLUMNS if c.startswith("contract_")]]
    np.testing.assert_allclose(onehot.sum(axis=1), 1.0)
