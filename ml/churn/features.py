"""特徴量エンジニアリング。

特徴量関数は「決定的（同じ入力 → 同じ出力）」なので、テストの費用対効果が
最も高い部分。出力カラムの契約（名前・順序・NaNなし）を必ずテストする。
"""
import numpy as np
import pandas as pd

from ml.churn.data import VALID_CONTRACTS

# 出力カラムの契約。モデルはこの順序に依存する
FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "charges_per_tenure",
    "is_new_customer",
    *[f"contract_{c}" for c in VALID_CONTRACTS],
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """生データから特徴量テーブルを作る。

    - charges_per_tenure: 在籍1か月あたりの支払額（tenure=0 でも割れるよう +1）
    - is_new_customer:    在籍 6 か月未満なら 1
    - contract_*:         契約種別の one-hot
    """
    out = pd.DataFrame(index=df.index)
    out["tenure_months"] = df["tenure_months"].astype(float)
    out["monthly_charges"] = df["monthly_charges"].astype(float)
    out["charges_per_tenure"] = out["monthly_charges"] / (out["tenure_months"] + 1.0)
    out["is_new_customer"] = (out["tenure_months"] < 6).astype(float)
    for c in VALID_CONTRACTS:
        out[f"contract_{c}"] = (df["contract_type"] == c).astype(float)
    return out[FEATURE_COLUMNS]
