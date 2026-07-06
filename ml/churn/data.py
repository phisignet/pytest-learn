"""データのバリデーションと分割。

実務では「学習に入る前にデータの前提を検証して早く落とす」のが定石。
その前提（スキーマ・値域・ラベルの妥当性）はそのままテスト項目になる。
"""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["tenure_months", "monthly_charges", "contract_type", "churned"]
VALID_CONTRACTS = ["month-to-month", "one-year", "two-year"]


class DataValidationError(ValueError):
    """データが前提を満たさないときに投げる例外。"""


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """データセットの前提条件を検証する。問題があれば DataValidationError。

    検証項目:
    - 必須カラムがそろっている
    - 空でない
    - tenure_months が非負
    - churned が 0/1 のみ
    - contract_type が既知のカテゴリのみ
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    if len(df) == 0:
        raise DataValidationError("dataset is empty")

    if (df["tenure_months"] < 0).any():
        raise DataValidationError("tenure_months must be >= 0")

    labels = set(df["churned"].unique().tolist())
    if not labels <= {0, 1}:
        raise DataValidationError(f"churned must be 0/1, got {sorted(labels)}")

    unknown = set(df["contract_type"].unique().tolist()) - set(VALID_CONTRACTS)
    if unknown:
        raise DataValidationError(f"unknown contract_type: {sorted(unknown)}")

    return df


def train_valid_split(
    df: pd.DataFrame, valid_ratio: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """行をシャッフルして (train, valid) に分割する。

    valid_ratio が (0, 1) の範囲外なら ValueError。
    seed を固定すれば分割は再現可能。
    """
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    n_valid = int(len(df) * valid_ratio)
    valid_idx, train_idx = indices[:n_valid], indices[n_valid:]
    return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()
