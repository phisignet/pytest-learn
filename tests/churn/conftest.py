"""tests/churn 共有のフィクスチャ。

実務のテストでは conftest.py に「データ → 特徴量 → 学習済みモデル」の
段階的なフィクスチャを積み上げるのが定番。下流のフィクスチャは上流を
引数に取るだけで再利用でき、重いもの（学習）は scope を広げて1回にする。
"""
import numpy as np
import pandas as pd
import pytest

from ml.churn.data import VALID_CONTRACTS, train_valid_split
from ml.churn.features import build_features
from ml.churn.model import ChurnModel


def make_raw_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """信号入りの合成チャーンデータを作る（テスト専用のデータ生成器）。

    実務でも「本物のデータを持ち込まず、性質を再現した合成データを
    テスト内で生成する」のは定石（個人情報・サイズ・再現性の問題を避ける）。
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, size=n)
    charges = rng.uniform(20.0, 120.0, size=n)
    contract = rng.choice(VALID_CONTRACTS, size=n, p=[0.5, 0.3, 0.2])

    # 「在籍が短い・支払いが多い・月契約」ほど解約しやすい、という信号を入れる
    logit = 0.8 - 0.06 * tenure + 0.015 * (charges - 70.0)
    logit = logit + np.where(contract == "month-to-month", 1.2, 0.0)
    p_churn = 1.0 / (1.0 + np.exp(-logit))
    churned = rng.binomial(1, p_churn)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": charges,
            "contract_type": contract,
            "churned": churned,
        }
    )


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """全テスト共通の生データ（session スコープ: 1回だけ生成）。"""
    return make_raw_df()


@pytest.fixture(scope="session")
def split_data(raw_df):
    """(train_df, valid_df) のタプル。"""
    return train_valid_split(raw_df, valid_ratio=0.25, seed=0)


@pytest.fixture(scope="session")
def trained_model(split_data) -> ChurnModel:
    """学習済みモデル（重いのでテスト全体で1回だけ学習する）。"""
    train_df, _ = split_data
    X = build_features(train_df)
    y = train_df["churned"]
    return ChurnModel(seed=42).fit(X, y)


@pytest.fixture
def feature_row() -> pd.DataFrame:
    """推論の入力に使える 1 行の特徴量（毎テスト新しく作る）。"""
    df = pd.DataFrame(
        {
            "tenure_months": [3],
            "monthly_charges": [95.0],
            "contract_type": ["month-to-month"],
            "churned": [0],
        }
    )
    return build_features(df)
