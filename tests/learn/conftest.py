"""conftest.py — テスト全体で共有する設定とフィクスチャ。

このファイルは import 不要で、同じディレクトリ以下の全テストから
自動的に見える（レッスン 06 と 08 で詳しく扱う）。ここでは
「全テスト共通のダミーデータ」と「乱数シードの固定」を定義しておく。
"""
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fix_random_seed():
    """全テストの前に乱数シードを固定する（再現性の担保）。

    autouse=True なので、テスト側で引数に書かなくても自動適用される。
    ML テストでは乱数を使う場面が多いので、これは定番の下ごしらえ。
    """
    np.random.seed(0)
    yield


@pytest.fixture(scope="session")
def sample_dataset():
    """全テスト共通のダミーデータセット。session スコープなので1回だけ作る。"""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y
