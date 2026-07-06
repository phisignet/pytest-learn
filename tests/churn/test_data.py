"""データ層のテスト: バリデーションと分割。

実務での狙い:
- 「学習前にデータの前提で早く落ちる」ことを保証する（スキーマ・値域・ラベル）
- 分割の再現性と、train/valid が重複しないことを保証する（リーク防止）
"""
import pandas as pd
import pytest

from ml.churn.data import (
    DataValidationError,
    train_valid_split,
    validate_dataset,
)


@pytest.fixture
def valid_df(raw_df) -> pd.DataFrame:
    """変更して壊すためのコピー（session フィクスチャを汚さない）。"""
    return raw_df.copy()


class TestValidateDataset:
    def test_valid_data_passes(self, valid_df):
        # 正常系: そのまま返る（例外が出ない）
        result = validate_dataset(valid_df)
        assert result is valid_df

    def test_missing_column_raises(self, valid_df):
        broken = valid_df.drop(columns=["monthly_charges"])
        with pytest.raises(DataValidationError, match="missing columns"):
            validate_dataset(broken)

    def test_empty_dataframe_raises(self, valid_df):
        with pytest.raises(DataValidationError, match="empty"):
            validate_dataset(valid_df.iloc[0:0])

    def test_negative_tenure_raises(self, valid_df):
        valid_df.loc[valid_df.index[0], "tenure_months"] = -1
        with pytest.raises(DataValidationError, match="tenure_months"):
            validate_dataset(valid_df)

    def test_invalid_label_raises(self, valid_df):
        valid_df.loc[valid_df.index[0], "churned"] = 2
        with pytest.raises(DataValidationError, match="churned"):
            validate_dataset(valid_df)

    def test_unknown_contract_raises(self, valid_df):
        valid_df.loc[valid_df.index[0], "contract_type"] = "lifetime"
        with pytest.raises(DataValidationError, match="contract_type"):
            validate_dataset(valid_df)


class TestTrainValidSplit:
    @pytest.mark.parametrize("valid_ratio, expected_valid", [(0.25, 100), (0.5, 200), (0.1, 40)])
    def test_split_sizes(self, raw_df, valid_ratio, expected_valid):
        train, valid = train_valid_split(raw_df, valid_ratio=valid_ratio)
        assert len(valid) == expected_valid
        assert len(train) == len(raw_df) - expected_valid

    @pytest.mark.parametrize("bad_ratio", [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_ratio_raises(self, raw_df, bad_ratio):
        with pytest.raises(ValueError, match="valid_ratio"):
            train_valid_split(raw_df, valid_ratio=bad_ratio)

    def test_no_overlap(self, raw_df):
        # train と valid に同じ行が入らない（データリーク防止の要）
        train, valid = train_valid_split(raw_df, valid_ratio=0.2)
        assert set(train.index).isdisjoint(set(valid.index))
        assert len(train) + len(valid) == len(raw_df)

    def test_reproducible_with_same_seed(self, raw_df):
        _, valid1 = train_valid_split(raw_df, seed=123)
        _, valid2 = train_valid_split(raw_df, seed=123)
        pd.testing.assert_frame_equal(valid1, valid2)

    def test_different_seed_gives_different_split(self, raw_df):
        _, valid1 = train_valid_split(raw_df, seed=1)
        _, valid2 = train_valid_split(raw_df, seed=2)
        assert list(valid1.index) != list(valid2.index)
