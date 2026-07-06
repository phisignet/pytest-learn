"""統合テスト: パイプライン全体を通しで検証する。

実務での狙い:
- 単体テストが全部通っても「つなぎ」が壊れていることはある。
  データ検証 → 分割 → 特徴量 → 学習 → 評価 → 保存 → 公開 を一気通貫で流す
- 学習を含み遅いので slow マーカーを付け、普段の実行から除外できるようにする:
      uv run pytest -m "not slow"     # 普段（統合テストを飛ばす）
      uv run pytest -m slow           # 統合テストだけ
- モデル品質のテストは「点」ではなく「下限」で書く（AUC は環境で微妙に揺れる）
"""
import pytest

from ml.churn.data import train_valid_split, validate_dataset
from ml.churn.evaluate import evaluation_report
from ml.churn.features import build_features
from ml.churn.model import ChurnModel
from ml.churn.registry import publish_model


@pytest.mark.slow
def test_end_to_end(raw_df, tmp_path, mocker):
    # 1. データ検証
    df = validate_dataset(raw_df)

    # 2. 分割（リークしない・再現可能な分割は test_data.py で単体検証済み）
    train_df, valid_df = train_valid_split(df, valid_ratio=0.25, seed=0)

    # 3. 特徴量
    X_train, y_train = build_features(train_df), train_df["churned"]
    X_valid, y_valid = build_features(valid_df), valid_df["churned"]

    # 4. 学習
    model = ChurnModel(seed=42).fit(X_train, y_train)

    # 5. 評価: 合成データに入れた信号を学習できていれば AUC は 0.7 を超えるはず
    #    （点で固定せず下限で書く: 環境・ライブラリ更新の揺れに強い）
    report = evaluation_report(y_valid, model.predict_proba(X_valid))
    assert report["auc"] > 0.7
    assert report["accuracy"] > 0.6

    # 6. 保存 → 読み込みの往復
    model_path = tmp_path / "churn_model.joblib"
    model.save(model_path)
    loaded = ChurnModel.load(model_path)

    # 7. レジストリへ公開（外部通信だけモック）
    mocker.patch("ml.churn.registry._http_post", return_value=200)
    model_id = publish_model(
        model_path, "https://registry.example.com", "churn", "1.0.0"
    )
    assert model_id == "churn:1.0.0"

    # 読み込んだモデルでも評価が同じであること（デプロイ物 = 学習物 の確認）
    report_loaded = evaluation_report(y_valid, loaded.predict_proba(X_valid))
    assert report_loaded["auc"] == pytest.approx(report["auc"])
