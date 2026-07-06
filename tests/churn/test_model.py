"""モデル層のテスト: 推論の契約・永続化・外部レジストリ（モック）。

実務での狙い:
- 出力の shape / 値域（確率は [0, 1]）という「推論の契約」を守る
- save → load の往復で挙動が変わらない（デプロイ前の最重要チェック）
- 未学習での推論はフェイルファスト
- 外部サービス（レジストリ）はモックして、ネットワークなしで検証
"""
import numpy as np
import pytest

from ml.churn.features import build_features
from ml.churn.model import ChurnModel, NotFittedError
from ml.churn.registry import publish_model


class TestPredictContract:
    def test_proba_shape(self, trained_model, split_data):
        _, valid_df = split_data
        X = build_features(valid_df)
        proba = trained_model.predict_proba(X)
        assert proba.shape == (len(valid_df),)

    def test_proba_range(self, trained_model, split_data):
        _, valid_df = split_data
        proba = trained_model.predict_proba(build_features(valid_df))
        assert (proba >= 0.0).all() and (proba <= 1.0).all()
        assert not np.isnan(proba).any()

    def test_unfitted_model_raises(self, feature_row):
        model = ChurnModel()
        with pytest.raises(NotFittedError, match="not fitted"):
            model.predict_proba(feature_row)


class TestPersistence:
    def test_save_load_roundtrip(self, trained_model, feature_row, tmp_path):
        # 保存 → 読み込み → 予測が完全一致（デプロイ前の必須チェック）
        path = tmp_path / "churn_model.joblib"
        trained_model.save(path)
        loaded = ChurnModel.load(path)
        np.testing.assert_allclose(
            loaded.predict_proba(feature_row),
            trained_model.predict_proba(feature_row),
        )

    def test_load_wrong_object_raises(self, tmp_path):
        # ChurnModel 以外が保存されていたら型エラーで落とす
        import joblib

        path = tmp_path / "not_a_model.joblib"
        joblib.dump({"just": "a dict"}, path)
        with pytest.raises(TypeError, match="expected ChurnModel"):
            ChurnModel.load(path)


class TestReproducibility:
    def test_same_seed_same_predictions(self, split_data, feature_row):
        train_df, _ = split_data
        X, y = build_features(train_df), train_df["churned"]
        m1 = ChurnModel(seed=42).fit(X, y)
        m2 = ChurnModel(seed=42).fit(X, y)
        np.testing.assert_allclose(
            m1.predict_proba(feature_row), m2.predict_proba(feature_row)
        )


class TestRegistry:
    """外部HTTP依存（_http_post）をモックに差し替えてテストする。"""

    def test_publish_success(self, mocker, tmp_path):
        # 200 を返す偽のHTTPクライアントに差し替え → 実際の通信は起きない
        mock_post = mocker.patch("ml.churn.registry._http_post", return_value=200)
        model_file = tmp_path / "model.joblib"
        model_file.write_bytes(b"dummy weights")

        model_id = publish_model(
            model_file, "https://registry.example.com", "churn", "1.2.0"
        )

        assert model_id == "churn:1.2.0"
        mock_post.assert_called_once()
        url, payload = mock_post.call_args.args
        assert url == "https://registry.example.com/models"     # 正しいエンドポイント
        assert payload["name"] == "churn"
        assert payload["size_bytes"] == len(b"dummy weights")   # 正しいメタデータ

    def test_publish_server_error_raises(self, mocker, tmp_path):
        # 異常系: レジストリが 500 を返したら RuntimeError
        mocker.patch("ml.churn.registry._http_post", return_value=500)
        model_file = tmp_path / "model.joblib"
        model_file.write_bytes(b"x")

        with pytest.raises(RuntimeError, match="status 500"):
            publish_model(model_file, "https://registry.example.com", "churn", "1.0.0")

    def test_publish_missing_file_raises(self, mocker, tmp_path):
        # ファイルが無ければHTTPを呼ぶ前に落ちる（モックが呼ばれないことも検証）
        mock_post = mocker.patch("ml.churn.registry._http_post")
        with pytest.raises(FileNotFoundError):
            publish_model(tmp_path / "missing.joblib",
                          "https://registry.example.com", "churn", "1.0.0")
        mock_post.assert_not_called()
