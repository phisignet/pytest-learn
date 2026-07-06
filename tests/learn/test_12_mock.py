"""レッスン12: モック（pytest-mock の mocker）

【ポイント】
- 外部依存（API・GPU・DB・重いモデルのロード）を「偽物」に差し替えるのがモック
- pytest-mock を入れると mocker フィクスチャが使える（後始末が自動）
- ML で最重要な用途:「外部/重い処理を呼ばずにロジックだけテストする」
- 基本形:
    mocker.patch("差し替える場所のパス", return_value=...)   # 戻り値を固定
    mocker.patch("...", side_effect=func_or_exc)             # 入力で変える/例外
- モックは「呼ばれ方」も記録している（正しい引数で呼んだかを検証できる）

ここでは ml/predictor.py の load_weights()（本来はディスクI/O）をモックし、
実ファイルなしで Predictor のロジックをテストする。

【試すコマンド】
    uv run pytest tests/learn/test_12_mock.py -v -s
"""
import numpy as np
import pytest

from ml.predictor import Predictor


def test_return_value(mocker):
    # load_weights を「常に固定の重みを返す偽物」に差し替える → 実ファイル不要
    fake_w = np.array([1.0, 0.0, 0.0])
    mocker.patch("ml.predictor.load_weights", return_value=fake_w)

    predictor = Predictor("path/does/not/exist.npy")   # I/O は起きない
    y = predictor.predict([[2.0, 9.0, 9.0]])
    # x @ w = 2*1 + 9*0 + 9*0 = 2.0
    np.testing.assert_allclose(y, [2.0])


def test_side_effect_function(mocker):
    # side_effect に関数を渡すと、呼び出しごとにその関数が実行される
    def fake_loader(path):
        # 渡されたパス名に応じて別の重みを返す（入力依存の偽物）
        return np.ones(3) if "big" in str(path) else np.zeros(3)

    mocker.patch("ml.predictor.load_weights", side_effect=fake_loader)

    assert Predictor("big.npy").w.sum() == 3.0
    assert Predictor("small.npy").w.sum() == 0.0


def test_side_effect_exception(mocker):
    # 異常系: ロードが失敗したときの挙動をテストする
    mocker.patch("ml.predictor.load_weights",
                 side_effect=FileNotFoundError("weights not found"))
    with pytest.raises(FileNotFoundError) as exc_info:
        Predictor("missing.npy")
    assert "not found" in str(exc_info.value)


def test_call_args(mocker):
    # モックは呼ばれ方を記録している。正しい引数で呼んだかを検証できる
    mock_loader = mocker.patch("ml.predictor.load_weights",
                               return_value=np.zeros(3))
    Predictor("model_v2.npy")

    mock_loader.assert_called_once()                     # ちょうど1回呼ばれた
    mock_loader.assert_called_once_with("model_v2.npy")  # この引数で呼ばれた
    print("\ncall_args_list:", mock_loader.call_args_list)
    assert mock_loader.call_args.args[0] == "model_v2.npy"
