"""レッスン14: monkeypatch（環境変数・属性・dict の一時的な差し替え）

【ポイント】
- monkeypatch は pytest 組み込みのフィクスチャ（追加パッケージ不要）
- 「テストの間だけ差し替え、終わったら自動で元に戻す」のが仕事
    monkeypatch.setenv / delenv   … 環境変数
    monkeypatch.setattr           … モジュール・クラスの属性や関数
    monkeypatch.setitem / delitem … dict の要素
    monkeypatch.chdir             … カレントディレクトリ
- mocker（レッスン12）との使い分けの目安:
    設定・環境（env, dict, パス）を変えたい     → monkeypatch
    呼び出しの記録・検証（呼ばれた回数/引数）が欲しい → mocker

【試すコマンド】
    uv run pytest tests/learn/test_14_monkeypatch.py -v
"""
import os

import pytest

from ml.preprocessing import min_max_scale


# ---- テスト対象: 環境変数から設定を読む関数（実務で頻出のパターン） ----
def get_model_dir() -> str:
    """モデルの保存先を環境変数から取得する。未設定ならデフォルト。"""
    return os.environ.get("MODEL_DIR", "/default/models")


def get_batch_size() -> int:
    raw = os.environ.get("BATCH_SIZE")
    if raw is None:
        raise KeyError("BATCH_SIZE is not set")
    return int(raw)


# ---- setenv / delenv: 環境変数の差し替え ---------------------------------
def test_setenv(monkeypatch):
    # このテストの間だけ MODEL_DIR を設定（終わったら自動で元に戻る）
    monkeypatch.setenv("MODEL_DIR", "/tmp/experiment_42")
    assert get_model_dir() == "/tmp/experiment_42"


def test_env_default(monkeypatch):
    # 逆に「未設定のとき」をテストしたいなら delenv（無くてもエラーにしない raising=False）
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert get_model_dir() == "/default/models"


def test_missing_env_raises(monkeypatch):
    monkeypatch.delenv("BATCH_SIZE", raising=False)
    with pytest.raises(KeyError):
        get_batch_size()


# ---- setattr: 関数・属性の差し替え（mocker.patch の軽量版） --------------
def test_setattr(monkeypatch):
    import ml.preprocessing

    # min_max_scale を「そのまま返すだけ」の偽物に差し替える
    monkeypatch.setattr(ml.preprocessing, "min_max_scale", lambda x: x)
    assert ml.preprocessing.min_max_scale([1, 2, 3]) == [1, 2, 3]


def test_setattr_restored():
    # 前のテストで差し替えた min_max_scale が自動で元に戻っている確認
    assert min_max_scale([0, 10]).tolist() == [0.0, 1.0]


# ---- setitem: dict の要素だけ差し替え -------------------------------------
HYPERPARAMS = {"lr": 0.01, "batch_size": 32}      # モジュールレベルの設定 dict


def test_setitem(monkeypatch):
    # dict 全体でなく、1要素だけテスト中差し替える
    monkeypatch.setitem(HYPERPARAMS, "lr", 0.5)
    assert HYPERPARAMS["lr"] == 0.5
    assert HYPERPARAMS["batch_size"] == 32        # 他のキーはそのまま


def test_setitem_restored():
    assert HYPERPARAMS["lr"] == 0.01              # 自動で元に戻っている
