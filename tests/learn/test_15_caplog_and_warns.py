"""レッスン15: ログの検証（caplog）と 警告の検証（pytest.warns）

【ポイント】
- 実務のコードは print ではなく logging を使う → その検証は capsys ではなく
  caplog（組み込みフィクスチャ）で行う
    caplog.records   … LogRecord のリスト（levelname, message などを持つ）
    caplog.text      … 全ログを結合した文字列
    caplog.set_level … 捕捉する最低レベルを指定（DEBUG まで拾う等）
- 警告（DeprecationWarning など）は pytest.warns で検証する
  pytest.raises の警告版。ML ライブラリは deprecation が多いので出番が多い

【試すコマンド】
    uv run pytest tests/learn/test_15_caplog_and_warns.py -v
"""
import logging
import warnings

import numpy as np
import pytest

logger = logging.getLogger("trainer")


# ---- テスト対象: logging を使う学習ループ ---------------------------------
def train_epoch(epoch: int, loss: float) -> None:
    logger.info("epoch=%d loss=%.3f", epoch, loss)
    if loss > 10.0:
        logger.warning("loss is exploding: %.1f", loss)


def normalize_legacy(x):
    """旧API。使うと DeprecationWarning を出す。"""
    warnings.warn(
        "normalize_legacy is deprecated; use min_max_scale",
        DeprecationWarning,
        stacklevel=2,
    )
    x = np.asarray(x, dtype=float)
    return x / x.max()


# ---- caplog: ログの検証 ----------------------------------------------------
def test_info_log(caplog):
    with caplog.at_level(logging.INFO, logger="trainer"):
        train_epoch(1, 0.523)
    assert "epoch=1 loss=0.523" in caplog.text


def test_warning_on_loss_explosion(caplog):
    with caplog.at_level(logging.INFO, logger="trainer"):
        train_epoch(2, 99.9)

    # records で「レベル」まで検証できる（文字列一致より頑丈）
    warnings_ = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_) == 1
    assert "exploding" in warnings_[0].message


def test_no_warning_when_loss_is_normal(caplog):
    # 「警告が出ないこと」も大事なテスト
    with caplog.at_level(logging.INFO, logger="trainer"):
        train_epoch(3, 0.1)
    assert all(r.levelno < logging.WARNING for r in caplog.records)


# ---- pytest.warns: 警告の検証 ----------------------------------------------
def test_legacy_function_warns():
    # このブロック内で DeprecationWarning が出なければテスト失敗
    with pytest.warns(DeprecationWarning, match="use min_max_scale"):
        result = normalize_legacy([1.0, 2.0, 4.0])
    np.testing.assert_allclose(result, [0.25, 0.5, 1.0])   # 警告は出るが動作は正しい


def test_new_function_does_not_warn(recwarn):
    # recwarn フィクスチャ: ブロックなしで「出た警告の一覧」を受け取る
    from ml.preprocessing import min_max_scale

    min_max_scale([1.0, 2.0, 4.0])
    assert len(recwarn) == 0                               # 警告ゼロを保証
