"""レッスン09: 例外のテスト（pytest.raises）

【ポイント】
- 「この入力なら例外を投げるべき」を検証する
- with pytest.raises(例外): のブロック内でその例外が出れば合格、出なければ失敗
- 複数の例外を許容: pytest.raises((ValueError, TypeError))
- 例外オブジェクトを受け取ってメッセージも検証できる（as で束縛 / match= で正規表現）

【試すコマンド】
    uv run pytest tests/learn/test_09_exceptions.py -v
"""
import pytest

from ml.metrics import check_same_length
from ml.preprocessing import train_test_split


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        check_same_length([0, 1, 1], [0, 1])     # 長さ違い → ValueError


def test_error_message_with_as():
    # as で例外オブジェクトを受け取り、メッセージ内容まで確認する
    with pytest.raises(ValueError) as exc_info:
        check_same_length([0, 1, 1], [0, 1])
    assert "mismatch" in str(exc_info.value)


def test_error_message_with_match():
    # match= なら正規表現で1行で検証できる
    with pytest.raises(ValueError, match="must be in"):
        train_test_split([1, 2, 3], test_ratio=1.5)


def test_valid_ratio_does_not_raise():
    # 正常系: 例外が出ないことも大事なテスト
    train, test = train_test_split([1, 2, 3, 4, 5], test_ratio=0.4)
    assert len(test) == 2
    assert len(train) == 3
