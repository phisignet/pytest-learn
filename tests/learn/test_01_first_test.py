"""レッスン01: 最初のテスト（test_ 命名 と assert）

【ポイント】
- ファイル名は test_*.py、関数名は test_ で始める → pytest が自動で見つける
- 検証は assert 文で書く。False なら失敗、True（例外なし）なら成功
- 特別なメソッドは不要。Python の式をそのまま書けるのが pytest の強み

【試すコマンド】
    uv run pytest tests/learn/test_01_first_test.py -v
"""
from ml.preprocessing import min_max_scale


def test_min_max_scale_range():
    # Arrange（準備）→ Act（実行）→ Assert（検証）の3段で書くと読みやすい
    scaled = min_max_scale([0, 5, 10])          # Act
    assert scaled.min() == 0.0                   # Assert
    assert scaled.max() == 1.0


def test_min_max_scale_constant_input():
    # 全部同じ値のとき、ゼロ除算せず 0 が返ることを確認
    scaled = min_max_scale([3, 3, 3])
    assert (scaled == 0).all()


def test_min_max_scale_middle_value():
    # 中央の値は 0.5 になるはず
    scaled = min_max_scale([0, 5, 10])
    assert scaled[1] == 0.5
