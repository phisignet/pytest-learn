"""レッスン02: 数値・配列の比較（approx と np.testing）

【ポイント】
- 浮動小数を == で比較すると誤差で落ちる → pytest.approx を使う
- numpy 配列は np.testing.assert_allclose で要素ごとに誤差込み比較する
- 形状(shape)や型(dtype)は素の == で比較してよい

【試すコマンド】
    uv run pytest tests/learn/test_02_assert_numeric.py -v
"""
import numpy as np
import pytest

from ml.preprocessing import min_max_scale


def test_float_needs_approx():
    # 素の == は誤差で False になる
    assert 0.1 + 0.2 != 0.3
    # approx なら誤差を許容して比較できる
    assert 0.1 + 0.2 == pytest.approx(0.3)


def test_accuracy_with_tolerance():
    acc = 0.9999999
    assert acc == pytest.approx(1.0, abs=1e-3)   # 絶対誤差 1e-3 まで許容


def test_array_allclose():
    result = min_max_scale([0, 5, 10])
    expected = np.array([0.0, 0.5, 1.0])
    # 配列全体を誤差込みで比較。落ちたときは差分も分かりやすく表示される
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_shape_and_dtype():
    result = min_max_scale(np.zeros((3, 4)))
    assert result.shape == (3, 4)                # 形状は == でOK
    assert result.dtype == np.float64
