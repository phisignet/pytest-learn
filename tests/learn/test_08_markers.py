"""レッスン08: マーカー（skip / skipif / xfail / 自作）

【ポイント】
- @pytest.mark.xxx でテストにタグや振る舞いを付ける
    skip     … 無条件でスキップ
    skipif   … 条件を満たすときだけスキップ（GPU無し環境など）
    xfail    … 失敗する想定。失敗→xfail、成功→xpass として区別される
- 自作マーカーは登録が必要（このプロジェクトは pyproject.toml に登録済み）
    実行時に -m で絞り込める:
        uv run pytest -m slow          # slow だけ
        uv run pytest -m "not slow"    # slow 以外（CI の高速チェック向け）

【試すコマンド】
    uv run pytest tests/learn/test_08_markers.py -v
    uv run pytest tests/learn/test_08_markers.py -v -m "not slow"
"""
import sys

import pytest


def test_normal():
    assert 1 + 1 == 2


@pytest.mark.skip(reason="前処理APIの仕様変更待ちのため一時的に無効化")
def test_not_ready():
    assert False                                 # スキップされるので実行されない


@pytest.mark.skipif(sys.platform == "win32", reason="CUDA が無い環境ではスキップ")
def test_gpu_only():
    # GPU が要るテストの想定。win32 ではスキップされる
    assert True


@pytest.mark.xfail(reason="既知のバグ。修正予定なので失敗しても赤にしない")
def test_known_bug():
    assert 1 == 2                                # 失敗するが xfail 扱い


@pytest.mark.slow
def test_full_training_pipeline():
    # 本来は時間のかかる学習テスト。-m "not slow" で普段は除外できる
    total = sum(range(1000))
    assert total == 499500


@pytest.mark.fast
def test_single_forward():
    assert (2 * 3) == 6
