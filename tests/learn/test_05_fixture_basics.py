"""レッスン05: フィクスチャの基本（return と yield）

【ポイント】
- @pytest.fixture を付けた関数を定義し、テスト関数の「引数名」に同じ名前を
  書くと、その戻り値が渡される（依存性注入）。データの準備を使い回せる
- 後処理も要るなら return ではなく yield を使う
    yield の前 = 前処理、yield の後ろ = 後処理（テスト失敗時も必ず実行）
- yield の後ろはリソース解放（ファイル削除・DB切断・GPUメモリ解放）に向く

【試すコマンド】
    uv run pytest tests/learn/test_05_fixture_basics.py -v -s
"""
import numpy as np
import pytest

from ml.metrics import accuracy


# ---- return で値を渡すだけのフィクスチャ ------------------------
@pytest.fixture
def sample_labels():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])           # 1つだけ外れ → 4/5
    return y_true, y_pred


def test_accuracy_with_fixture(sample_labels):
    y_true, y_pred = sample_labels               # 引数名 = フィクスチャ名
    assert accuracy(y_true, y_pred) == pytest.approx(0.8)


def test_same_fixture_reused(sample_labels):
    # 同じ準備データを別のテストからも使い回せる（重複が消える）
    y_true, y_pred = sample_labels
    assert len(y_true) == 5


# ---- yield で前処理＋後処理を書くフィクスチャ -------------------
@pytest.fixture
def experiment_dir(tmp_path):
    # --- 前処理 ---
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    print("\n[fixture] 実験ディレクトリ作成")

    yield run_dir                                 # ← この値がテストに渡る

    # --- 後処理（テストが失敗しても必ず実行される）---
    print("[fixture] 実験ディレクトリ後始末")


def test_save_checkpoint(experiment_dir):
    ckpt = experiment_dir / "model.txt"
    ckpt.write_text("weights")
    assert ckpt.exists()
