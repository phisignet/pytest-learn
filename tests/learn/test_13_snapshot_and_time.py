"""レッスン13（発展）: スナップショットテスト と 時刻の固定

このレッスンだけ追加パッケージが要る。未インストールなら自動でスキップされる
ので、まずは読んで理解するだけでもよい。導入すると実際に動く。

    uv add --dev syrupy           # スナップショット
    uv add --dev pytest-freezer   # 時刻の固定（freeze_time マーカー）

--------------------------------------------------------------------
■ スナップショットテスト（snapshot）
【ポイント】
- 出力が複雑で期待値を手書きしづらいとき、初回の出力を保存し、次回以降
  それと一致するかを比較する（回帰テスト）。モデル設定辞書などに向く
- 使い方: テストで snapshot フィクスチャを使い、assert 対象 == snapshot
    初回:   uv run pytest --snapshot-update   でスナップショットを生成
    以降:   uv run pytest                     で保存済みと比較
    変更時: 再度 --snapshot-update で更新
- 重要: スナップショット保存先（__snapshots__）は必ず git 管理に含めること。
  含めないと clone 先で「比較対象なし＝常に成功」になり、壊れても気づけない

■ 時刻の固定（freeze_time）
【ポイント】
- 「現在時刻」に依存するコードは、そのままだと結果が毎回変わる
- @pytest.mark.freeze_time("...") を付けると、その関数内だけ時刻が固定される
- ログのタイムスタンプ・実験ランID・TTL・スケジューラのテストに使う

【試すコマンド】（パッケージ導入後）
    uv run pytest tests/learn/test_13_snapshot_and_time.py -v
"""
import datetime

import pytest


def build_model_config(lr, n_layers):
    return {
        "optimizer": "adam",
        "lr": lr,
        "n_layers": n_layers,
        "activation": "relu",
    }


def make_run_id():
    # 実行時刻からラン ID を作る（実験管理でよくあるパターン）
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")


# ---- スナップショット（syrupy 未導入なら skip）-------------------
def test_model_config_snapshot(snapshot):
    # snapshot フィクスチャは syrupy が提供する。未導入なら下の理由でスキップされる
    config = build_model_config(0.01, 3)
    assert config == snapshot


# ---- 時刻の固定（pytest-freezer 未導入なら skip）----------------
@pytest.mark.freeze_time("2026-07-07 09:00:00")
def test_make_run_id():
    assert make_run_id() == "run_20260707_090000"


# ---- 未導入プラグインを検知して丁寧にスキップする仕掛け ---------
def _plugin_installed(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


collect_ignore = []
if not _plugin_installed("syrupy"):
    # snapshot フィクスチャが無い環境では test_model_config_snapshot を無効化
    test_model_config_snapshot = pytest.mark.skip(
        reason="syrupy 未インストール（uv add --dev syrupy で有効化）"
    )(test_model_config_snapshot)

if not _plugin_installed("freezegun"):
    # pytest-freezer は freezegun に依存。無ければ freeze_time マーカーが効かない
    test_make_run_id = pytest.mark.skip(
        reason="pytest-freezer 未インストール（uv add --dev pytest-freezer で有効化）"
    )(test_make_run_id)
