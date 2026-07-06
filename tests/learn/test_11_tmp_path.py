"""レッスン11: 一時ファイル・ディレクトリ（tmp_path）

【ポイント】
- tmp_path はテストごとに使い捨ての一時ディレクトリ（pathlib.Path）
- 実プロジェクトを汚さずにファイル入出力をテストできる
- / でパス連結できる: tmp_path / "sub" / "file.npy"
- モデルの保存→読み込みの往復（roundtrip）テストに向く
- テスト後、pytest が自動で片付ける

【試すコマンド】
    uv run pytest tests/learn/test_11_tmp_path.py -v
"""
import numpy as np


def save_predictions(path, preds):
    path.write_text("\n".join(map(str, preds)))


def test_save_predictions(tmp_path):
    out_file = tmp_path / "preds.txt"            # / でパス連結
    save_predictions(out_file, [0, 1, 1, 0])
    assert out_file.exists()
    assert out_file.read_text() == "0\n1\n1\n0"


def test_checkpoint_roundtrip(tmp_path):
    # 保存 → 読み込み で元に戻るか（ML で頻出のI/Oテスト）
    ckpt = tmp_path / "weights.npy"
    w = np.array([1.0, 2.0, 3.0])
    np.save(ckpt, w)
    loaded = np.load(ckpt)
    np.testing.assert_allclose(loaded, w)


def test_nested_dir(tmp_path):
    nested = tmp_path / "run" / "artifacts"
    nested.mkdir(parents=True)                   # 途中のディレクトリごと作成
    (nested / "log.txt").write_text("ok")
    assert (nested / "log.txt").read_text() == "ok"
