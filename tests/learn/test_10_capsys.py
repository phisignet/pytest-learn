"""レッスン10: 標準出力のテスト（capsys）

【ポイント】
- capsys は print などの標準出力／標準エラー出力を捕捉する組み込みフィクスチャ
- テスト関数の引数に capsys を書き、out, err = capsys.readouterr() で取り出す
    out … 標準出力、err … 標準エラー出力
- 学習ループのログや CLI の表示内容の検証に使う
- 注意: capsys は「検証のために捕捉」する。実行時の -s（表示）とは目的が別

【試すコマンド】
    uv run pytest tests/learn/test_10_capsys.py -v
"""


def train_report(epoch, loss):
    print(f"epoch={epoch} loss={loss:.3f}")


def test_train_report(capsys):
    train_report(1, 0.1234)
    out, err = capsys.readouterr()
    assert out == "epoch=1 loss=0.123\n"
    assert err == ""


def test_multiple_lines(capsys):
    for e in range(3):
        train_report(e, 1.0 / (e + 1))
    out, _ = capsys.readouterr()
    lines = out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0] == "epoch=0 loss=1.000"
