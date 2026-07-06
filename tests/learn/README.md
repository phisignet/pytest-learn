# pytest を段階的に学ぶ（tests/learn/）

このディレクトリのスクリプトを **番号順に読みながら実行**すると、pytest の使い方が
一通り身につくように構成している。読者は Python に慣れた機械学習エンジニアを想定。
pytest の使い方自体は一般的な題材で説明し、**実装例は機械学習の場面**を使う。

## 準備

```bash
uv sync
```

テスト対象コードはリポジトリ直下の `ml/` にある（`preprocessing.py` /
`metrics.py` / `predictor.py`）。モックのレッスンで使う `translator` の例は
`ml/predictor.py` に置き換えている（外部ネットワーク不要で動くようにするため）。

## 実行方法

```bash
# 全レッスンを実行
uv run pytest tests/learn

# レッスンを1つずつ、詳細 + print 表示で
uv run pytest tests/learn/test_01_first_test.py -v -s

# 特定の1関数だけ
uv run pytest tests/learn/test_01_first_test.py::test_min_max_scale_range -v
```

## カリキュラム（この順に読む）

| # | ファイル | 学ぶこと |
|---|---|---|
| 01 | `test_01_first_test.py` | `test_` 命名と `assert`、最初のテスト |
| 02 | `test_02_assert_numeric.py` | 浮動小数・numpy配列の比較（`approx` / `assert_allclose`） |
| 03 | `test_03_class.py` | クラスでテストをまとめる（`Test` 始まり） |
| 04 | `test_04_setup_teardown.py` | 前処理・後処理（xunit スタイル） |
| 05 | `test_05_fixture_basics.py` | フィクスチャの基本（`return` / `yield`） |
| 06 | `test_06_fixture_scope_autouse.py` | スコープと autouse |
| 07 | `test_07_parametrize.py` | パラメータ化テスト |
| 08 | `test_08_markers.py` | マーカー（skip / skipif / xfail / 自作） |
| 09 | `test_09_exceptions.py` | 例外のテスト（`pytest.raises`） |
| 10 | `test_10_capsys.py` | 標準出力のテスト（`capsys`） |
| 11 | `test_11_tmp_path.py` | 一時ファイル・ディレクトリ（`tmp_path`） |
| 12 | `test_12_mock.py` | モック（`pytest-mock` の `mocker`） |
| 13 | `test_13_snapshot_and_time.py` | スナップショット / 時刻固定（発展・要追加パッケージ） |
| 14 | `test_14_monkeypatch.py` | `monkeypatch`（環境変数・属性・dict の一時差し替え） |
| 15 | `test_15_caplog_and_warns.py` | ログの検証（`caplog`）と警告の検証（`pytest.warns`） |
| 16 | `test_16_fixture_advanced.py` | フィクスチャ応用（`params=` / ファクトリ / `tmp_path_factory`） |
| 17 | `test_17_mock_advanced.py` | モック応用（`side_effect` リスト / `spy` / `patch.object` / `patch.dict`） |

各ファイルの冒頭コメントに「このレッスンのポイント」と「試してみるコマンド」を書いてある。
まず読んで、`-v -s` を付けて実行し、出力と照らし合わせるのがおすすめ。

より広い解説や ML でのテスト設計の勘所は、リポジトリ直下の `pytest_guide.md` を参照。

## 次のステップ

レッスンを終えたら、`tests/churn/` にある**実務に近い実装例**
（チャーン予測ミニプロジェクトのテスト一式）を読む。ここで学んだ各機能が、
実際のプロジェクト構成の中でどう組み合わさるかが分かる。
