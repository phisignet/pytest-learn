# 実務に近い実装例（tests/churn/）

チャーン（解約）予測のミニプロジェクト `ml/churn/` に対する、
**実務スタイルのテスト一式**。`tests/learn/` のレッスンで学んだ機能が、
実際のプロジェクト構成の中でどう組み合わさるかを見せる。

## 構成（ソースとテストを1:1で対応させる）

```
ml/churn/                 tests/churn/
├── data.py        ←──   ├── test_data.py        バリデーション・分割
├── features.py    ←──   ├── test_features.py    特徴量の契約・境界値
├── model.py       ←──   ├── test_model.py       推論の契約・永続化・レジストリ(モック)
├── evaluate.py    ←──   ├── test_evaluate.py    指標の既知ケース検証
└── registry.py           ├── test_pipeline.py    E2E統合テスト（slow マーカー）
                          └── conftest.py         データ→特徴量→学習済みモデルの段階フィクスチャ
```

## 実行

```bash
uv run pytest tests/churn                 # 全部
uv run pytest tests/churn -m "not slow"   # 統合テスト（学習あり）を除く普段用
uv run pytest tests/churn -m slow -v      # 統合テストだけ
```

## ここで見せている実務パターン（→ 対応レッスン）

| パターン | 場所 | レッスン |
|---|---|---|
| 合成データをテスト内で生成（実データを持ち込まない） | `conftest.py: make_raw_df` | 06, 08 |
| データ→特徴量→学習済みモデルの段階フィクスチャ | `conftest.py` | 05, 06 |
| 重い学習は `scope="session"` で1回だけ | `conftest.py: trained_model` | 06 |
| session フィクスチャを汚さないためのコピー用フィクスチャ | `test_data.py: valid_df` | 05 |
| バリデーションの異常系を `raises` + `match` で網羅 | `test_data.py` | 09 |
| データリーク防止（train/valid の重複なし）の検証 | `test_data.py: test_no_overlap` | — |
| 出力カラムの「契約」テスト（名前・順序・NaNなし） | `test_features.py` | 02 |
| 境界値を `parametrize` + `ids` で両側から | `test_features.py: is_new_customer` | 07 |
| 推論の契約（shape・確率の値域・未学習エラー） | `test_model.py` | 02, 09 |
| save→load の往復一致（roundtrip） | `test_model.py` + `tmp_path` | 11 |
| seed 固定の再現性 | `test_model.py: TestReproducibility` | 06 |
| 外部HTTP（レジストリ）をモック、呼ばれ方まで検証 | `test_model.py: TestRegistry` | 12 |
| 「モックが呼ばれない」ことの検証（`assert_not_called`） | `test_model.py` | 12 |
| 指標を手計算できる既知ケースで固定 | `test_evaluate.py` | 07 |
| E2E統合テストは `slow` マーカーで分離 | `test_pipeline.py` | 08 |
| モデル品質は「点」でなく「下限」で assert（AUC > 0.7） | `test_pipeline.py` | 02 |

## 実務での考え方（要点）

- **テストはソース構成をミラーする**。どこが壊れたか一目で分かる
- **単体は速く・網羅的に、統合は少数を通しで**。統合はマーカーで分離して CI を速く保つ
- **外部依存（ネットワーク・実データ・実ファイル）はテストに持ち込まない**。
  モック・合成データ・tmp_path で代替する
- **「契約」をテストする**: カラム構成、shape、値域、レポートのキー。
  実装の内部ではなく、下流が依存する外形を固定するとリファクタに強い
