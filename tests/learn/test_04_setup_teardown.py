"""レッスン04: 前処理・後処理（xunit スタイル）

【ポイント】
- 決まった名前の関数を定義すると、テストの前後に自動で呼ばれる
    setup_module / teardown_module … ファイル全体で1回
    setup_function / teardown_function … 各テスト関数の前後
    setup_class / teardown_class … クラスで1回（@classmethod）
    setup_method / teardown_method … クラス内メソッドの前後
- 出力を見るには -s が必要（print が捕捉されて隠れるため）
- これは今も使えるが、次レッスンの「フィクスチャ」の方が柔軟で推奨

【試すコマンド】
    uv run pytest tests/learn/test_04_setup_teardown.py -v -s
"""


# ---- 関数レベル -------------------------------------------------
def setup_function(function):
    print(f"\n[setup_function] before {function.__name__}")


def teardown_function(function):
    print(f"[teardown_function] after {function.__name__}")


def test_a():
    print("  -> test_a 本体")


def test_b():
    print("  -> test_b 本体")


# ---- クラスレベル -----------------------------------------------
class TestModelLifecycle:
    @classmethod
    def setup_class(cls):
        print("\n[setup_class] 重いモデルを1回だけロードする想定")

    @classmethod
    def teardown_class(cls):
        print("[teardown_class] 後片付け")

    def setup_method(self, method):
        print(f"  [setup_method] before {method.__name__}")

    def test_predict_1(self):
        print("    -> 推論テスト1")

    def test_predict_2(self):
        print("    -> 推論テスト2")
