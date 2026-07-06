"""レッスン17: モックの応用（side_effect リスト・spy・patch.object・patch.dict）

【ポイント】
1. side_effect にリストを渡す → 呼び出しごとに順番に消費される
   「2回失敗して3回目に成功」のようなリトライ処理のテストに最適
2. mocker.spy … 本物を実行しつつ、呼ばれ方だけ記録する
   「差し替えたくはないが、呼ばれた回数・引数は検証したい」とき
3. mocker.patch.object … オブジェクト（やクラス）を直接指定して差し替える
   文字列パスを書かずに済むので、リネームに強くタイポも防げる
4. mocker.patch.dict … dict の中身をテスト中だけ差し替える（os.environ にも使える）

【試すコマンド】
    uv run pytest tests/learn/test_17_mock_advanced.py -v
"""
import numpy as np
import pytest

import ml.predictor
import ml.preprocessing
from ml.predictor import Predictor


# ---- テスト対象: リトライつきダウンローダ（実務で頻出） --------------------
def download_with_retry(fetch, max_retries: int = 3):
    """fetch() を最大 max_retries 回試す。全部失敗したら最後の例外を投げ直す。"""
    for attempt in range(max_retries):
        try:
            return fetch()
        except ConnectionError:
            if attempt == max_retries - 1:
                raise
    raise AssertionError("unreachable")


# ---- 1. side_effect にリスト: リトライのテスト ------------------------------
def test_retry_succeeds_on_third_attempt(mocker):
    # 1回目・2回目は ConnectionError、3回目は成功、という台本を渡す
    fetch = mocker.Mock(side_effect=[
        ConnectionError("timeout"),
        ConnectionError("timeout"),
        b"model weights",
    ])
    result = download_with_retry(fetch, max_retries=3)
    assert result == b"model weights"
    assert fetch.call_count == 3                   # ちゃんと3回試した


def test_retry_gives_up_after_max(mocker):
    fetch = mocker.Mock(side_effect=ConnectionError("down"))   # 常に失敗
    with pytest.raises(ConnectionError):
        download_with_retry(fetch, max_retries=2)
    assert fetch.call_count == 2                   # 上限で打ち切っている


def test_side_effect_sequence_of_values(mocker):
    # 例外だけでなく「戻り値の列」も渡せる（呼ぶたびに次の値が返る）
    fetch = mocker.Mock(side_effect=[1, 2, 3])
    assert [fetch(), fetch(), fetch()] == [1, 2, 3]


# ---- 2. mocker.spy: 本物を動かしつつ記録する --------------------------------
def test_spy_records_but_executes(mocker):
    spy = mocker.spy(ml.preprocessing, "min_max_scale")

    result = ml.preprocessing.min_max_scale([0, 5, 10])

    # 本物が実行されている（結果は正しい）
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])
    # かつ、呼ばれ方も記録されている
    spy.assert_called_once_with([0, 5, 10])
    np.testing.assert_allclose(spy.spy_return, [0.0, 0.5, 1.0])


# ---- 3. mocker.patch.object: 文字列パスなしで差し替え -----------------------
def test_patch_object_on_class(mocker):
    # クラスのメソッドを差し替える（"ml.predictor.Predictor.predict" と書くのと同じ）
    mocker.patch.object(Predictor, "predict", return_value=np.array([9.9]))
    # __init__ 内の load_weights（ディスクI/O）も止める
    mocker.patch.object(ml.predictor, "load_weights", return_value=np.zeros(3))

    p = Predictor("dummy.npy")
    np.testing.assert_allclose(p.predict([[1, 2, 3]]), [9.9])


def test_patch_object_on_instance(mocker):
    # インスタンス1個だけ差し替える（他のインスタンスには影響しない）
    mocker.patch.object(ml.predictor, "load_weights",
                        return_value=np.array([1.0, 0.0]))
    p1 = Predictor("a.npy")
    p2 = Predictor("b.npy")

    mocker.patch.object(p1, "predict", return_value="mocked!")
    assert p1.predict([[1, 2]]) == "mocked!"
    np.testing.assert_allclose(p2.predict([[3.0, 7.0]]), [3.0])   # p2 は本物のまま


# ---- 4. mocker.patch.dict: dict / 環境変数の差し替え ------------------------
LABEL_MAP = {"cat": 0, "dog": 1}


def test_patch_dict(mocker):
    # テスト中だけキーを追加・上書き（終わったら元に戻る）
    mocker.patch.dict(LABEL_MAP, {"bird": 2})
    assert LABEL_MAP == {"cat": 0, "dog": 1, "bird": 2}


def test_patch_dict_clear(mocker):
    # clear=True で「その内容だけ」にする
    mocker.patch.dict(LABEL_MAP, {"fish": 9}, clear=True)
    assert LABEL_MAP == {"fish": 9}


def test_dict_restored():
    # 前の2テストの変更が残っていないこと（自動復元の確認）
    assert LABEL_MAP == {"cat": 0, "dog": 1}
