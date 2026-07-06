"""モデルレジストリへの公開（外部HTTP依存の例）。

実務では学習済みモデルを MLflow / S3 / 社内レジストリ等にアップロードする。
テストで本物のネットワークを叩くわけにはいかないので、
_http_post をモックに差し替えてロジックだけを検証する（tests/churn/test_model.py 参照）。
"""
import json
import urllib.request
from pathlib import Path


def _http_post(url: str, payload: dict) -> int:
    """JSON を POST して HTTP ステータスコードを返す（実際の外部通信）。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status


def publish_model(model_path, registry_url: str, model_name: str, version: str) -> str:
    """モデルファイルをレジストリに登録し、"name:version" の ID を返す。

    - ファイルが存在しなければ FileNotFoundError（アップロード前に検証）
    - レジストリが 200 以外を返したら RuntimeError
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    payload = {
        "name": model_name,
        "version": version,
        "size_bytes": model_path.stat().st_size,
    }
    status = _http_post(f"{registry_url}/models", payload)
    if status != 200:
        raise RuntimeError(f"registry returned status {status}")

    return f"{model_name}:{version}"
