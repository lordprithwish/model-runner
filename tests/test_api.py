import time

from fastapi.testclient import TestClient

from app.main import app
from app.services.runner import registry


def test_api():
    with TestClient(app) as client:
        assert client.get("/").json() == {"message": "Hello from Model Runner"}
        assert client.get("/health").json() == {"status": "ok"}
        models = {m["id"] for m in client.get("/v1/models").json()}
        assert models == {"sdxl-placeholder", "wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.2-i2v-a14b", "hunyuanvideo-1.5-720p-t2v", "hunyuanvideo-1.5-720p-i2v"}

        assert client.post("/v1/generate/image", json={"prompt": "x"}).status_code == 503
        assert client.post("/v1/generate/image", json={"prompt": "x", "model": "nope"}).status_code == 404
        assert client.post("/v1/generate/image", json={"prompt": "x", "model": "wan2.2-ti2v-5b"}).status_code == 400
        assert client.post("/v1/generate/image", json={"prompt": ""}).status_code == 422

        # Rejected by validation before the (GPU-only) model is touched.
        assert client.post("/v1/generate/video", json={"prompt": "x", "width": 1000}).status_code == 422
        assert client.post("/v1/generate/video", json={"prompt": "x", "image": "a"}).status_code == 422
        # Single-mode models reject the wrong input before queueing ("eA==" is base64 for "x").
        assert client.post("/v1/generate/video", json={"prompt": "x", "model": "wan2.2-i2v-a14b"}).status_code == 400
        r = client.post("/v1/generate/video", json={"prompt": "x", "model": "wan2.2-t2v-a14b", "image": "eA=="})
        assert r.status_code == 400 and "does not accept" in r.json()["detail"]

        assert client.get("/v1/jobs/nope").status_code == 404

        # Queued jobs run on the worker; the stub adapter makes the job fail with its message.
        registry.models["sdxl-placeholder"].enabled = True
        r = client.post("/v1/generate/image", json={"prompt": "x"})
        assert r.status_code == 202 and r.json()["status"] in ("queued", "running")
        for _ in range(100):
            job = client.get(f"/v1/jobs/{r.json()['id']}").json()
            if job["status"] == "failed":
                break
            time.sleep(0.01)
        assert job["status"] == "failed" and "not configured" in job["error"]
        assert job["params"]["prompt"] == "x" and job["run_seconds"] >= 0

        assert [j["id"] for j in client.get("/v1/jobs?status=failed").json()] == [job["id"]]
        stats = client.get("/v1/stats").json()
        assert stats["queue"]["failed"] == 1 and stats["queue"]["total"] == 1
        assert {m["id"]: m["loaded"] for m in stats["models"]} == dict.fromkeys(models, False)
        for section in stats:
            assert client.get(f"/v1/stats/{section}").status_code == 200
        assert client.get("/v1/stats/nope").status_code == 422
