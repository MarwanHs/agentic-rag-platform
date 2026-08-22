from __future__ import annotations

import uuid


def test_create_and_get_job(client, pg_conn) -> None:
    resp = client.post("/jobs", json={"url": "https://github.com/example/repo"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]

    try:
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "queued"
        assert body["pipeline_state"] == {}
        assert body["failure_reason"] is None
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
        pg_conn.commit()


def test_get_job_not_found(client) -> None:
    resp = client.get(f"/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404
