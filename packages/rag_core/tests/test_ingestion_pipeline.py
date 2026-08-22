from __future__ import annotations

from pathlib import Path

from rag_core.code_navigation.schema import ensure_schema as ensure_code_navigation_schema
from rag_core.ingestion.pipeline import discover_python_files, parse_files, run_pipeline
from rag_core.jobs.schema import ensure_schema as ensure_jobs_schema
from rag_core.jobs.store import create_job, get_job
from rag_core.retrieval.qdrant_index import collection_name_for_codebase

FIXTURES_DIR = Path(__file__).parent / "fixtures"
INGESTION_REPO_DIR = FIXTURES_DIR / "ingestion_repo"


class FailingEmbeddingClient:
    """Always raises -- used to exercise the pipeline's failure path."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding provider outage")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("simulated embedding provider outage")


def _cleanup(conn, qdrant_client, codebase_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM symbol_references WHERE codebase_id = %s", (codebase_id,))
        cur.execute("DELETE FROM symbols WHERE codebase_id = %s", (codebase_id,))
        cur.execute("DELETE FROM jobs WHERE id = %s", (codebase_id,))
    conn.commit()
    collection_name = collection_name_for_codebase(codebase_id)
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)


def test_discover_python_files_skips_noise_dirs(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "ignored.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("not python\n")

    files = discover_python_files(tmp_path)

    assert [str(f.relative_to(tmp_path)) for f in files] == ["pkg/mod.py"]


def test_parse_files_uses_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n")
    files = discover_python_files(tmp_path)

    parsed = parse_files(tmp_path, files)

    assert len(parsed) == 1
    assert parsed[0].file_path == "mod.py"


def test_run_pipeline_end_to_end_resolves_cross_file_references(pg_conn, qdrant_client, fake_embedding_client) -> None:
    ensure_jobs_schema(pg_conn)
    ensure_code_navigation_schema(pg_conn)
    job = create_job(pg_conn, "https://github.com/example/ingestion-fixture")

    try:
        run_pipeline(pg_conn, qdrant_client, fake_embedding_client, job, clone_dir=INGESTION_REPO_DIR)

        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "ready"
        assert fetched.failure_reason is None
        assert fetched.pipeline_state["embed"]["batches_total"] >= 1
        assert fetched.pipeline_state["embed"]["batches_done"] == fetched.pipeline_state["embed"]["batches_total"]

        # code-navigation: the "helper" symbol is defined in z_definer.py
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM symbols WHERE codebase_id = %s AND name = 'helper' AND file_path = 'z_definer.py'",
                (job.id,),
            )
            symbol_row = cur.fetchone()
        assert symbol_row is not None
        symbol_id = symbol_row[0]

        # the call to helper() lives in a_caller.py, which sorts and is
        # processed *before* z_definer.py -- this only resolves correctly if
        # every file's symbols were inserted before any file's references.
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT symbol_id FROM symbol_references WHERE codebase_id = %s AND name = 'helper' "
                "AND file_path = 'a_caller.py'",
                (job.id,),
            )
            reference_row = cur.fetchone()
        assert reference_row is not None
        assert reference_row[0] == symbol_id

        # retrieval: chunks were embedded and upserted into the codebase's collection
        collection_name = collection_name_for_codebase(job.id)
        assert qdrant_client.collection_exists(collection_name)
        count = qdrant_client.count(collection_name=collection_name).count
        assert count > 0
    finally:
        _cleanup(pg_conn, qdrant_client, job.id)


def test_run_pipeline_marks_job_failed_on_embedding_error(pg_conn, qdrant_client) -> None:
    ensure_jobs_schema(pg_conn)
    ensure_code_navigation_schema(pg_conn)
    job = create_job(pg_conn, "https://github.com/example/ingestion-fixture")

    try:
        run_pipeline(pg_conn, qdrant_client, FailingEmbeddingClient(), job, clone_dir=INGESTION_REPO_DIR)

        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.failure_reason is not None
        assert "simulated embedding provider outage" in fetched.failure_reason

        # the pipeline must not have deleted the caller-supplied clone_dir
        assert INGESTION_REPO_DIR.exists()
        assert (INGESTION_REPO_DIR / "a_caller.py").exists()
    finally:
        _cleanup(pg_conn, qdrant_client, job.id)


def test_run_pipeline_does_not_delete_injected_clone_dir_on_success(pg_conn, qdrant_client, fake_embedding_client) -> None:
    ensure_jobs_schema(pg_conn)
    ensure_code_navigation_schema(pg_conn)
    job = create_job(pg_conn, "https://github.com/example/ingestion-fixture")

    try:
        run_pipeline(pg_conn, qdrant_client, fake_embedding_client, job, clone_dir=INGESTION_REPO_DIR)
        assert INGESTION_REPO_DIR.exists()
    finally:
        _cleanup(pg_conn, qdrant_client, job.id)
