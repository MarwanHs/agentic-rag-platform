from fastapi import FastAPI

app = FastAPI(title="agentic-rag-platform")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
