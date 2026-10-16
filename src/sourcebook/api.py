"""Same-origin local API and static application. Run one worker for cache consistency."""

import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from sourcebook import __version__
from sourcebook.ingestion import MAX_BYTES
from sourcebook.models import Answer
from sourcebook.providers import ProviderError
from sourcebook.service import KnowledgeBase, configured_kb
from ui import ROOT as UI_ROOT


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=8)
    document_ids: list[str] | None = Field(default=None, max_length=100)


def create_app(kb: KnowledgeBase | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.kb = kb or configured_kb()
        yield

    app = FastAPI(title="Sourcebook", version=__version__, lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse(
                {"detail": "Cross-origin requests are not allowed."}, status_code=403
            )
        if request.method == "POST":
            try:
                length = int(request.headers.get("content-length", "-1"))
            except ValueError:
                length = -1
            limit = MAX_BYTES + 65536 if request.url.path == "/api/documents" else 65536
            if length < 0:
                return JSONResponse({"detail": "Content-Length is required."}, status_code=411)
            if length > limit:
                return JSONResponse(
                    {"detail": "Request body exceeds the size limit."}, status_code=413
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith("/docs"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid_input(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ProviderError)
    async def provider_failed(request: Request, exc: ProviderError):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/api/status")
    def status(request: Request):
        kb = request.app.state.kb
        docs = kb.store.documents()
        return {
            "documents": len(docs),
            "chunks": sum(d["chunks"] for d in docs),
            "provider": kb.provider.identity.split(":")[0],
            "retrieval": "hybrid-rrf" if kb.embedder else "bm25",
            "reranker": kb.reranker.identity if kb.reranker else "none",
        }

    @app.get("/api/documents")
    def documents(request: Request):
        return request.app.state.kb.store.documents()

    @app.post("/api/documents")
    async def upload(request: Request, file: UploadFile):
        try:
            data = await file.read(MAX_BYTES + 1)
            return await run_in_threadpool(request.app.state.kb.ingest, file.filename or "", data)
        finally:
            await file.close()

    @app.get("/api/documents/{document_id}/passages")
    def passages(
        request: Request,
        document_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=50),
        focus: str | None = None,
    ):
        try:
            return request.app.state.kb.store.read_document(document_id, offset, limit, focus)
        except KeyError as exc:
            raise HTTPException(404, exc.args[0]) from exc

    @app.delete("/api/documents/{document_id}")
    def delete(request: Request, document_id: str):
        if not request.app.state.kb.delete(document_id):
            raise HTTPException(404, "Document not found.")
        return {"deleted": document_id}

    @app.post("/api/demo")
    def demo(request: Request):
        return [
            request.app.state.kb.ingest(p.name, p.read_bytes())
            for p in sorted(Path(__file__).with_name("samples").glob("*.md"))
        ]

    @app.post("/api/ask", response_model=Answer)
    def ask(request: Request, question: Question):
        return request.app.state.kb.ask(question.question, question.top_k, question.document_ids)

    # Host MIME tables disagree: Windows reports text/plain for .js, which browsers refuse
    # to execute. The application knows what it serves, so it pins those types.
    for suffix, content_type in (
        (".css", "text/css"),
        (".html", "text/html"),
        (".js", "text/javascript"),
        (".svg", "image/svg+xml"),
    ):
        mimetypes.add_type(content_type, suffix)
    app.mount("/", StaticFiles(directory=UI_ROOT, html=True), name="web")
    return app
