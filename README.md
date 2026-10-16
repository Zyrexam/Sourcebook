<div align="center">

# Sourcebook

**A small document workspace. Answers you can trace.**

![Python](https://img.shields.io/badge/Python-3.11–3.13-306450)
![Storage](https://img.shields.io/badge/storage-SQLite-306450)

Bring PDFs, notes and Markdown together. Search them, inspect the evidence,
and connect a local or remote model when you want generated answers.

![Document workspace with the bundled example collection](assets/workspace.png)

</div>

## What it is

Sourcebook indexes PDF, Markdown and text files into one SQLite database and answers questions
with linked source passages. Ingestion, retrieval and answer validation all live in one small
Python package with a bundled web interface — no frontend build step, no mandatory cloud service.

- **Incremental ingestion** — SHA-256 change detection, transactional replace and delete.
- **Inspectable retrieval** — SQLite FTS5/BM25, optional dense retrieval with rank fusion, optional reranking.
- **Traceable answers** — page, section and character provenance plus exact-quote validation; an answer that fails validation is withheld.
- **Provider choice** — offline source excerpts, Ollama, or any Chat Completions-compatible endpoint.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/Zyrexam/Sourcebook.git
cd Sourcebook
uv sync --locked
uv run sourcebook serve
```

Open **http://127.0.0.1:8000** and click **Try the example collection**, or upload your own PDF,
Markdown or UTF-8 text. The default mode returns verbatim excerpts and needs no API key or model
download. The collection lives in `.data/knowledge.sqlite3`; re-uploading an unchanged file is a
no-op. Select documents with the library checkboxes, click a filename to read its indexed text, or
**Read in document** beside an answer to highlight the cited passage.

```bash
uv run sourcebook demo
uv run sourcebook ingest notes.md handbook.pdf
uv run sourcebook ask 'How does incremental indexing work?'
```

## Generated answers

Configuration is by environment variable; [.env.example](.env.example) is a reference, not an
auto-loaded secrets file.

```bash
# A model already installed in Ollama
export SOURCEBOOK_PROVIDER=ollama
export SOURCEBOOK_BASE_URL=http://127.0.0.1:11434
export SOURCEBOOK_MODEL=your-installed-model
uv run sourcebook serve

# Any Chat Completions-compatible API (HTTPS base URL, /v1 if required)
export SOURCEBOOK_PROVIDER=compatible
export SOURCEBOOK_BASE_URL=https://your-provider.example/v1
export SOURCEBOOK_MODEL=your-model-id
export SOURCEBOOK_API_KEY=your-key
```

By default the model writes each quotation. Add `SOURCEBOOK_ANSWER_STYLE=span` and it instead
returns sentence numbers; the application writes the claim and its quote from the retrieved text,
so a quotation cannot be a paraphrase and a statement cannot contain words the source lacks. The
trade-off is fluency — answers become the source's own sentences.

Remote generation sends the question and retrieved passages to your provider; credentials stay on
the server. Source ids and quotations are validated before anything is shown. **An exact quotation
proves provenance, not that a statement follows from it** — review the linked passages.

## Optional semantic retrieval

```bash
uv sync --locked --extra semantic
export SOURCEBOOK_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export SOURCEBOOK_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
uv run --extra semantic sourcebook serve
```

These are small English examples, not a best-in-class claim — choose models for your language and
documents. Weights download on first run unless the identifiers point to local directories.
Embeddings are keyed by model identity and chunk hash; existing documents embed lazily.

## Evaluation

```bash
uv run sourcebook evaluate eval/questions.jsonl --output artifacts/evaluation.json
```

The evaluator builds an isolated temporary index and reports document recall@5, MRR@5, exact
citation validity and abstention on unanswerable questions. The bundled set is a deterministic
regression fixture, **not a general RAG benchmark**.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/sourcebook src/ui
uv run pytest --cov=sourcebook --cov-fail-under=85
uv build
```

## Architecture

Ingestion, retrieval and answer validation live in one small Python package.
[ARCHITECTURE.md](ARCHITECTURE.md) covers index lifecycle, cache keys and model boundaries. The API
reference is at **http://127.0.0.1:8000/docs** — `POST /api/documents`, `DELETE /api/documents/{id}`,
`POST /api/ask`, `GET /api/status`.

## Limits

A local, single-user workspace. Uploads are capped at 20 MiB, 500 PDF pages and two million
extracted characters; scans need OCR first. Complex PDF tables and columns may extract poorly.
There is no authentication, folder watch, background job service or token streaming. SQLite stores
unencrypted local data and deletion is not secure erasure — keep sensitive documents out.
