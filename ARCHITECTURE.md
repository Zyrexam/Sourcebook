# Architecture and tradeoffs

```text
src/ui: HTML / CSS / JavaScript
           │ same-origin HTTP
       FastAPI (api.py)
           │
    KnowledgeBase (service.py)
      ┌────┴───────────────┐
 ingest / replace      query
      │                   │
 PDF / UTF-8         BM25 candidates
 page / section      + optional cosine candidates
 chunk + offsets       → RRF → optional cross-encoder
      │                   │
      └── SQLite WAL ──────┘
                          │
                AnswerProvider protocol
                 ├ extractive, offline
                 ├ Ollama, local HTTP
                 └ compatible Chat Completions API
                          │
                 structured claims + evidence
                  (model writes the quote, or selects a span)
                          │
                 exact quote / source validation
```

## Repository layout

```text
src/
  sourcebook/          the Python package, published as `sourcebook`
    api.py             FastAPI application, security headers, static mount
    service.py         KnowledgeBase: ingestion, retrieval, answers, cache
    store.py           SQLite schema, FTS5 index, chunks and cached vectors
    ingestion.py       PDF and text extraction, chunking, tokenization
    retrieval.py       BM25, dense candidates, reciprocal rank fusion
    providers.py       extractive, Ollama and Chat Completions adapters
    evaluation.py      dataset scoring: recall, MRR, citation validity
    cli.py             serve, demo, ingest, ask, evaluate
    samples/           bundled example documents, packaged in the wheel
  ui/                  the web interface, served by the same-origin API
    index.html         application shell
    app.js             upload, ask, source reader and citation behaviour
    style.css          the entire stylesheet, no build step
    mark.svg           brand mark and favicon
tests/                 pytest suite
tools/                 browser and semantic smoke scripts
eval/                  labeled datasets and their committed results
assets/                screenshots used by the README
```

The interface is a package of its own rather than a folder inside `sourcebook`, so
the frontend and the retrieval code can be read and changed independently. Both
packages ship in the wheel, which is why the installed application serves its own
interface with no build step and no external asset directory.

## Index lifecycle

SQLite is the source of truth; there is no independent positional vector file.
Document identity derives from its filename. SHA-256 over bytes plus chunker version
detects unchanged uploads. A changed file is parsed and embedded before its old rows
are replaced inside one transaction. Failure rolls back the replacement. Renaming a
file creates a distinct document; remove the old name explicitly when appropriate.

Chunks preserve page, the most recent Markdown heading, and zero-based character
offsets in normalized page text (half-open ranges). They target 1,200 characters with
160 characters of overlap. PDF pages and Markdown sections remain separate. Tables, columns, scanned text,
and complex layouts are not reconstructed or OCRed.

An FTS5 index uses BM25 and Porter stemming for Latin words, with explicit CJK
character tokenization. Dense retrieval is optional. Model identity and chunk-content
hash key the embedding cache. RRF fuses up to 30 candidates from each channel, then an
optional cross-encoder orders the pool. Duplicate and heavily overlapping passages
are suppressed before selecting at most eight sources. Retrieval scores are ordering
signals, not calibrated confidence probabilities. The dense cosine cutoff is 0.35;
it is an engineering default, not a validated relevance threshold for arbitrary models.

## Answers and caches

Providers return structured claims and evidence. Either the model writes each quotation, or with
`SOURCEBOOK_ANSWER_STYLE=span` it selects a sentence number and the application writes the
quotation from retrieved source text, so a quotation cannot be a paraphrase and a claim cannot
contain words the source lacks. Every evidence entry must point to a retrieved chunk and quote a
nonempty exact substring. Any invalid entry withholds the complete generated draft. An empty
draft or empty retrieval result abstains. Semantic entailment requires separate evaluation and
human review. Offline mode returns the most query-relevant paragraph from each of the top three
passages as verbatim excerpts rather than generated interpretations.

Cache keys include question, retrieval configuration, provider/model identity, prompt
version and the document fingerprints. Mutations clear stored answers. The cache is
bounded to 128 entries. The process serializes ingestion and answers with one lock;
SQLite transactions protect database integrity across CLI invocations. A query already
in progress may finish with its earlier snapshot. Fingerprinted keys prevent that
snapshot from becoming a current cached answer after an update.

## Deliberate scope

One process, one SQLite file, one static frontend. No LangChain, agent runtime,
Redis, vector service, user accounts, containers, or frontend package manager is
required. Dense search scans cached vectors in process and is intended for small
personal collections. Long indexing or generation requests serialize other work.
Background jobs, token streaming, OCR, parser isolation and large-corpus vector
indexes should be added only when a measured workload justifies them.

## References

- [SQLite FTS5](https://www.sqlite.org/fts5.html): full-text indexing and BM25.
- [Sentence Transformers retrieve and rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html): two-stage search.
- [Ollama chat API](https://docs.ollama.com/api/chat): local structured generation.
- [Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create): the compatibility adapter's wire format.

The core architecture is intentionally independent of a specific model vendor.


## Document selection and reading

An optional document-ID scope is applied inside both lexical and dense retrieval before
candidate caps. Missing IDs fail explicitly; an empty scope returns no evidence. The
canonical scope participates in cache identity. Selection is a retrieval convenience
for this single-user app, not an access-control boundary.

The source reader serves bounded pages of indexed chunks from a consistent SQLite read
snapshot. A citation focus resolves its current document and exact chunk ID; replacement
invalidates old chunk IDs, so a stale citation produces an explicit error instead of
opening a different passage. Content is rendered as text, including Markdown/HTML input.
