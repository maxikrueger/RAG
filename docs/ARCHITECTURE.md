# Architecture

QMS-RAG indexes a directory of quality management documents and answers questions about them.
The pipeline is deterministic Python from the file system up to the assembled context; a
language model is only involved in the last step, and only with the passages it was handed.

This document describes what the code does today. Features that are planned but not
implemented are listed under *Known issues* in the [README](../README.md).

## Layers and boundaries

```
app.py              Streamlit UI — the only module that knows about session state
core/rag.py         retrieval + context assembly + source extraction
core/search/        hybrid search strategy (hybrid_search) over the store (vector_store)
core/ingestion/     scan → parse → chunk → embed → store
core/parser/        one parser per format, selected by a registry
core/chunking/      document → chunks
core/embeddings/    EmbeddingProvider interface + local implementation
core/llm/           LLMProvider interface + Claude and Ollama implementations
config.py           all settings, read once from the environment
```

Each layer depends only on the interfaces below it. `anthropic` is imported in
`core/llm/claude_provider.py` and nowhere else, `lancedb` only in `core/search/vector_store.py`,
`fitz`/`docx`/`openpyxl`/`pptx` only in their own parser modules, and `streamlit` only in
`app.py`. The RAG pipeline talks to `HybridSearch` and `LLMProvider` and cannot tell which
provider or vector store it got — which is what makes it testable without a model or an API key.

`config.py` reads the environment at import time but creates no directories; `ensure_directories()`
does that when the application starts. Importing `config` has no side effects.

## Ingestion

`IngestionPipeline.run()` walks the document directory recursively and, per file:

1. Computes a change key from file size and modification time.
2. Skips the file if that key matches what the index already holds.
3. Deletes the file's existing chunks if it changed.
4. Parses it, chunks it, embeds the chunks and writes them to LanceDB.
5. Counts the file as new or updated **after** the write succeeds, so a failed file is reported
   as a failure only.

Hidden files, Office lock files (`~$…`) and unsupported extensions are skipped. A single
unreadable document — password-protected, corrupt, or in a format whose parser dependency is
missing — is logged, counted and skipped; the run continues. The BM25 index is rebuilt once at
the end of a run that changed anything.

The parser registry maps a file extension to the first parser that claims it. Parsers whose
dependency is not installed are left out at registry construction, so a partial installation
degrades to fewer supported formats instead of failing at import.

## Chunking

Embedding a whole document gives a vector that is close to everything and useful for nothing, so
documents are split before they are embedded. The engine works top-down:

1. **Section**: parsers return the document's own structure — Word heading styles, PDF headings
   detected from numbering (`4.2 Freigabe`) or all-caps lines, Excel sheets, PowerPoint slides.
   A section that fits inside the token budget becomes one chunk and keeps its title and page
   number as metadata.
2. **Paragraph**: an oversized section is filled paragraph by paragraph up to the budget.
3. **Sentence**: a single oversized paragraph is split at sentence boundaries, with German
   capitalisation taken into account.

Afterwards, each chunk gets the tail of its predecessor prepended (`CHUNK_OVERLAP`, default 150
tokens, cut at a sentence boundary) so a statement that straddles a boundary is retrievable from
either side, and chunks below `MIN_CHUNK_SIZE` are merged into their predecessor — a heading
with two lines under it is not worth its own vector.

Token counts are estimated at four characters per token, which is close enough for German prose
and costs nothing to compute.

## Search

Both searches run over one LanceDB table: a vector column with the 384-dimensional embeddings
and a Tantivy BM25 index over the chunk text.

Pure vector search fails exactly where QMS questions are most specific. "Was steht in VA-0815?"
needs an exact match on a token the embedding model has never seen; "Wie läuft eine
Kundenreklamation ab?" needs semantics. So the query decides the weighting:

| Query contains | Vector | BM25 | Example |
|---|---|---|---|
| A document number or a quoted phrase | 0.3 | 0.7 | `VA-0815`, `"Lenkung von Aufzeichnungen"` |
| A norm reference | 0.4 | 0.6 | `ISO 9001`, `IATF 16949` |
| Neither (free-form question) | 0.6 | 0.4 | `Wie läuft die Freigabe ab?` |

`HybridSearch` first tries LanceDB's native hybrid query. If that is unavailable — an older
version, or no full-text index yet — it runs both searches separately and merges them by rank:
each hit scores `(1 - rank / n) × weight` for its side, and a chunk found by both sides adds up
and rises to the top. That is deliberately rank-based: vector distances and BM25 scores are not
on a comparable scale, so merging them by raw value would let one side dominate by accident.

Search modes `semantic` and `keyword` bypass the mixing and use one index only. LanceDB reports
relevance in a different column per mode (`_distance`, `_relevance_score`, `_score`), which is
normalised to one "higher is better" score before results leave the search layer.

Metadata filters are translated into SQL `WHERE` clauses against an allow-list of columns, and
values are escaped — a Windows path or a file name containing an apostrophe would otherwise
break the clause.

## RAG pipeline

`RAGPipeline.query()` retrieves the top *k* chunks (default 8), labels each one with its source
and assembles them into a context block:

```
[Quelle 1: VA-0815 Lenkung von Dokumenten.pdf > 4.2 Freigabe, S. 7]
<chunk text>

---

[Quelle 2: QM-Handbuch.pdf > 7.5 Dokumentierte Information]
<chunk text>
```

Those labels are what lets the model attribute statements, and the pipeline returns the same
sources as structured data next to the answer, deduplicated per file, so the UI can show them
without parsing the answer text.

The system prompt instructs the model to answer only from the provided passages, to say so when
the passages do not contain the answer, and to name its sources. If retrieval finds nothing, the
pipeline says so itself and never calls the model — a question with no hits costs no tokens.

The last six chat messages are prepended to the question so follow-ups ("und wer gibt das
frei?") keep their referent, each truncated to 500 characters to bound the prompt.

`stream_query()` yields the answer token by token and the source list as its final element.

## Model providers

`LLMProvider` requires `generate`, `stream`, `model_name` and `is_available`.

- **ClaudeProvider** sends context and question as one user message, with the QMS system prompt
  in the `system` field, and reports the token counts the API returns. It accepts an injected
  client, which is how the tests exercise it without an API key. A missing key fails at
  construction with an actionable message rather than at the first question.
- **OllamaProvider** talks to the local `/api/chat` endpoint over the standard library — no
  extra dependency for a feature that is one HTTP POST — passes the system prompt as its own
  message, as Ollama expects, and bounds every call with a timeout. An unreachable server is
  raised as a `ConnectionError` naming the URL.

Switching providers is `LLM_PROVIDER` in `.env`; the factory in `core/llm/__init__.py` imports
only the implementation that is actually used, so a deployment without the `anthropic` package
still runs on Ollama.

## Privacy

| Step | Where | Detail |
|---|---|---|
| Parsing | Local | Documents are read from the share and processed in memory |
| Chunking | Local | Plain Python, no service involved |
| Embeddings | Local | sentence-transformers on the CPU |
| Index | Local | LanceDB writes to `DATA_DIR`, next to the application |
| Search | Local | Both vector and BM25 search run against the local index |
| Answer generation | Cloud or local | Claude API, or Ollama for a fully local setup |

What reaches the cloud provider, if one is configured, is the current question, the chat history
and the retrieved passages — not the document base, and never a whole document. Document
properties such as author and last editor are read by the parsers for diagnostics but are not
written to the index, so they are never part of a prompt.

API keys are read from the environment or `.env`, which `.gitignore` excludes. The repository
ships `.env.example` with placeholders only.

## Logging

`setup_logging()` installs a console handler plus rotating file handlers (10 MB, 5 backups) per
category: `application.log`, `search.log` and `ingestion.log` under
`DATA_DIR/logs`. Search logs the query, mode, hit count and weighting; ingestion logs per-file
decisions and a run summary. Log files are local and git-ignored.

## Tests

144 tests cover the chunking strategy, each parser, the search weighting and merge logic, the
store's SQL escaping and filtering, incremental indexing, the RAG pipeline's context and source
handling, both LLM providers, and the configuration and logging setup.

Embeddings come from a deterministic bag-of-words fake — words are hashed onto dimensions, so
texts sharing vocabulary end up close to each other, which is enough to assert ranking without
loading a model. The language model is a fake that records the calls it received. PDF, Word,
Excel and PowerPoint fixtures are written at runtime, so the repository holds no binary test
data. Store and ingestion tests run against a real LanceDB in a temporary directory: the parts
that are easy to get wrong — escaping, filtering, incremental re-indexing — are tested against
the real engine rather than a mock of it.
