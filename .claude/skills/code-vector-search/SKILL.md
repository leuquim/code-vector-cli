---
name: code-vector-search
description: Semantic code search over a local vector index (code-vector-cli / cvc). Find code by meaning, locate implementations and similar patterns, gather files for a task, and analyze change impact. Use when asked to find code, locate where something is implemented, understand a codebase area, or find what a change affects - especially in large repos where grep by keyword is not enough.
allowed-tools: Bash
---

# Code Vector Search

Semantic code search via `code-vector-cli` (alias: `cvc`). Finds code by meaning,
not literal text, over a locally indexed vector database (zembed-1 embeddings +
zerank-1 reranking). Prefer this over grep when the user describes *what code
does* rather than an exact string.

The binary is `cvc` (short alias) or `code-vector-cli` - they are identical.

## Decision tree

| User intent | Command |
|---|---|
| Find code by what it does | `cvc search "rate calculation logic" -n 10` |
| Knows a specific name/identifier | `cvc search-hybrid "SlotPricingCalculator" -n 10` |
| "What files relate to X" / starting a feature | `cvc context "add demand-based pricing" -n 15` |
| Find similar/duplicate code | `cvc similar path/to/file.cs -n 10` |
| "What does changing this affect" | `cvc impact path/to/file.cs` |
| Search docs/markdown | `cvc search-docs "pricing setup" -n 5` |
| Search commit history | `cvc search-git "refactor pricing" -n 5` |

All commands default to the current directory; add `--path /repo` to target
another. Add `--repo NAME` in multi-repo workspaces to filter.

## Before searching: confirm the index exists

Run once per session if unsure:
```bash
cvc stats --path /repo
```
- Non-zero points across `code_functions` / `code_classes` -> ready to search.
- 0 points or a connection error -> not indexed or Qdrant is down (see Setup).

## Setup (only on user request - never index unprompted)

First-time indexing of a repo:
```bash
cvc index --path /repo          # creates collections + embeds all TRACKED files
```
- Indexes only git-tracked files (uses `git ls-files`).
- `init` also works but `index` now creates collections itself; either is fine.

Incremental update after code changes (fast - only changed files):
```bash
cvc index --incremental --path /repo
```
- Detects changes via git diff -> content hash -> mtime (in that order).
- Run this after the user has edited code and wants fresh search results.
- A git post-commit hook can automate it: `cvc install-hook`.

Requirements for indexing/search to work:
- A Qdrant server reachable (config in `~/.code-vector-db.env`), or
  `QDRANT_LOCAL=true` for embedded mode. If a command reports it cannot reach
  Qdrant, tell the user to start it (`docker compose up -d`) - do not retry blindly.
- A `ZEROENTROPY_API_KEY` in `~/.code-vector-db.env` for the default cloud
  embeddings (or `EMBEDDING_PROVIDER=local` for offline). A missing key produces
  a clear error.

## Reading results

Each result includes, by design for triage in one call:
- `relevance` - the authoritative score that ordered the list (rerank score).
  This is what to trust; do not re-sort by anything else.
- `file_path` + `lines` - open this range directly; no follow-up search needed.
- `signature` and a bounded `code` snippet - enough to judge relevance without
  reading the file. Read the full `lines` range only when the snippet is truncated.
- `kind` - `source` vs `test` vs `documentation`. Prefer `source` results when
  the user wants the implementation; tests can rank high on natural-language queries.
- `type` (function/class/file), `name`, `parent`, `language`.

The CLI text output shows `[rank X | sim Y]`: `rank` (the reranker) drives the
order; `sim` is the raw vector similarity, shown for transparency only.

## Command details

```bash
# search - semantic, the default. --show-content prints code, --show-parent shows enclosing class.
cvc search "error handling middleware" -n 10
cvc search "database retry" -t 0.2          # lower threshold = wider net
cvc search "occupancy strategy" --show-content --show-parent

# search-hybrid - vector + BM25 keyword, then rerank. Best for known identifiers/filenames.
cvc search-hybrid "BuildPricedResult" -n 10
cvc search-hybrid "weather modifier" --semantic-weight 0.5 --bm25-weight 0.5

# context - ranked files to start a task (code + docs). Add --json for tool use.
cvc context "implement demand heatmap pricing" -n 15 --json

# impact - similarity-based blast radius (NOT true dependency tracking; pair with grep for imports).
cvc impact src/Pricing/SlotPricingCalculator.cs

# similar - related code, by file path OR description.
cvc similar src/Pricing/WeatherStrategy.cs -n 10
cvc similar "rounding policy" -t 0.5
```

## Thresholds (-t)

zembed-1 relevance scores run lower than older models; calibrate accordingly:

| Value | Use |
|---|---|
| 0.1-0.2 | Exploratory, cast a wide net |
| 0.3 | Default for `search` - balanced |
| 0.5 | Default for `similar` - related but not identical |
| 0.6+ | Strict, high-confidence only |

If a search returns nothing, lower `-t` before concluding the code is absent.

## When NOT to use this

- User gave an exact file path -> just read it.
- Looking for a literal string/symbol -> use grep (faster, exact).
- General programming question with no codebase lookup -> answer directly.

## If no results

1. Lower the threshold: `-t 0.2`.
2. Try `search-hybrid` if the query contains a specific name.
3. Check the index is populated: `cvc stats --path /repo`.
4. If stale, suggest: `cvc index --incremental --path /repo`.

## Gotchas

- Switching embedding provider/model/dimension changes vector size; the index
  must be rebuilt. If a command reports a dimension mismatch, the fix is
  `cvc delete --force --path /repo` then `cvc index --path /repo`.
- `impact` and `similar` are vector-similarity, not call-graph analysis - good
  for "what looks related," not a guarantee of every caller.

## Note for AI-driven use

When this project is configured with the MCP server (`cvc mcp serve`), the same
capabilities are available as tools (`semantic_search`, `hybrid_search`,
`find_similar`, `get_context`, `analyze_impact`, `search_docs`, `get_stats`) -
prefer those over shelling out to the CLI, as the server stays warm (sub-second
repeat searches vs ~4s cold CLI start).

## Maintenance (user-initiated only)

```bash
cvc index --incremental --path /repo   # refresh after edits
cvc stats --path /repo                   # check index health
cvc list-projects                         # all indexed repos
cvc delete --force --path /repo           # reset (then re-index)
```

Never run `index`, `init`, or `delete` without an explicit user request.
