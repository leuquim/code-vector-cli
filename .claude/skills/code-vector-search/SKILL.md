---
name: code-vector-search
description: Semantic code search using vector embeddings to find code by meaning, locate similar patterns, analyze impact, and search conversations. Use when user asks to find code, locate implementations, understand architecture, or analyze change impact.
allowed-tools: Bash
---

# Code Vector Search

Semantic code search via `code-vector-cli`. Find code by meaning, not keywords.

## Quick Decision Tree

**User wants to find code?**
```bash
code-vector-cli search "user authentication" -n 10
```

**User mentions a specific function/class name?**
```bash
code-vector-cli search-hybrid "getUserById" -n 10
```

**User asks "what files relate to X" or starting a feature?**
```bash
code-vector-cli context "implement user permissions" -n 15
```

**User asks "what will this change affect"?**
```bash
code-vector-cli impact path/to/file.js
```

**User asks about similar code or patterns?**
```bash
code-vector-cli similar path/to/file.js -n 10
```

**User asks about past discussions/decisions?**
```bash
code-vector-cli search-conversations "auth design" -n 5
```

---

## Before Searching

Check if indexed (run once per session if unsure):
```bash
code-vector-cli stats
```

If 0 points or error, tell user: "Project not indexed. Run `code-vector-cli init` first."

---

## Command Reference

### search - Find by meaning
```bash
code-vector-cli search "error handling middleware" -n 10
code-vector-cli search "S3 upload" -t 0.2        # Lower threshold = more results
code-vector-cli search "React hooks" --show-content  # Include code snippets
```

### search-hybrid - Find by name + meaning
```bash
code-vector-cli search-hybrid "validateToken" -n 10
code-vector-cli search-hybrid "StorageService S3" --semantic-weight 0.5 --bm25-weight 0.5
```
Best for: function names, class names, specific technical terms.

### context - Files for a task
```bash
code-vector-cli context "add OAuth login" -n 15
```
Returns ranked files relevant to implementing a feature.

### impact - What might break
```bash
code-vector-cli impact path/to/service.js
code-vector-cli impact "authentication changes" -t 0.6
```
Similarity-based, not true dependency tracking. Combine with `grep` for imports.

### similar - Find related code
```bash
code-vector-cli similar path/to/file.js -n 10
code-vector-cli similar "error handling pattern" -t 0.7
```

### search-conversations - Past discussions
```bash
code-vector-cli search-conversations "database migration" -n 5
```
Requires: `code-vector-cli migrate-conversations` run once.

### search-docs - Documentation
```bash
code-vector-cli search-docs "API authentication" -n 5
```

---

## Thresholds (-t)

| Value | Use case |
|-------|----------|
| 0.1-0.2 | Exploratory, cast wide net |
| 0.3 | Default, balanced |
| 0.4-0.5 | Strict, high confidence only |

---

## When NOT to Use

- User gave exact file path → just read it
- Looking for literal string → use `grep`
- General programming question → answer directly

---

## If No Results

1. Lower threshold: `-t 0.2`
2. Try hybrid search: `search-hybrid`
3. Check index: `code-vector-cli stats`
4. Suggest reindex: `code-vector-cli index --incremental`

---

## Maintenance (user-initiated only)

```bash
code-vector-cli index --incremental  # Update index
code-vector-cli stats                # Check status
code-vector-cli delete --force       # Reset and reinit
```

Never run `init`, `index`, or `delete` without user request.
