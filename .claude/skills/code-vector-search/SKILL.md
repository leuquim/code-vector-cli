---
name: code-vector-search
description: Semantic code search using vector embeddings to find code by meaning, locate similar patterns, analyze impact, and search git history or conversations. Use when user asks to find code, locate implementations, understand architecture, check what code handles something, find similar patterns, or search past work.
allowed-tools: Bash
---

# Code Vector Search Skill

Use semantic code search via `code-vector-cli` to efficiently find code, understand architecture, and analyze dependencies across indexed codebases.

## When to Use This Skill

**Automatically trigger when:**
- User asks to find code matching a description or concept
- User wants to understand "how we implement X" or "where is Y defined"
- User needs to see examples of existing patterns
- User asks about code architecture or organization
- User wants to trace changes or understand why code was written
- User needs to assess impact of changes

**Do NOT use when:**
- User already provided exact file paths
- Searching for a specific string (use `grep` instead)
- User wants general programming knowledge (not codebase-specific)

## Prerequisites

The project must be indexed first. Check if indexed:

```bash
code-vector-cli stats
```

If not indexed or stats show 0 points, initialize:

```bash
code-vector-cli init
```

## Core Commands

### 1. Semantic Code Search

Find code by **meaning**, not exact keywords.

```bash
# Basic search
code-vector-cli search "authentication middleware" -n 10

# With code snippets
code-vector-cli search "S3 file upload" --show-content -C 5

# Adjust threshold for more/fewer results (lower = more results)
code-vector-cli search "database queries" -t 0.2
```

### 2. Hybrid Search (Semantic + Keyword)

Best for finding **specific function/class names** or when you need exact keywords + semantic understanding.

```bash
# Basic hybrid search (70% semantic, 30% keyword by default)
code-vector-cli search-hybrid "getUserById function"

# More keyword weight for exact matches
code-vector-cli search-hybrid "handleStripeWebhook" --semantic-weight 0.3 --bm25-weight 0.7

# More semantic weight for conceptual search
code-vector-cli search-hybrid "API rate limiting" --semantic-weight 0.8 --bm25-weight 0.2
```

### 3. Find Similar Code

Find code patterns similar to a specific file or concept.

```bash
# Similar to a file
code-vector-cli similar "src/utils/auth.py" -n 5

# Similar code by description
code-vector-cli similar "rate limiting middleware" --show-content
```

### 4. Get Context for Task

AI-powered file selection for implementing features.

```bash
# Get relevant files for a task
code-vector-cli context "add user authentication"

# JSON output for tool integration
code-vector-cli context "fix payment processing" --json
```

### 5. Impact Analysis

Understand dependencies and affected code.

```bash
code-vector-cli impact "src/models/user.py"
```

### 6. Search Conversations

Search past Claude Code sessions (requires conversation indexing setup).

**Prerequisites:**
1. Enable SessionEnd hook to save transcripts to `.claude-transcripts/`
2. Run `code-vector-cli migrate-conversations` to index them

```bash
# Search indexed conversations
code-vector-cli search-conversations "deployment issues" -n 5

# First-time setup: index conversations
code-vector-cli migrate-conversations
```

## Best Practices

### Choosing the Right Command

- **`search`**: General concepts, architecture questions, "how do we handle X"
- **`search-hybrid`**: Specific function/class names, exact terminology
- **`similar`**: "Show me code like this", pattern finding
- **`context`**: "What files do I need to modify for X task"
- **`impact`**: "What will break if I change this file"

### Effective Search Strategies

1. **Start broad, then narrow:**
   ```bash
   code-vector-cli search "authentication" -n 20
   # Review results, then get more specific
   code-vector-cli search-hybrid "authenticateUser middleware"
   ```

2. **Use context for implementation tasks:**
   ```bash
   code-vector-cli context "add rate limiting to API endpoints" --json
   ```

3. **Combine with impact analysis:**
   ```bash
   # Find the code
   code-vector-cli search "payment processing"
   # Analyze impact before changing
   code-vector-cli impact "src/payments/stripe.py"
   ```

4. **Adjust thresholds based on codebase size:**
   - Small codebase (<1000 files): `-t 0.5` (stricter)
   - Large codebase (>5000 files): `-t 0.2` (more permissive)

## Example Workflows

### Workflow 1: "Where do we handle authentication?"

```bash
# Start with semantic search
code-vector-cli search "authentication login user verification" -n 5 --show-content

# If too broad, use hybrid for specific terms
code-vector-cli search-hybrid "authenticate login" --semantic-weight 0.5 --bm25-weight 0.5

# Find similar implementations
code-vector-cli similar "src/auth/login.py"
```

### Workflow 2: "I need to add a new API endpoint"

```bash
# Get context for what files to modify
code-vector-cli context "add new API endpoint for user settings" --json

# Find similar endpoint implementations
code-vector-cli search "API endpoint route handler" --show-content

# Check impact on existing routes
code-vector-cli impact "src/routes/api.js"
```

### Workflow 3: "How did we solve this before?"

```bash
# Search past conversations (if indexed)
code-vector-cli search-conversations "database connection pooling"

# Search the actual implementation
code-vector-cli search "database connection pool implementation"

# Note: Conversations require setup via SessionEnd hook + migrate-conversations
```

## Troubleshooting

**No results found?**
- Lower threshold: `-t 0.1` or `-t 0.0`
- Try different wording or more general terms
- Check if codebase is indexed: `code-vector-cli stats`

**Too many irrelevant results?**
- Increase threshold: `-t 0.5` or `-t 0.7`
- Use `search-hybrid` with higher keyword weight
- Be more specific in your query

**Need to reindex?**
```bash
# Incremental (only changed files)
code-vector-cli index --incremental

# Full reindex
code-vector-cli delete --force
code-vector-cli init
```
