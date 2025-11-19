---
description: Semantic code search using vector embeddings for finding relevant code across the codebase
triggers:
  - find code
  - search codebase
  - similar code
  - what code handles
  - where is implemented
  - code context
---

# Vector Search Skill

Use code-vector-cli to search the indexed codebase semantically. This tool finds code by meaning, not just keywords.

## When to Use

- User asks to find specific functionality
- Need to locate code that handles certain logic
- Want to find similar code patterns
- Need context for implementing features
- Analyzing impact of changes

## Available Commands

### Semantic Search
```bash
code-vector-cli search "query" [--limit N] [--threshold 0.3] [--show-content]
```
Finds code matching the semantic meaning of the query.

### Hybrid Search (Semantic + Keyword)
```bash
code-vector-cli search-hybrid "query" [--semantic-weight 0.7] [--bm25-weight 0.3]
```
Combines semantic similarity with keyword matching. Useful for finding specific function names or terms.

### Find Similar Code
```bash
code-vector-cli similar "path/to/file.py" [--limit 5]
```
Finds code similar to a specific file or description.

### Get Context for Task
```bash
code-vector-cli context "task description" [--json]
```
AI-powered selection of relevant files for a specific task.

### Impact Analysis
```bash
code-vector-cli impact "path/to/file.py"
```
Analyzes dependencies and finds code that might be affected by changes.

### Search Conversations
```bash
code-vector-cli search-conversations "deployment issues"
```
Search through past Claude Code sessions (requires conversation indexing).

## Usage Pattern

1. **First, check if codebase is indexed:**
   ```bash
   code-vector-cli stats
   ```

2. **If not indexed, index it:**
   ```bash
   code-vector-cli index
   ```

3. **Then search:**
   ```bash
   code-vector-cli search "authentication logic" --show-content
   ```

## Tips

- Use `search` for general semantic queries
- Use `search-hybrid` when looking for specific function/class names
- Use `--show-content` to see code snippets in results
- Use `--threshold` to adjust result quality (lower = more results)
- Adjust hybrid search weights for better results:
  - More semantic: `--semantic-weight 0.8 --bm25-weight 0.2`
  - More keyword: `--semantic-weight 0.3 --bm25-weight 0.7`

## Example Workflow

When user asks "Where do we handle authentication?"

1. Search semantically:
   ```bash
   code-vector-cli search "authentication login user verification" --limit 5 --show-content
   ```

2. If too broad, use hybrid search with specific terms:
   ```bash
   code-vector-cli search-hybrid "authenticate login" --semantic-weight 0.5 --bm25-weight 0.5
   ```

3. Analyze related code:
   ```bash
   code-vector-cli similar "src/auth/login.py"
   ```
