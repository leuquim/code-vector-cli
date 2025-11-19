---
name: code-vector-search
description: Semantic code search using vector embeddings to find code by meaning, locate similar patterns, analyze impact, search git history and conversations. Use when user asks to find code, locate implementations, understand architecture, check what code handles something, find similar patterns, search past work, or analyze change impact.
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

If not indexed or stats show 0 points, inform the user they need to initialize:
- For single git repo: `code-vector-cli init`
- For workspace with multiple repos: `cd /path/to/workspace && code-vector-cli init`

Always let the user initiate indexing operations.

## Core Commands

### 1. Semantic Code Search

Find code by **meaning**, not exact keywords.

```bash
# Basic search
code-vector-cli search "authentication middleware" -n 10

# With code snippets
code-vector-cli search "S3 file upload" --show-content -C 5

# Adjust sensitivity
code-vector-cli search "database connection pooling" -t 0.2  # More results
code-vector-cli search "React hooks" -t 0.5  # Fewer, higher quality
```

**When to use:**
- "Find code that handles user authentication"
- "Where do we implement file uploads?"
- "Show me examples of API endpoints"

**Thresholds:**
- `0.1-0.2`: Broad search, many results
- `0.3` (default): Balanced
- `0.4-0.6`: Strict, only close matches

---

### 2. Hybrid Search (Semantic + Keyword)

Combines vector similarity with BM25 keyword matching for **better precision** on exact terms.

```bash
# Default weights: 70% semantic, 30% keyword
code-vector-cli search-hybrid "StorageService S3 bucket"

# Adjust weights
code-vector-cli search-hybrid "getUserById" --semantic-weight 0.5 --bm25-weight 0.5
```

**When to use:**
- Searching for specific class/function names
- Looking for exact technical terms (S3, Redis, JWT)
- Need balance between conceptual and literal matching

**Best for:**
- Function names: `getUserById`, `validateToken`
- Technical terms: `S3`, `MongoDB`, `WebSocket`
- Specific APIs: `Express.Router`, `React.useState`

---

### 3. Find Similar Code

Discover duplicate code, similar implementations, or related patterns.

```bash
# From a file path
code-vector-cli similar path/to/file.js -n 10

# From semantic description
code-vector-cli similar "error handling with try-catch" -t 0.7
```

**When to use:**
- "Find code similar to this file"
- "Are there other implementations like this?"
- "Show me related patterns"
- Refactoring to consolidate duplicates

**Thresholds:**
- `0.7` (default): Similar implementations
- `0.8-0.9`: Near duplicates
- `0.6`: Loosely related code

---

### 4. Context Gathering

Get relevant files for a specific task.

```bash
# Find files related to a feature
code-vector-cli context "implement user permissions system" -n 10

# JSON output for programmatic use
code-vector-cli context "add OAuth login" --json
```

**When to use:**
- Starting a new feature and need context
- "What files are relevant for X?"
- Building a mental model of a subsystem

**Output:** Ranked list of files with reasons why they're relevant.

---

### 5. Impact Analysis

Understand what code might be affected by changes.

```bash
# From a file path
code-vector-cli impact path/to/service.js -t 0.6

# From semantic description
code-vector-cli impact "authentication middleware changes"
```

**When to use:**
- Before refactoring shared code
- "What will break if I change this?"
- Understanding dependency blast radius

**Levels:**
- **Direct impacts**: High similarity (>0.75)
- **Indirect impacts**: Medium similarity (0.6-0.75)

**Note:** This is similarity-based, not true dependency tracking. Use alongside `grep` for imports.

---

### 6. Git History Search

Search commit messages semantically to understand **why** code exists.

```bash
# Find relevant commits
code-vector-cli search-git "S3 storage refactoring" -n 5

# Recent changes
code-vector-cli search-git "fix authentication bug" -t 0.3
```

**When to use:**
- "Why was this implemented?"
- "What commits relate to feature X?"
- Understanding architectural decisions
- Finding related bug fixes

**Indexing:**
```bash
# Initial index (run once)
code-vector-cli index-git  # Indexes last 1000 commits per repo
code-vector-cli index-git -n 5000  # Index more commits

# Incremental (daily - only new commits since last index)
code-vector-cli index-git --incremental
```

---

### 7. Conversation Search

Search past Claude conversations for context and decisions.

```bash
# Find past discussions
code-vector-cli search-conversations "database migration strategy" -n 5

# Lower threshold for exploratory search
code-vector-cli search-conversations "refactoring" -t 0.2
```

**When to use:**
- "What did we discuss about X?"
- Finding past architectural decisions
- Recalling implementation rationale

**Indexing:** Run once to populate from `.claude-transcripts`:
```bash
code-vector-cli migrate-conversations
```

---

### 8. Documentation Search

Search markdown docs, READMEs, and documentation files.

```bash
code-vector-cli search-docs "API authentication" -n 5
```

**When to use:**
- Finding relevant documentation
- Understanding documented patterns
- Locating API guides

---

## Maintenance Commands

### Indexing

```bash
# Full reindex (initial or after major changes)
code-vector-cli index

# Incremental (only changed files) - MUCH FASTER
code-vector-cli index --incremental

# Index specific repo in workspace
code-vector-cli index --repo cms

# Reindex single file
code-vector-cli reindex-file path/to/file.js
```

**Best practices:**
- Use `--incremental` for daily work (500x faster)
- Full reindex after major refactors or branch switches
- Check what changed: `git diff --name-only HEAD~1`

### Statistics

```bash
# View index status
code-vector-cli stats

# List all indexed projects
code-vector-cli list-projects -v
```

### Cleanup

```bash
# Remove stale project metadata
code-vector-cli cleanup-metadata

# Delete all data for current project
code-vector-cli delete --force
```

---

## Multi-Project Usage

When multiple projects are indexed:

```bash
# Search within specific project
cd /path/to/project1
code-vector-cli search "auth handler"

# View all indexed projects
code-vector-cli list-projects -v
```

Each project has isolated indexes stored in Qdrant.

---

## Workflow Patterns

### Pattern 1: Starting a New Feature

```bash
# 1. Gather context
code-vector-cli context "implement user roles and permissions" -n 15

# 2. Find examples
code-vector-cli search "authorization middleware" --show-content

# 3. Check similar implementations
code-vector-cli similar "role-based access control" -n 10

# 4. Search past discussions
code-vector-cli search-conversations "permissions system"
```

### Pattern 2: Refactoring Shared Code

```bash
# 1. Check impact
code-vector-cli impact path/to/shared-service.js

# 2. Find similar code (possible consolidation targets)
code-vector-cli similar path/to/shared-service.js -t 0.8

# 3. Review related commits
code-vector-cli search-git "shared service changes"
```

### Pattern 3: Understanding Unfamiliar Code

```bash
# 1. Search by concept
code-vector-cli search "websocket connection handling"

# 2. View similar implementations
code-vector-cli similar path/to/unfamiliar-file.js

# 3. Find documentation
code-vector-cli search-docs "websocket architecture"

# 4. Search past conversations
code-vector-cli search-conversations "websocket implementation"
```

### Pattern 4: Daily Development

```bash
# Morning: Update indexes with recent changes
code-vector-cli index --incremental        # Update code index
code-vector-cli index-git --incremental    # Index new commits/merges

# During work: Search as needed
code-vector-cli search "email sending"
code-vector-cli hybrid-search "validateEmail function"

# Before committing: Check impact
code-vector-cli impact path/to/modified-file.js
```

---

## Performance Tips

### Fast Operations (< 1s)
- `search` - Already indexed
- `search-hybrid` - Minimal overhead
- `similar` - Fast vector lookup
- `stats` - Metadata only

### Slow Operations (seconds to minutes)
- `init` - Full indexing (run once)
- `index` - Full reindex (use `--incremental` instead)
- `index-git` - Embedding 1000s of commits (run once, then use `--incremental`)
- `migrate-conversations` - Embedding conversation history (run once)

### Optimization
- **Always use `--incremental`** for daily indexing
- Batch searches in parallel when exploring
- Use higher thresholds (`-t 0.4`) to reduce noise
- Limit results (`-n 5`) when skimming

---

## Troubleshooting

### No results found

```bash
# 1. Check if indexed
code-vector-cli stats

# 2. Lower threshold
code-vector-cli search "query" -t 0.1

# 3. Try hybrid search
code-vector-cli search-hybrid "query"

# 4. Reindex if stale
code-vector-cli index --incremental
```

### Stale results

```bash
# Incremental reindex (fast)
code-vector-cli index --incremental

# Full reindex (slow but thorough)
code-vector-cli delete --force
code-vector-cli init
```

### Wrong project

```bash
# Check current project
code-vector-cli stats

# Navigate to correct project
cd /path/to/correct/project
```

---

## Implementation Notes

### Under the Hood
- **Vector DB:** Qdrant (localhost:6333)
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dims)
- **Storage:** `~/.local/share/code-vector-db/`
- **Metadata:** `~/.local/share/code-vector-db/indexes/project-registry.json`

### Collections
- `code_functions` - Function-level chunks
- `code_classes` - Class-level chunks
- `code_files` - File-level chunks
- `documentation` - Markdown/docs
- `git_history` - Commit messages
- `conversations` - Claude transcripts

### File Types Indexed
- **Code:** `.js`, `.ts`, `.py`, `.java`, `.go`, `.rs`, `.php`, `.rb`, etc.
- **Docs:** `.md`, `.rst`, `.txt`, `.adoc`
- **Config:** `.json`, `.yaml`, `.yml`, `.toml`, `.ini`

---

## Quick Reference

```bash
# Setup (once)
code-vector-cli init
code-vector-cli index-git
code-vector-cli migrate-conversations

# Daily use
code-vector-cli index --incremental           # Update code index
code-vector-cli index-git --incremental       # Index new commits
code-vector-cli search "concept" -n 10
code-vector-cli search-hybrid "exact term"
code-vector-cli similar path/to/file.js

# Investigation
code-vector-cli context "task description" -n 15
code-vector-cli impact path/to/file.js
code-vector-cli search-git "commit topic"
code-vector-cli search-conversations "discussion topic"

# Maintenance
code-vector-cli stats
code-vector-cli list-projects
code-vector-cli cleanup-metadata
```

---

## Advanced Usage

### Combining Results

```bash
# Find code + related commits + past discussions
code-vector-cli search "authentication system" > /tmp/code.txt
code-vector-cli search-git "auth changes" > /tmp/commits.txt
code-vector-cli search-conversations "auth design" > /tmp/convs.txt
cat /tmp/*.txt
```

### Filtering by Score

```bash
# Only high-confidence matches
code-vector-cli search "query" -t 0.5 | head -20

# Exploratory search
code-vector-cli search "query" -t 0.1 -n 50
```

### Multi-Repo Workflows

```bash
# Index all repos in workspace
cd /workspace/root
code-vector-cli init

# Search shows repo labels
code-vector-cli search "shared utility"
# Output: [repo1] path/to/util.js
#         [repo2] lib/util.js

# Index only one repo
code-vector-cli index --repo cms --incremental
```

---

## Best Practices for Claude

1. **Start broad, narrow down:** Begin with `search`, then use `similar` on promising results
2. **Use hybrid for names:** When searching for specific functions/classes
3. **Check context first:** Before implementing, run `context` to understand existing patterns
4. **Incremental is your friend:** Always use `--incremental` for daily indexing
5. **Combine with grep:** Use vector search for concepts, grep for exact strings
6. **Lower thresholds when exploring:** Use `-t 0.2` to discover unexpected connections
7. **Show snippets sparingly:** `--show-content` is verbose, use only when needed

---

## Skill Execution Strategy

When user requests code search:

1. **Identify intent:** Find code? Understand architecture? Check impact?
2. **Choose command:**
   - Concept → `search`
   - Exact term → `search-hybrid`
   - Similar code → `similar`
   - Task context → `context`
   - Change impact → `impact`
   - Why/when → `search-git` or `search-conversations`
3. **Set parameters:**
   - Start with defaults (`-n 10`, `-t 0.3`)
   - Adjust based on results
4. **Present results:** Summarize findings, show paths, suggest next steps
5. **Iterate:** If no results, lower threshold or try different command

Remember: Vector search finds **meaning**, not **text**. Use it for semantic queries, use grep for literal matches.
