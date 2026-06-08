# Code Vector CLI

A fast, local semantic code search tool powered by vector embeddings. Index your codebase once, then search using natural language queries to find relevant code instantly.

> **⚠️ AI-Built Tool Disclaimer**
> This tool was fully built using AI assistants (Claude Code) to serve personal use cases and workflows. While functional and actively used, it may contain bugs or edge cases. Contributions, bug reports, and feedback are welcome via GitHub Issues.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install Qdrant](#install-qdrant)
  - [Install Code Vector CLI](#install-code-vector-cli)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Default: zembed-1 + zerank-1](#default-zembed-1--zerank-1-cloud-recommended)
  - [OpenAI embeddings](#openai-embeddings)
  - [Local embeddings](#local-embeddings-offline-no-api-key)
  - [Qdrant connection](#qdrant-connection)
  - [Windows Configuration](#windows-configuration)
- [Usage](#usage)
  - [Indexing Commands](#indexing-commands)
  - [Search Commands](#search-commands)
  - [Similarity & Context Commands](#similarity--context-commands)
  - [Documentation & History Search](#documentation--history-search)
  - [Management Commands](#management-commands)
- [Architecture](#architecture)
- [Supported Languages](#supported-languages)
- [Claude Code Integration](#claude-code-integration)
  - [MCP Server](#mcp-server)
  - [Conversation Indexing](#conversation-indexing)
  - [Claude Code Skill](#claude-code-skill)
- [Development](#development)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Support](#support)

## Features

- **Semantic Code Search**: Find code by meaning, not just keywords
- **Cross-Encoder Reranking**: A zerank-1 rerank stage reorders results for precision (the single biggest quality lever)
- **Hybrid Search**: Fuse semantic similarity with BM25 keyword matching, then rerank
- **Code + Text in One Model**: zembed-1 embeds both code and prose, so a single embedding space covers functions, docs, and commits
- **Incremental Indexing**: Smart file change detection - only reindex modified files
- **Multi-Repository Support**: Index entire workspaces with multiple projects
- **AST-Aware Chunking**: Intelligently splits code at function/class boundaries using Tree-sitter (Python, JS/TS, C#, Go, Rust, Java, PHP, and more)
- **Cloud or Local Embeddings**: zembed-1 (default) or OpenAI by API, or fully offline local models
- **Contextual Retrieval**: Each chunk is embedded with its path/class/signature header so location is captured, not just the body
- **Qdrant Vector Database**: High-performance vector storage with collections for functions, classes, and files
- **Cross-Repo Search**: Search across all indexed repositories simultaneously
- **Impact Analysis**: Analyze dependencies and find code affected by changes
- **Context Selection**: AI-powered file selection for specific tasks
- **Similarity Search**: Find similar code patterns across your codebase
- **Cross-Platform**: Works on Windows, Linux, and macOS with shared indexes between Windows and WSL

## Installation

### Prerequisites

- Python 3.9+
- [Qdrant](https://qdrant.tech/) vector database running locally (or remote)
- Git (for multi-repo workspace detection)
- A ZeroEntropy API key for the default cloud embeddings ([free tier](https://www.zeroentropy.dev/)), or use `[local]` for offline

### Install Qdrant

```bash
# Using Docker Compose (recommended)
docker compose up -d

# Or using Docker directly
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Or using native binary - see https://qdrant.tech/documentation/quick-start/
```

### Install Code Vector CLI

```bash
# Install from PyPI (once published)
pip install code-vector-cli

# Or install directly from GitHub
pip install git+https://github.com/leuquim/code-vector-cli.git
```

#### Development Installation

For contributors or if you want immediate code changes without reinstalling:

```bash
# Clone the repository
git clone https://github.com/leuquim/code-vector-cli.git
cd code-vector-cli

# Install in editable/development mode
pip install -e .
```

## Quick Start

**Note:** All commands default to the current working directory. Use `--path /path/to/project` to specify a different location.

**Short alias:** `cvc` is installed as an alias for `code-vector-cli` - e.g. `cvc search "auth logic"`. All examples below work with either name.

### 1. Index Your Codebase

```bash
# Navigate to your project
cd /path/to/your/project

# Index the codebase (creates vector embeddings)
code-vector-cli index

# Or index a different location
code-vector-cli index --path /path/to/workspace
```

### 2. Search Your Code

```bash
# Semantic search - finds code by meaning
code-vector-cli search "authentication logic" --limit 5

# Hybrid search - combines semantic + keyword matching
code-vector-cli search-hybrid "getUserById function"
```

### 3. View Index Statistics

```bash
code-vector-cli stats
```

**Vector Database Location:** Index data is stored in `~/.local/share/code-vector-db/qdrant/` with per-project collections.

## Configuration

All configuration lives in `~/.code-vector-db.env`. The embedding provider is a
single choice: `zeroentropy` (default), `openai`, or `local`.

> Switching provider, model, or dimension changes the vector size. Collections
> are fixed-size at creation, so after a switch you must `delete --force` and
> reindex. The tool detects a mismatch and tells you this rather than corrupting
> search silently.

### Default: zembed-1 + zerank-1 (cloud, recommended)

zembed-1 embeds both code and text in one model; zerank-1 reranks results for
precision. Get a key at [zeroentropy.dev](https://www.zeroentropy.dev/) (free tier available).

```bash
# ~/.code-vector-db.env
EMBEDDING_PROVIDER=zeroentropy
ZEROENTROPY_API_KEY=ze_your_key_here
EMBEDDING_DIMENSIONS=1280      # one of: 2560, 1280, 640, 320, 160, 80, 40
RERANK_ENABLED=true            # zerank-1 reranking (default on for zeroentropy)
```

### OpenAI embeddings

```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small   # 1536 dims
```

### Local embeddings (offline, no API key)

Requires the optional ML dependencies: `pip install -e ".[local]"` (pulls torch +
transformers + sentence-transformers). Models download on first use to
`~/.local/share/code-vector-db/models/`.

```bash
EMBEDDING_PROVIDER=local
# Code: Salesforce/codet5p-110m-embedding (256d); Text: all-mpnet-base-v2 (768d)
```

### Qdrant connection

```bash
# Remote server mode (default). gRPC is preferred automatically to avoid
# Windows socket exhaustion during large indexing runs.
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334          # set if your gRPC port isn't REST port + 1
QDRANT_PREFER_GRPC=true

# Or embedded local mode (no server needed)
QDRANT_LOCAL=true
QDRANT_LOCAL_PATH=~/.local/share/code-vector-db/qdrant-local
```

### Windows Configuration

The CLI works natively on Windows. For best performance:

**1. Use IP address instead of hostname:**

Windows DNS resolution for `localhost` can be slow (~5 seconds per request). Use `127.0.0.1` instead:

```bash
# In ~/.code-vector-db.env
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
```

**2. Cross-platform index sharing (Windows + WSL):**

The CLI automatically normalizes paths so Windows and WSL share the same index. A project at `C:\projects\myapp` and `/mnt/c/projects/myapp` will use the same Qdrant collections.

**3. Running Qdrant on Windows:**

Option A - Docker Desktop (recommended):
```bash
docker compose up -d
```

Option B - WSL with port forwarding:
```bash
# In WSL, Qdrant binds to WSL's IP. From Windows, use:
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
```

Option C - Embedded local mode (no server):
```bash
# In ~/.code-vector-db.env
QDRANT_LOCAL=true
```
Note: Local mode is slower for searches (~10s vs ~2s) due to disk I/O.

## Usage

All commands operate on the current working directory by default. Use `--path /path/to/project` to target a different location.

### Indexing Commands

```bash
# Initialize and index a new project
code-vector-cli init

# Index or update existing index
code-vector-cli index

# Incremental indexing (only changed files since last index)
code-vector-cli index --incremental

# Index specific repository in multi-repo workspace
code-vector-cli index --repo frontend

# Reindex a single file
code-vector-cli reindex-file relative/path/to/file.py
```

### Search Commands

```bash
# Semantic search - find code by meaning
code-vector-cli search "authentication logic"

# Adjust result count and score threshold
code-vector-cli search "error handling" --limit 10 --threshold 0.3

# Show code snippets in results
code-vector-cli search "database queries" --show-content

# Show parent class/module information
code-vector-cli search "validation" --show-parent

# Adjust context lines when showing content
code-vector-cli search "api endpoints" --show-content --context-lines 5

# Hybrid search - combine semantic understanding with keyword matching
code-vector-cli search-hybrid "login user authentication"

# Adjust semantic vs keyword weights (defaults: semantic=0.7, keyword=0.3)
code-vector-cli search-hybrid "API rate limit" --semantic-weight 0.5 --bm25-weight 0.5

# Use hybrid search with more keyword focus for specific terms
code-vector-cli search-hybrid "getUserById" --semantic-weight 0.3 --bm25-weight 0.7
```

### Similarity & Context Commands

```bash
# Find similar code to a specific file
code-vector-cli similar "src/utils/auth.py" --limit 5

# Find similar code by semantic description
code-vector-cli similar "rate limiting middleware"

# Get relevant context for a task (AI-powered file selection)
code-vector-cli context "fix authentication bug"

# Output context as JSON for tool integration
code-vector-cli context "add user permissions" --json

# Analyze impact of changes to a file
code-vector-cli impact "src/models/user.py"
```

### Documentation & History Search

```bash
# Search documentation (markdown, config files)
code-vector-cli search-docs "api setup"

# Index git commit history
code-vector-cli index-git

# Search git commits by message or diff content
code-vector-cli search-git "authentication refactor"

# Index conversation transcripts (one-time setup)
code-vector-cli migrate-conversations

# Search conversation history (requires SessionEnd hook + migrate-conversations)
code-vector-cli search-conversations "deployment issues"
```

### Management Commands

```bash
# View index statistics
code-vector-cli stats

# List all indexed projects
code-vector-cli list-projects

# Delete index for current project
code-vector-cli delete --force

# Clean up metadata for missing projects
code-vector-cli cleanup-metadata

# Install git post-commit hook for auto-indexing
code-vector-cli install-hook
```

## Architecture

### Collections

The tool creates separate Qdrant collections for different code granularities:

- **code_functions**: Individual functions/methods
- **code_classes**: Classes and their methods
- **code_files**: Entire files (when no AST available)
- **documentation**: Markdown and docs
- **git_history**: Commit messages and diffs
- **conversations**: Claude Code chat history

### Multi-Repo Workspace

When indexing a directory with multiple git repositories:

1. Auto-detects all git repos in subdirectories
2. Creates repo metadata (name, path, main branch)
3. Tags all vectors with `repo_name` for filtering
4. Enables cross-repo search with `--repo` filter

### Chunking Strategy

Uses Tree-sitter AST parsing to intelligently chunk code:

- **Functions**: Extracted with full signature and body
- **Classes**: Split into class definition + individual methods
- **Fallback**: Character-based chunking for non-parseable files

### Performance Optimizations

- **Parallel Parsing**: Uses 50% of CPU cores (8 workers on 16-core CPU)
- **Batch Processing**: 800 files per batch
- **Parallel Embedding**: ThreadPoolExecutor for concurrent API calls (OpenAI)
- **Smart Batching**: Dynamic batch sizing based on text characteristics
- **Rate Limit Handling**: Automatic retry with exponential backoff

## Supported Languages

Currently optimized for:
- PHP
- Python
- JavaScript/TypeScript
- Go
- Rust
- Java

Additional languages can be added by configuring Tree-sitter grammars in `ast_chunker.py`.

## Claude Code Integration

This tool was designed to work seamlessly with [Claude Code](https://claude.ai/code), Anthropic's AI coding assistant, though it works standalone as well.

There are two integration modes:

| Mode | Best for |
|------|----------|
| **MCP Server** | AI-driven search - Claude calls tools directly, no CLI needed |
| **CLI + Skill** | Human-driven search, scripting, CI, maintenance tasks |

### MCP Server

The MCP (Model Context Protocol) server exposes all search capabilities as tools that Claude can call directly during a conversation - no manual CLI invocation needed.

**Starting the server:**

```bash
code-vector-cli mcp serve --path /path/to/your/project
```

**Configuring in Claude Code** (`.mcp.json` in your project root):

```json
{
  "mcpServers": {
    "code-vector": {
      "command": "code-vector-cli",
      "args": ["mcp", "serve", "--path", "/path/to/your/project"]
    }
  }
}
```

**Available MCP tools:**

| Tool | Description |
|------|-------------|
| `semantic_search` | Find code by natural language description |
| `hybrid_search` | Semantic + BM25 keyword matching (best for identifiers) |
| `find_similar` | Find duplicate or related code patterns |
| `get_context` | Get relevant files for a task or feature |
| `analyze_impact` | Find code affected by a change before refactoring |
| `search_docs` | Search project documentation |
| `get_stats` | View index statistics |

All tools accept a `repo` parameter to filter results by repository in multi-repo workspaces.

**When to use `semantic_search` vs `hybrid_search`:**
- Use `semantic_search` for conceptual queries: `"user authentication logic"`, `"error handling middleware"`
- Use `hybrid_search` when you know specific names: `"UpdatePausedTimeResource"`, `"handleCheckout function"`

### Conversation Indexing

To enable conversation search (index your Claude Code sessions):

1. **Enable conversation tracking** with a hook that saves transcripts to `.claude-transcripts/` in your project

   Add this to your `.claude/settings.json`:
   ```json
   {
     "hooks": {
       "SessionEnd": {
         "command": "mkdir -p .claude-transcripts && cat > .claude-transcripts/$(date +%Y%m%d-%H%M%S).jsonl",
         "input": "transcript"
       }
     }
   }
   ```

2. **Index conversations** (reads from `{project}/.claude-transcripts/`):
   ```bash
   code-vector-cli migrate-conversations
   ```

3. **Search conversations:**
   ```bash
   code-vector-cli search-conversations "deployment issues"
   ```

This allows you to search through past Claude Code sessions in your project to find solutions, decisions, and context from previous work.

### Claude Code Skill

A Claude Code skill is included in `.claude/skills/code-vector-search/SKILL.md` that provides Claude with direct access to this tool.

**Features:**
- YAML frontmatter with skill name, description, and allowed tools
- Automatic triggers for code search scenarios (model-invoked)
- 8 core commands: search, search-hybrid, similar, context, impact, search-git, search-conversations, search-docs
- Maintenance commands: indexing, stats, cleanup
- 4 workflow patterns: new feature, refactoring, understanding code, daily development
- Performance optimization tips
- Advanced usage examples
- Best practices for Claude execution
- Comprehensive troubleshooting guide

**Installation:**
```bash
# For all projects (global)
cp -r .claude/skills/code-vector-search ~/.claude/skills/

# Or keep it project-specific (already in this repo)
```

The skill is model-invoked-Claude automatically activates it when users ask questions matching the description (e.g., "find code that handles authentication").

## Troubleshooting

### Tree-sitter Version Issues

If you see `TypeError: __init__() takes exactly 1 argument (2 given)` you are on
the old, unmaintained `tree-sitter-languages` package. This tool now uses
`tree-sitter-language-pack` (works with tree-sitter >= 0.22):

```bash
pip uninstall -y tree-sitter-languages
pip install tree-sitter-language-pack 'tree-sitter>=0.22'
```

### Qdrant Connection Failed

Ensure Qdrant is running:

```bash
docker ps | grep qdrant
curl http://localhost:6333/collections
```

### OpenAI Rate Limits

The tool automatically retries with exponential backoff. For very large codebases, consider:

1. Reducing concurrent requests in `embeddings.py` (MAX_CONCURRENT_REQUESTS)
2. Using local embeddings instead
3. Indexing repositories individually with `--repo` flag

### Vector Dimension Mismatch

If you switch embedding provider, model, or dimension, the vector size changes
and existing collections become incompatible. The tool detects this and asks you
to delete and reindex:

```bash
code-vector-cli delete --force --path /path/to/project
code-vector-cli index --path /path/to/project
```

### Qdrant Unreachable

If a search reports a connection error to Qdrant, start the server
(`docker compose up -d`) or switch to embedded mode (`QDRANT_LOCAL=true`).

## Development

### Project Structure

```
code-vector-cli/
├── code_vector_db/
│   ├── __init__.py
│   ├── cli.py                 # CLI entry point and command definitions
│   ├── mcp_server.py          # MCP server (7 search tools via stdio)
│   ├── embeddings.py          # Embedding providers (zembed-1 / OpenAI / local)
│   ├── reranker.py            # zerank-1 cross-encoder rerank stage
│   ├── vector_store.py        # Qdrant operations (gRPC, dimension guard)
│   ├── indexer.py             # Main indexing logic
│   ├── workspace_indexer.py   # Multi-repo support
│   ├── ast_chunker.py         # Tree-sitter parsing (tree-sitter-language-pack)
│   ├── query.py               # Search interface (+ rerank, hybrid)
│   └── metadata.py            # Project tracking
├── bin/
│   └── code-vector-cli        # Shell wrapper
├── setup.py
├── requirements.txt
└── README.md
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (when available)
pytest
```

### Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Roadmap

- [x] Incremental indexing (only reindex changed files) ✅
- [x] Impact analysis (dependency tracking) ✅
- [x] Context selection for tasks ✅
- [x] Similarity search ✅
- [x] Hybrid search (vector + BM25 keyword) ✅
- [x] MCP server (Claude Code and Claude Desktop integration) ✅
- [x] Documentation search (Markdown, config files) ✅
- [x] Git history indexing (commit messages, diffs) ✅
- [x] Conversation history indexing (Claude Code chat logs) ✅
- [ ] VSCode extension
- [ ] Language server protocol (LSP) integration
- [ ] More language support (Ruby, C++, C#)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Qdrant](https://qdrant.tech/) - Vector database
- [Tree-sitter](https://tree-sitter.github.io/) - Incremental parsing
- [Salesforce CodeT5+](https://github.com/salesforce/CodeT5) - Code embeddings
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [OpenAI](https://openai.com/) - Optional cloud embeddings

## Support

For issues, questions, or contributions, please visit:
- GitHub Issues: https://github.com/leuquim/code-vector-cli/issues
- Discussions: https://github.com/leuquim/code-vector-cli/discussions
