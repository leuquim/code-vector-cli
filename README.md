# Code Vector CLI

A fast, local semantic code search tool powered by vector embeddings. Index your codebase once, then search using natural language queries to find relevant code instantly.

## Features

- **Semantic Code Search**: Find code by meaning, not just keywords
- **Incremental Indexing**: Smart file change detection - only reindex modified files
- **Multi-Repository Support**: Index entire workspaces with multiple projects
- **AST-Aware Chunking**: Intelligently splits code at function/class boundaries using Tree-sitter
- **Fast Local or Cloud Embeddings**: Choose between local models (CodeT5+, mpnet) or OpenAI embeddings
- **Qdrant Vector Database**: High-performance vector storage with collections for functions, classes, and files
- **Cross-Repo Search**: Search across all indexed repositories simultaneously
- **Impact Analysis**: Analyze dependencies and find code affected by changes
- **Context Selection**: AI-powered file selection for specific tasks
- **Similarity Search**: Find similar code patterns across your codebase

## Installation

### Prerequisites

- Python 3.8+
- [Qdrant](https://qdrant.tech/) vector database running locally (or remote)
- Git (for multi-repo workspace detection)

### Install Qdrant

```bash
# Using Docker (recommended)
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Or using native binary - see https://qdrant.tech/documentation/quick-start/
```

### Install Code Vector CLI

#### Development Installation (Recommended)

For active development with immediate code changes:

```bash
# Clone the repository
git clone https://github.com/leuquim/code-vector-cli.git
cd code-vector-cli

# Install in editable/development mode
pip install -e .
```

With editable mode, any changes you make to the code take effect immediately without reinstalling.

#### Standard Installation

```bash
# Install from PyPI (once published)
pip install code-vector-cli

# Or install directly from GitHub
pip install git+https://github.com/leuquim/code-vector-cli.git
```

## Quick Start

### 1. Index Your Codebase

```bash
# Index a single project
code-vector-cli index /path/to/your/project

# Index a multi-repo workspace (auto-detects git repos)
code-vector-cli index /path/to/workspace
```

### 2. Search Your Code

```bash
# Search for code related to "authentication logic"
code-vector-cli search /path/to/your/project "authentication logic" --limit 5

# Search only in functions
code-vector-cli search /path/to/your/project "error handling" --collection functions

# Search across all collections
code-vector-cli search /path/to/your/project "database connection"
```

### 3. View Index Statistics

```bash
code-vector-cli stats /path/to/your/project
```

## Configuration

### Using OpenAI Embeddings (Faster, Requires API Key)

For better performance on large codebases, you can use OpenAI embeddings:

```bash
# Create ~/.code-vector-db.env
echo "USE_OPENAI_EMBEDDINGS=true" >> ~/.code-vector-db.env
echo "OPENAI_API_KEY=sk-your-api-key-here" >> ~/.code-vector-db.env
```

**Performance Comparison** (5000+ PHP files):
- Local embeddings (CodeT5+): ~12-15 minutes
- OpenAI embeddings: ~2-4 minutes

### Using Local Embeddings (Free, No API Key Needed)

By default, the tool uses local models:
- **Code**: Salesforce/codet5p-110m-embedding (256 dimensions)
- **Text**: sentence-transformers/all-mpnet-base-v2 (768 dimensions)

Models are automatically downloaded on first use to `~/.local/share/code-vector-db/models/`.

### Advanced Configuration

Create `~/.code-vector-db.env` to customize:

```bash
# OpenAI settings
USE_OPENAI_EMBEDDINGS=true
OPENAI_API_KEY=sk-your-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Qdrant connection
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Usage

All commands support the `--path` flag to specify the project directory. If omitted, uses current directory.

### Indexing Commands

```bash
# Initialize and index a new project
code-vector-cli init --path /path/to/project

# Index or update existing index
code-vector-cli index --path /path/to/project

# Incremental indexing (only changed files)
code-vector-cli index --path /path/to/project --incremental

# Index specific repository in multi-repo workspace
code-vector-cli index --path /path/to/workspace --repo frontend

# Reindex a single file
code-vector-cli reindex-file --path /path/to/project /relative/path/to/file.py
```

### Search Commands

```bash
# Semantic search - find code by meaning
code-vector-cli search --path /path/to/project "authentication logic"

# Adjust result count and score threshold
code-vector-cli search --path /path/to/project "error handling" --limit 10 --threshold 0.3

# Show code snippets in results
code-vector-cli search --path /path/to/project "database queries" --show-content

# Show parent class/module information
code-vector-cli search --path /path/to/project "validation" --show-parent

# Adjust context lines when showing content
code-vector-cli search --path /path/to/project "api endpoints" --show-content --context-lines 5
```

### Similarity & Context Commands

```bash
# Find similar code to a specific file
code-vector-cli similar --path /path/to/project "src/utils/auth.py" --limit 5

# Find similar code by semantic description
code-vector-cli similar --path /path/to/project "rate limiting middleware"

# Get relevant context for a task (AI-powered file selection)
code-vector-cli context --path /path/to/project "fix authentication bug"

# Output context as JSON for tool integration
code-vector-cli context --path /path/to/project "add user permissions" --json

# Analyze impact of changes to a file
code-vector-cli impact --path /path/to/project "src/models/user.py"
```

### Documentation & History Search

```bash
# Search documentation
code-vector-cli search-docs --path /path/to/project "api setup"

# Search conversation history
code-vector-cli search-conversations --path /path/to/project "deployment issues"
```

### Management Commands

```bash
# View index statistics
code-vector-cli stats --path /path/to/project

# List all indexed projects
code-vector-cli list-projects

# Delete index for a project
code-vector-cli delete --path /path/to/project --force

# Clean up metadata for missing projects
code-vector-cli cleanup-metadata

# Install git post-commit hook for auto-indexing
code-vector-cli install-hook --path /path/to/project
```

## Architecture

### Collections

The tool creates separate Qdrant collections for different code granularities:

- **code_functions**: Individual functions/methods
- **code_classes**: Classes and their methods
- **code_files**: Entire files (when no AST available)
- **documentation**: Markdown and docs (future)
- **git_history**: Commit messages (future)
- **conversations**: Chat history (future)

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

## Troubleshooting

### Tree-sitter Version Issues

If you see `TypeError: __init__() takes exactly 1 argument (2 given)`:

```bash
pip install 'tree-sitter<0.21' --upgrade
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

If you switch between local and OpenAI embeddings, delete and reindex:

```bash
code-vector-cli delete /path/to/project --force
code-vector-cli index /path/to/project
```

## Development

### Project Structure

```
code-vector-cli/
├── code_vector_db/
│   ├── __init__.py
│   ├── embeddings.py          # Embedding models (local + OpenAI)
│   ├── vector_store.py        # Qdrant operations
│   ├── indexer.py             # Main indexing logic
│   ├── workspace_indexer.py   # Multi-repo support
│   ├── ast_chunker.py         # Tree-sitter parsing
│   ├── query.py               # Search interface
│   └── metadata.py            # Project tracking
├── bin/
│   └── code-vector-cli        # CLI entry point
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
- [ ] Hybrid search (vector + BM25 keyword) - In progress
- [ ] Support for documentation (Markdown, RST)
- [ ] Git history indexing (commit messages, diffs)
- [ ] Conversation history indexing (chat logs)
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
