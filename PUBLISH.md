# Publishing Code Vector CLI to GitHub

This document contains step-by-step instructions for publishing this tool to GitHub under the `leuquim` account.

## Prerequisites

1. GitHub account: `leuquim`
2. Git installed locally
3. GitHub CLI (`gh`) installed (optional, recommended)

## Steps to Publish

### Option A: Using GitHub CLI (Recommended)

```bash
# Navigate to the repository
cd /tmp/code-vector-cli-repo

# Initialize git repository
git init
git add .
git commit -m "Initial commit: Code Vector CLI v0.1.0

- Fast semantic code search powered by vector embeddings
- Support for local (CodeT5+, mpnet) and OpenAI embeddings
- Multi-repository workspace indexing
- AST-aware code chunking with Tree-sitter
- Qdrant vector database integration
- Comprehensive CLI with search, similar, context, and impact commands
"

# Create GitHub repository and push (requires gh CLI authenticated)
gh auth login  # If not already authenticated
gh repo create leuquim/code-vector-cli --public --source=. --remote=origin --push

# Repository will be created at: https://github.com/leuquim/code-vector-cli
```

### Option B: Using GitHub Web Interface

1. **Create Repository on GitHub**:
   - Go to https://github.com/new
   - Repository name: `code-vector-cli`
   - Description: "Fast local semantic code search powered by vector embeddings"
   - Make it Public
   - Do NOT initialize with README (we already have one)
   - Click "Create repository"

2. **Push Local Code**:
   ```bash
   cd /tmp/code-vector-cli-repo

   # Initialize git repository
   git init
   git add .
   git commit -m "Initial commit: Code Vector CLI v0.1.0

   - Fast semantic code search powered by vector embeddings
   - Support for local (CodeT5+, mpnet) and OpenAI embeddings
   - Multi-repository workspace indexing
   - AST-aware code chunking with Tree-sitter
   - Qdrant vector database integration
   - Comprehensive CLI with search, similar, context, and impact commands
   "

   # Add remote and push
   git branch -M main
   git remote add origin https://github.com/leuquim/code-vector-cli.git
   git push -u origin main
   ```

## Post-Publish Tasks

### 1. Set Repository Topics

Go to repository settings and add topics for discoverability:
- `code-search`
- `vector-embeddings`
- `semantic-search`
- `qdrant`
- `tree-sitter`
- `openai`
- `developer-tools`
- `cli`
- `python`

### 2. Enable GitHub Pages (Optional)

If you want to add documentation later:
- Settings → Pages
- Source: Deploy from a branch
- Branch: main / docs (if you add a docs folder)

### 3. Add Repository Description

In the repository main page, click "⚙️" next to "About" and add:
- Description: "Fast local semantic code search powered by vector embeddings"
- Website: (leave empty for now, or add docs URL later)
- Topics: (add the topics from step 1)

### 4. Create First Release

```bash
# Tag the release
git tag -a v0.1.0 -m "Initial release v0.1.0

Features:
- Semantic code search with natural language queries
- Multi-repository workspace support
- Local (CodeT5+, mpnet) and OpenAI embeddings
- AST-aware code chunking
- Qdrant vector database integration
"

# Push tags
git push origin v0.1.0

# Create release on GitHub
gh release create v0.1.0 \
  --title "Code Vector CLI v0.1.0" \
  --notes "See README for installation and usage instructions"
```

### 5. Update Your Profile README (Optional)

Add this project to your GitHub profile README:

```markdown
## 🔧 Tools

- **[code-vector-cli](https://github.com/leuquim/code-vector-cli)** - Fast local semantic code search powered by vector embeddings
```

## Publishing to PyPI (Future Step)

Once the repository is stable and you want to make it installable via `pip install code-vector-cli`:

1. **Create PyPI Account**: https://pypi.org/account/register/

2. **Install Build Tools**:
   ```bash
   pip install build twine
   ```

3. **Build Distribution**:
   ```bash
   cd /tmp/code-vector-cli-repo
   python -m build
   ```

4. **Upload to PyPI**:
   ```bash
   # Test on TestPyPI first
   python -m twine upload --repository testpypi dist/*

   # If everything works, upload to real PyPI
   python -m twine upload dist/*
   ```

5. **Update README**: Change installation instructions to use PyPI

## Next Steps After Publishing

1. **Monitor Issues**: Watch for bug reports and feature requests
2. **Improve Documentation**: Add examples, troubleshooting guides
3. **Write Blog Post**: Share your tool on dev.to, Medium, or your blog
4. **Share on Social Media**: Tweet, LinkedIn post about your tool
5. **Add CI/CD**: GitHub Actions for testing and releases
6. **Add Tests**: pytest for code quality
7. **Add More Languages**: Expand Tree-sitter language support

## Checklist

- [ ] Repository created on GitHub
- [ ] Code pushed to `main` branch
- [ ] Repository description and topics added
- [ ] First release (v0.1.0) created
- [ ] README displays correctly on GitHub
- [ ] LICENSE file is visible
- [ ] .gitignore working correctly (no .env, __pycache__ files)
- [ ] Star your own repository (optional but traditional 😄)
- [ ] Share with community

## Support

For help with GitHub:
- Documentation: https://docs.github.com
- GitHub CLI: https://cli.github.com/manual

Good luck with your new open source project! 🚀
