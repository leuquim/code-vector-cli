from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="code-vector-cli",
    version="0.1.0",
    author="Miquel",
    author_email="",  # Add your email
    description="Fast local semantic code search powered by vector embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leuquim/code-vector-cli",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Text Processing :: Indexing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Cloud-first default: zembed-1 embeddings + zerank-1 rerank
        # (only pre-release published; >=0.1.0a1 opts into accepting it)
        "zeroentropy>=0.1.0a1",
        "openai>=1.0.0",
        # AST parsing (maintained successor to tree-sitter-languages)
        "tree-sitter>=0.22.0",
        "tree-sitter-language-pack>=1.0.0",
        # Vector DB + hybrid search + plumbing
        "qdrant-client>=1.7.0",
        "bm25s>=0.2.0",
        "python-dotenv>=1.0.0",
        "mcp>=1.0.0",
    ],
    extras_require={
        # Offline local embeddings (no API key); pulls heavy ML deps.
        "local": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "code-vector-cli=code_vector_db.cli:main",
            "cvc=code_vector_db.cli:main",  # short alias
        ],
    },
    include_package_data=True,
    keywords="code search semantic vector embeddings qdrant ast tree-sitter",
    project_urls={
        "Bug Reports": "https://github.com/leuquim/code-vector-cli/issues",
        "Source": "https://github.com/leuquim/code-vector-cli",
    },
)
