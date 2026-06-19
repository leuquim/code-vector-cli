"""Embedding models for code and natural language.

Provider is a single configurable choice via environment variables:

    EMBEDDING_PROVIDER = zeroentropy | openai | local   (default: zeroentropy)
    EMBEDDING_DIMENSIONS = <int>                          (provider-specific)

- zeroentropy (default): cloud zembed-1, a single model that handles BOTH code
  and text, so the historical two-model (code + text) split collapses into one
  embedder. Supports asymmetric retrieval via input_type ("query" vs "document").
- openai: text-embedding-3-* cloud models.
- local: CodeT5+ (code) + mpnet (text) run on CPU. No API key, fully offline.

All embed() methods accept input_type="query"|"document". Only ZeroEntropy uses
it (asymmetric retrieval); the others ignore it. Index paths must pass
input_type="document" (the default); query paths must pass input_type="query".
"""

import os
from typing import List, Union
from pathlib import Path
# Imports moved to lazy loading in classes to improve CLI startup speed


# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in home directory or current directory
    env_paths = [
        Path.home() / '.code-vector-db.env',
        Path.cwd() / '.env',
        Path.home() / '.env'
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass  # dotenv not installed, use system env vars only

# Set cache directory
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.local/share/code-vector-db/models')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.local/share/code-vector-db/models')

# Optimize CPU parallelism
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())


# ---------------------------------------------------------------------------
# Provider / dimension configuration
# ---------------------------------------------------------------------------

# ZeroEntropy zembed-1 only supports these output dimensions (Matryoshka).
ZEMBED_VALID_DIMENSIONS = [2560, 1280, 640, 320, 160, 80, 40]
ZEMBED_DEFAULT_DIMENSION = 1280
ZEMBED_MODEL = "zembed-1"

# OpenAI dimensions (text-embedding-3-small native = 1536).
OPENAI_DEFAULT_DIMENSION = 1536


def get_provider() -> str:
    """Return the configured embedding provider.

    Precedence:
    1. EMBEDDING_PROVIDER (zeroentropy | openai | local)
    2. Legacy USE_OPENAI_EMBEDDINGS=true -> openai
    3. Default: zeroentropy
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.environ.get("USE_OPENAI_EMBEDDINGS", "").lower() == "true":
        return "openai"
    return "zeroentropy"


def get_embedding_dimension() -> int:
    """Return the active embedding dimension for the configured provider.

    This is the single source of truth the vector store uses to size every
    collection. Code and text share one dimension because zembed-1 is a single
    model for both.
    """
    provider = get_provider()
    configured = os.environ.get("EMBEDDING_DIMENSIONS", "").strip()

    if provider == "zeroentropy":
        if configured:
            dim = int(configured)
            if dim not in ZEMBED_VALID_DIMENSIONS:
                raise ValueError(
                    f"EMBEDDING_DIMENSIONS={dim} is invalid for zembed-1. "
                    f"Choose one of: {ZEMBED_VALID_DIMENSIONS}"
                )
            return dim
        return ZEMBED_DEFAULT_DIMENSION

    if provider == "openai":
        return int(configured) if configured else OPENAI_DEFAULT_DIMENSION

    # local: code and text models have fixed, different native dimensions.
    # We don't unify them; the vector store handles the per-collection split
    # only in local mode (see get_code_dimension / get_text_dimension).
    return int(configured) if configured else 0  # 0 = "use model-native sizes"


def get_code_dimension() -> int:
    """Dimension for code collections (provider-aware)."""
    provider = get_provider()
    if provider == "local":
        return 256  # CodeT5+ native
    return get_embedding_dimension()


def get_text_dimension() -> int:
    """Dimension for text collections (provider-aware)."""
    provider = get_provider()
    if provider == "local":
        return 768  # mpnet native
    return get_embedding_dimension()


# ---------------------------------------------------------------------------
# Local embedders (offline, CPU)
# ---------------------------------------------------------------------------

class CodeEmbedder:
    """Local embedder for code using CodeT5+ (offline, CPU)."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.dimension = 256

    def load(self):
        """Lazy load the model"""
        if self.model is None:
            # Lazy import heavy dependencies
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Optimize PyTorch for CPU inference - use all available cores
            num_threads = os.cpu_count()
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(4)

            self.model = AutoModel.from_pretrained(
                "Salesforce/codet5p-110m-embedding",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Salesforce/codet5p-110m-embedding",
                trust_remote_code=True
            )
            self.model.eval()

            print(f"  CodeEmbedder using {num_threads} CPU threads")

    def embed(self, texts: Union[str, List[str]], input_type: str = "document") -> List[List[float]]:
        """Generate embeddings for code. input_type is ignored (local model)."""
        self.load()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = 256
        all_embeddings = []

        import torch

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    embeddings = outputs

                all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings


class TextEmbedder:
    """Local embedder for natural language using mpnet (offline, CPU)."""

    def __init__(self):
        self.model = None
        self.dimension = 768

    def load(self):
        """Lazy load the model"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def embed(self, texts: Union[str, List[str]], input_type: str = "document") -> List[List[float]]:
        """Generate embeddings for text. input_type is ignored (local model)."""
        self.load()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=256,
            show_progress_bar=False
        )
        return embeddings.tolist()


# ---------------------------------------------------------------------------
# ZeroEntropy zembed-1 (cloud, single model for code + text)
# ---------------------------------------------------------------------------

class ZeroEntropyEmbedder:
    """Cloud embedder using ZeroEntropy zembed-1.

    A single model for both code and text. Supports asymmetric retrieval:
    index chunks with input_type="document", search queries with
    input_type="query".
    """

    # Batching / resilience
    MAX_BATCH_SIZE = 128          # texts per API call
    MAX_CHARS_PER_TEXT = 28000    # ~7k tokens, under the 32k context window
    MAX_RETRIES = 5

    def __init__(self, model: str = ZEMBED_MODEL, dimensions: int = None):
        self.model_name = model
        self.dimensions = dimensions or get_embedding_dimension()
        if self.dimensions not in ZEMBED_VALID_DIMENSIONS:
            raise ValueError(
                f"dimensions={self.dimensions} invalid for {self.model_name}. "
                f"Choose one of: {ZEMBED_VALID_DIMENSIONS}"
            )
        self.dimension = self.dimensions  # alias for parity with other embedders
        self._client = None

    def load(self):
        """Initialize the ZeroEntropy client."""
        if self._client is None:
            try:
                from zeroentropy import ZeroEntropy
            except ImportError:
                raise ImportError(
                    "zeroentropy package not installed. Install with: pip install zeroentropy"
                )
            api_key = os.environ.get("ZEROENTROPY_API_KEY")
            if not api_key:
                raise ValueError(
                    "ZEROENTROPY_API_KEY environment variable not set. "
                    "Add it to ~/.code-vector-db.env"
                )
            self._client = ZeroEntropy(api_key=api_key)

    @property
    def client(self):
        self.load()
        return self._client

    def _sanitize(self, text: str) -> str:
        if not text or text.isspace():
            return "empty"
        return text[:self.MAX_CHARS_PER_TEXT]

    def embed(self, texts: Union[str, List[str]], input_type: str = "document") -> List[List[float]]:
        """Generate embeddings via zembed-1.

        Args:
            texts: a string or list of strings to embed.
            input_type: "document" for indexed content, "query" for searches.
        """
        if input_type not in ("query", "document"):
            raise ValueError("input_type must be 'query' or 'document'")

        self.load()

        if isinstance(texts, str):
            texts = [texts]

        sanitized = [self._sanitize(t) for t in texts]

        all_embeddings: List[List[float]] = []
        for i in range(0, len(sanitized), self.MAX_BATCH_SIZE):
            batch = sanitized[i:i + self.MAX_BATCH_SIZE]
            all_embeddings.extend(self._embed_batch(batch, input_type, i // self.MAX_BATCH_SIZE, len(sanitized)))

        return all_embeddings

    def _embed_batch(self, batch: List[str], input_type: str, batch_idx: int, total: int) -> List[List[float]]:
        import time

        if total > 500:
            print(f"    zembed batch {batch_idx + 1} ({len(batch)} texts, {input_type})...")

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.client.models.embed(
                    model=self.model_name,
                    input=batch,
                    input_type=input_type,
                    dimensions=self.dimensions,
                )
                return [r.embedding for r in resp.results]
            except Exception as e:
                msg = str(e)
                is_rate = "429" in msg or "rate" in msg.lower()
                if attempt < self.MAX_RETRIES - 1:
                    wait = min(60, 2 ** attempt) if is_rate else 2 ** attempt
                    print(f"    zembed API error (attempt {attempt + 1}/{self.MAX_RETRIES}): {msg[:120]}")
                    print(f"    retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"zembed embedding failed after {self.MAX_RETRIES} attempts "
                        f"(batch of {len(batch)}). Aborting to avoid an incomplete index."
                    ) from e


# ---------------------------------------------------------------------------
# OpenAI embedder (cloud)
# ---------------------------------------------------------------------------

# Probe availability without importing: `import openai` costs ~1.2s and would
# be paid on every CLI invocation even when the provider is zeroentropy.
import importlib.util

OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None


class OpenAIEmbedder:
    """Embedder using OpenAI API."""

    def __init__(self, model=None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        self.model_name = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = None
        self.dimension = get_embedding_dimension() if get_provider() == "openai" else OPENAI_DEFAULT_DIMENSION

    def load(self):
        """Initialize OpenAI client"""
        if self.client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)

    # Configuration constants
    MAX_CHARS_PER_TEXT = 24000
    MAX_TOKENS_PER_REQUEST = 250000
    MAX_BATCH_SIZE = 200
    MAX_CONCURRENT_REQUESTS = 3
    AGGRESSIVE_TRUNCATION = 12000

    def embed(self, texts: Union[str, List[str]], input_type: str = "document") -> List[List[float]]:
        """Generate embeddings using OpenAI API. input_type is ignored."""
        self.load()

        if isinstance(texts, str):
            texts = [texts]

        sanitized_texts = [self._sanitize_text(t) for t in texts]
        batches = self._create_batches(sanitized_texts)

        if len(batches) > 1:
            all_embeddings = self._process_batches_parallel(batches, len(sanitized_texts))
        else:
            all_embeddings = []
            self._process_batch(batches[0], 1, len(sanitized_texts), all_embeddings)

        return all_embeddings

    def _sanitize_text(self, text: str) -> str:
        """Sanitize and truncate text for OpenAI API"""
        sanitized = text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
        sanitized = ''.join(
            char for char in sanitized
            if char in '\n\t' or ord(char) >= 32
        )

        if not sanitized or sanitized.isspace():
            return "empty"

        return sanitized[:self.MAX_CHARS_PER_TEXT]

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Create optimized batches with dynamic sizing based on text characteristics"""
        if not texts:
            return []

        avg_chars = sum(len(t) for t in texts) / len(texts)
        avg_tokens = avg_chars // 4

        if avg_tokens < 100:
            dynamic_batch_size = min(1000, self.MAX_BATCH_SIZE * 5)
        elif avg_tokens < 500:
            dynamic_batch_size = min(500, self.MAX_BATCH_SIZE * 2)
        elif avg_tokens < 2000:
            dynamic_batch_size = self.MAX_BATCH_SIZE
        else:
            dynamic_batch_size = max(20, self.MAX_BATCH_SIZE // 2)

        batches = []
        current_batch = []
        current_chars = 0

        for text in texts:
            text_chars = len(text)
            estimated_tokens = text_chars // 4

            would_exceed = (
                len(current_batch) >= dynamic_batch_size or
                (current_batch and current_chars // 4 + estimated_tokens > self.MAX_TOKENS_PER_REQUEST)
            )

            if would_exceed:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(text)
            current_chars += text_chars

        if current_batch:
            batches.append(current_batch)

        return batches

    def _process_batch(self, batch, batch_idx, total_texts, all_embeddings):
        """Process a single batch with retry logic and error recovery"""
        import time
        import re
        from openai import APIError, APITimeoutError, RateLimitError, BadRequestError

        if total_texts > 1000:
            tokens = sum(len(t) for t in batch) // 4
            print(f"    API batch {batch_idx} ({len(batch)} texts, ~{tokens:,} tokens)...")

        max_retries = 3
        max_rate_limit_retries = 10
        rate_limit_attempts = 0

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                    timeout=60.0
                )
                all_embeddings.extend([item.embedding for item in response.data])
                return

            except BadRequestError as e:
                if "maximum context length" in str(e):
                    self._handle_token_limit_error(batch, all_embeddings, e)
                    return
                else:
                    self._handle_bad_request_error(batch, e)

            except RateLimitError as e:
                rate_limit_attempts += 1
                error_msg = str(e)

                wait_time = 1
                match = re.search(r'try again in (\d+)ms', error_msg)
                if match:
                    wait_ms = int(match.group(1))
                    wait_time = max(1, (wait_ms / 1000) + 0.5)
                else:
                    wait_time = min(60, 2 ** rate_limit_attempts)

                if rate_limit_attempts < max_rate_limit_retries:
                    print(f"    Rate limit hit (attempt {rate_limit_attempts}/{max_rate_limit_retries}), waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    attempt -= 1
                else:
                    print(f"    FATAL: Rate limit exceeded after {max_rate_limit_retries} attempts")
                    raise

            except (APITimeoutError, APIError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"    API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    print(f"    Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    Failed batch info: {len(batch)} texts, first text length: {len(batch[0]) if batch else 0}")
                    if batch:
                        print(f"    First text preview: {batch[0][:100].encode('unicode_escape').decode('ascii')}")
                    raise

    def _handle_token_limit_error(self, batch, all_embeddings, original_error):
        """Handle token limit errors by truncating or splitting batch"""
        if len(batch) == 1:
            print(f"    ERROR: Text too large ({len(batch[0])} chars), truncating to {self.AGGRESSIVE_TRUNCATION}...")
            try:
                response = self.client.embeddings.create(
                    input=[batch[0][:self.AGGRESSIVE_TRUNCATION]],
                    model=self.model_name,
                    timeout=60.0
                )
                all_embeddings.extend([item.embedding for item in response.data])
                print(f"    [OK] Successfully embedded truncated text")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot embed text even after aggressive truncation. "
                    f"Text length: {len(batch[0])} chars. Aborting to prevent incomplete index."
                ) from original_error
        else:
            print(f"    ERROR: Oversized text in batch, processing {len(batch)} texts individually...")
            for i, text in enumerate(batch):
                try:
                    truncated = text[:self.AGGRESSIVE_TRUNCATION]
                    response = self.client.embeddings.create(
                        input=[truncated],
                        model=self.model_name,
                        timeout=60.0
                    )
                    all_embeddings.extend([item.embedding for item in response.data])
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to embed text {i+1}/{len(batch)} "
                        f"(length: {len(text)} chars). Aborting to prevent incomplete index."
                    ) from e
            print(f"    [OK] Successfully processed all {len(batch)} texts individually")

    def _handle_bad_request_error(self, batch, error):
        """Handle non-token-limit bad request errors"""
        error_msg = str(error)
        print(f"    FATAL: Bad request error")
        print(f"    Batch size: {len(batch)} texts")
        print(f"    Error: {error_msg[:300]}")
        if batch:
            print(f"    First text: {len(batch[0])} chars, preview: {batch[0][:100]}")
        raise RuntimeError(
            f"Cannot embed batch due to bad request. Aborting to prevent incomplete index."
        ) from error

    def _process_batches_parallel(self, batches, total_texts):
        """Process multiple batches in parallel using ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_embeddings = [None] * len(batches)

        def process_batch_wrapper(batch_info):
            batch_idx, batch = batch_info
            embeddings = []
            self._process_batch(batch, batch_idx + 1, total_texts, embeddings)
            return batch_idx, embeddings

        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS) as executor:
            future_to_idx = {
                executor.submit(process_batch_wrapper, (idx, batch)): idx
                for idx, batch in enumerate(batches)
            }

            completed = 0
            for future in as_completed(future_to_idx):
                try:
                    batch_idx, embeddings = future.result()
                    all_embeddings[batch_idx] = embeddings
                    completed += 1

                    if total_texts > 1000:
                        progress_pct = (completed * 100) // len(batches)
                        print(f"    Progress: {completed}/{len(batches)} batches ({progress_pct}%)")

                except Exception as e:
                    for f in future_to_idx:
                        f.cancel()
                    raise RuntimeError(f"Batch processing failed: {e}") from e

        result = []
        for embeddings in all_embeddings:
            if embeddings:
                result.extend(embeddings)

        return result


# ---------------------------------------------------------------------------
# Factory / singletons
# ---------------------------------------------------------------------------

_code_embedder = None
_text_embedder = None
_openai_embedder = None
_zembed_embedder = None


def _get_zembed():
    global _zembed_embedder
    if _zembed_embedder is None:
        _zembed_embedder = ZeroEntropyEmbedder()
    return _zembed_embedder


def get_code_embedder(use_openai=False):
    """Get the code embedder for the configured provider.

    With provider=zeroentropy (default), code and text share one zembed-1
    instance. use_openai is kept for backward compatibility.
    """
    global _code_embedder, _openai_embedder

    provider = "openai" if use_openai else get_provider()

    if provider == "zeroentropy":
        return _get_zembed()
    if provider == "openai":
        if _openai_embedder is None:
            _openai_embedder = OpenAIEmbedder()
        return _openai_embedder
    # local
    if _code_embedder is None:
        _code_embedder = CodeEmbedder()
    return _code_embedder


def get_text_embedder(use_openai=False):
    """Get the text embedder for the configured provider.

    With provider=zeroentropy (default), code and text share one zembed-1
    instance.
    """
    global _text_embedder, _openai_embedder

    provider = "openai" if use_openai else get_provider()

    if provider == "zeroentropy":
        return _get_zembed()
    if provider == "openai":
        if _openai_embedder is None:
            _openai_embedder = OpenAIEmbedder()
        return _openai_embedder
    # local
    if _text_embedder is None:
        _text_embedder = TextEmbedder()
    return _text_embedder
