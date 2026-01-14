"""Embedding models for code and natural language"""

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

# OpenAI imports (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class CodeEmbedder:
    """Embedder for code using CodeT5+"""

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
            import os
            num_threads = os.cpu_count()
            torch.set_num_threads(num_threads)  # Use all CPU cores

            # Additional PyTorch optimizations
            torch.set_num_interop_threads(4)  # Parallelism for operations

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

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for code"""
        self.load()

        if isinstance(texts, str):
            texts = [texts]

        # Process in sub-batches for memory efficiency and speed
        # Larger batches = better throughput on CPU
        batch_size = 256  # Increased for better CPU utilization
        all_embeddings = []

        import torch

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # CodeT5+ embedding model returns embeddings directly
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Model returns embeddings directly
                    embeddings = outputs

                all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings


class TextEmbedder:
    """Embedder for natural language using mpnet"""

    def __init__(self):
        self.model = None
        self.dimension = 768

    def load(self):
        """Lazy load the model"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for natural language"""
        self.load()

        if isinstance(texts, str):
            texts = [texts]

        # SentenceTransformer already handles batching efficiently
        # Set batch_size for better throughput
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=256,  # Process 256 texts at once for better throughput
            show_progress_bar=False
        )
        return embeddings.tolist()


class OpenAIEmbedder:
    """Embedder using OpenAI API"""

    def __init__(self, model=None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        # Use model from env var or default
        self.model_name = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = None
        self.dimension = 1536  # text-embedding-3-small dimension

    def load(self):
        """Initialize OpenAI client"""
        if self.client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)

    # Configuration constants
    MAX_CHARS_PER_TEXT = 24000  # ~6k tokens, safe under 8192 limit
    MAX_TOKENS_PER_REQUEST = 250000  # Safe buffer under 300k API limit
    MAX_BATCH_SIZE = 200  # Max texts per API call (increased for better throughput)
    MAX_CONCURRENT_REQUESTS = 3  # Parallel API requests (reduced to avoid rate limits)
    AGGRESSIVE_TRUNCATION = 12000  # Fallback for oversized texts

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using OpenAI API with robust error handling and parallel processing"""
        self.load()

        if isinstance(texts, str):
            texts = [texts]

        # Step 1: Sanitize and truncate all texts
        sanitized_texts = [self._sanitize_text(t) for t in texts]

        # Step 2: Batch texts intelligently based on token estimates
        batches = self._create_batches(sanitized_texts)

        # Step 3: Process batches in parallel for better throughput
        if len(batches) > 1:
            all_embeddings = self._process_batches_parallel(batches, len(sanitized_texts))
        else:
            # Single batch - no need for parallelization overhead
            all_embeddings = []
            self._process_batch(batches[0], 1, len(sanitized_texts), all_embeddings)

        return all_embeddings

    def _sanitize_text(self, text: str) -> str:
        """Sanitize and truncate text for OpenAI API"""
        # Remove problematic characters
        sanitized = text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
        sanitized = ''.join(
            char for char in sanitized
            if char in '\n\t' or ord(char) >= 32
        )

        # Ensure non-empty
        if not sanitized or sanitized.isspace():
            return "empty"

        # Truncate to safe length
        return sanitized[:self.MAX_CHARS_PER_TEXT]

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Create optimized batches with dynamic sizing based on text characteristics"""
        if not texts:
            return []

        # Calculate average text size to determine optimal batch size
        avg_chars = sum(len(t) for t in texts) / len(texts)
        avg_tokens = avg_chars // 4

        # Dynamic batch size based on text characteristics
        if avg_tokens < 100:
            # Very small texts (e.g., single-line comments, short functions)
            # Can fit many in a batch - optimize for fewer API calls
            dynamic_batch_size = min(1000, self.MAX_BATCH_SIZE * 5)
        elif avg_tokens < 500:
            # Small texts (e.g., small functions, config snippets)
            # Good balance - use larger batches
            dynamic_batch_size = min(500, self.MAX_BATCH_SIZE * 2)
        elif avg_tokens < 2000:
            # Medium texts (e.g., typical functions/classes)
            # Use standard batch size
            dynamic_batch_size = self.MAX_BATCH_SIZE
        else:
            # Large texts (e.g., big classes, full files)
            # Use smaller batches to avoid token limits
            dynamic_batch_size = max(20, self.MAX_BATCH_SIZE // 2)

        batches = []
        current_batch = []
        current_chars = 0

        for text in texts:
            text_chars = len(text)
            estimated_tokens = text_chars // 4

            # Start new batch if limits would be exceeded
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

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def _process_batch(self, batch, batch_idx, total_texts, all_embeddings):
        """Process a single batch with retry logic and error recovery"""
        import time
        import re
        from openai import APIError, APITimeoutError, RateLimitError, BadRequestError

        # Show progress for large operations
        if total_texts > 1000:
            tokens = sum(len(t) for t in batch) // 4
            print(f"    API batch {batch_idx} ({len(batch)} texts, ~{tokens:,} tokens)...")

        max_retries = 3
        max_rate_limit_retries = 10  # More retries for rate limits
        rate_limit_attempts = 0

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                    timeout=60.0
                )
                all_embeddings.extend([item.embedding for item in response.data])
                return  # Success

            except BadRequestError as e:
                if "maximum context length" in str(e):
                    self._handle_token_limit_error(batch, all_embeddings, e)
                    return  # Successfully recovered
                else:
                    self._handle_bad_request_error(batch, e)

            except RateLimitError as e:
                rate_limit_attempts += 1
                error_msg = str(e)

                # Extract wait time from error message if available
                # e.g., "Please try again in 221ms"
                wait_time = 1  # Default 1 second
                match = re.search(r'try again in (\d+)ms', error_msg)
                if match:
                    wait_ms = int(match.group(1))
                    wait_time = max(1, (wait_ms / 1000) + 0.5)  # Convert to seconds, add buffer
                else:
                    # Use exponential backoff if no specific wait time given
                    wait_time = min(60, 2 ** rate_limit_attempts)  # Cap at 60s

                if rate_limit_attempts < max_rate_limit_retries:
                    print(f"    Rate limit hit (attempt {rate_limit_attempts}/{max_rate_limit_retries}), waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # Don't increment main retry counter for rate limits
                    attempt -= 1  # Keep retrying on rate limits
                else:
                    print(f"    FATAL: Rate limit exceeded after {max_rate_limit_retries} attempts")
                    raise

            except (APITimeoutError, APIError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"    API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    print(f"    Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Log problematic batch info before failing
                    print(f"    Failed batch info: {len(batch)} texts, first text length: {len(batch[0]) if batch else 0}")
                    if batch:
                        print(f"    First text preview: {batch[0][:100].encode('unicode_escape').decode('ascii')}")
                    raise  # Failed all retries

    def _handle_token_limit_error(self, batch, all_embeddings, original_error):
        """Handle token limit errors by truncating or splitting batch"""
        if len(batch) == 1:
            # Single oversized text - truncate aggressively
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
            # One text in batch is oversized - process individually
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

        all_embeddings = [None] * len(batches)  # Preserve order

        def process_batch_wrapper(batch_info):
            """Wrapper to process a batch and return its index and results"""
            batch_idx, batch = batch_info
            embeddings = []
            self._process_batch(batch, batch_idx + 1, total_texts, embeddings)
            return batch_idx, embeddings

        # Process batches in parallel with limited concurrency
        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_REQUESTS) as executor:
            # Submit all batches
            future_to_idx = {
                executor.submit(process_batch_wrapper, (idx, batch)): idx
                for idx, batch in enumerate(batches)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                try:
                    batch_idx, embeddings = future.result()
                    all_embeddings[batch_idx] = embeddings
                    completed += 1

                    # Show overall progress
                    if total_texts > 1000:
                        progress_pct = (completed * 100) // len(batches)
                        print(f"    Progress: {completed}/{len(batches)} batches ({progress_pct}%)")

                except Exception as e:
                    # Cancel remaining futures on error
                    for f in future_to_idx:
                        f.cancel()
                    raise RuntimeError(f"Batch processing failed: {e}") from e

        # Flatten results while preserving order
        result = []
        for embeddings in all_embeddings:
            if embeddings:
                result.extend(embeddings)

        return result


# Global instances
_code_embedder = None
_text_embedder = None
_openai_embedder = None


def get_code_embedder(use_openai=False) -> Union[CodeEmbedder, OpenAIEmbedder]:
    """Get or create the global code embedder instance"""
    global _code_embedder, _openai_embedder

    # Check environment variable for OpenAI preference
    if use_openai or os.environ.get("USE_OPENAI_EMBEDDINGS", "").lower() == "true":
        if _openai_embedder is None:
            _openai_embedder = OpenAIEmbedder()
        return _openai_embedder
    else:
        if _code_embedder is None:
            _code_embedder = CodeEmbedder()
        return _code_embedder


def get_text_embedder(use_openai=False) -> Union[TextEmbedder, OpenAIEmbedder]:
    """Get or create the global text embedder instance"""
    global _text_embedder, _openai_embedder

    # Check environment variable for OpenAI preference
    if use_openai or os.environ.get("USE_OPENAI_EMBEDDINGS", "").lower() == "true":
        if _openai_embedder is None:
            _openai_embedder = OpenAIEmbedder()
        return _openai_embedder
    else:
        if _text_embedder is None:
            _text_embedder = TextEmbedder()
        return _text_embedder
