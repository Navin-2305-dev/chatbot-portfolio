"""
embeddings.py
─────────────
Shared Gemini embedding wrapper used by both build_db.py and app.py.
Keeping it in one place ensures the document/query preparation logic
is always identical at build time and at query time.
"""

import logging
from typing import List, Optional

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def is_retryable_genai_error(exc: Exception) -> bool:
    """Identify transient Gemini / network errors worth retrying."""
    status_code = getattr(exc, "status_code", None)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    error_text = str(exc).upper()
    retryable_keywords = [
        "408", "429", "500", "502", "503", "504",
        "UNAVAILABLE", "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED", "SERVICE UNAVAILABLE",
    ]
    return any(kw in error_text for kw in retryable_keywords)


class GeminiGenAIEmbeddings(Embeddings):
    """
    LangChain-compatible embedding wrapper using google-genai SDK.

    Uses gemini-embedding-2 with instruction-style text formatting for
    asymmetric retrieval (different preparation for documents vs queries).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        output_dimensionality: int = 768,
        batch_size: int = 1,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.batch_size = batch_size

    # ── Text preparation ─────────────────────────────────────────────

    def _prepare_document(self, text: str, title: Optional[str] = None) -> str:
        clean_text = text.replace("\n", " ").strip()
        clean_title = title or "portfolio_document"
        return f"title: {clean_title} | text: {clean_text}"

    def _prepare_query(self, query: str) -> str:
        clean_query = query.replace("\n", " ").strip()
        return f"task: search result | query: {clean_query}"

    # ── Core embedding call with retry ───────────────────────────────

    @retry(
        retry=retry_if_exception(is_retryable_genai_error),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_single_batch(self, texts: List[str]) -> List[List[float]]:
        contents = [
            types.Content(parts=[types.Part.from_text(text=text)])
            for text in texts
        ]
        result = self.client.models.embed_content(
            model=self.model,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality
            ),
        )
        return [embedding.values for embedding in result.embeddings]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch = texts[i: i + self.batch_size]
            logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} text(s))")
            all_embeddings.extend(self._embed_single_batch(batch))
        return all_embeddings

    # ── LangChain Embeddings interface ───────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts([self._prepare_document(t) for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([self._prepare_query(text)])[0]