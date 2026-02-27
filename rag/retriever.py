"""
Hybrid retriever combining semantic search + BM25, with LLM reranking.

Precision strategy:
1. Cast a wide net with both semantic and keyword retrieval
2. Fuse results using Reciprocal Rank Fusion (RRF)
3. Rerank with an LLM for final high-precision top-k
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict

from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import GeneratorConfig, RetrieverConfig
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class BM25Index:
    """Lightweight BM25 keyword index over document chunks."""

    def __init__(self):
        self._documents: list[dict] = []
        self._index: BM25Okapi | None = None

    def build(self, documents: list[dict]) -> None:
        """Build BM25 index from document dicts with 'text' and 'chunk_id'."""
        self._documents = documents
        tokenized = [self._tokenize(doc["text"]) for doc in documents]
        self._index = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(documents)} documents")

    def query(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top-k BM25 results."""
        if self._index is None or not self._documents:
            return []

        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        scored_docs = list(zip(self._documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for doc, score in scored_docs[:top_k]:
            results.append({**doc, "bm25_score": float(score)})

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()


class HybridRetriever:
    """Combines semantic and BM25 retrieval with LLM reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        api_key: str,
        retriever_config: RetrieverConfig | None = None,
        generator_config: GeneratorConfig | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.client = OpenAI(api_key=api_key)
        self.config = retriever_config or RetrieverConfig()
        self.gen_config = generator_config or GeneratorConfig()
        self.bm25_index = BM25Index()
        self._bm25_built = False

    def _ensure_bm25(self) -> None:
        """Lazily build BM25 index from vector store contents."""
        if not self._bm25_built:
            docs = self.vector_store.get_all_documents()
            if docs:
                self.bm25_index.build(docs)
            self._bm25_built = True

    def retrieve(self, query: str, use_rerank: bool = True) -> list[dict]:
        """
        Full hybrid retrieval pipeline:
        1. Semantic search via embeddings
        2. BM25 keyword search
        3. Reciprocal Rank Fusion
        4. LLM reranking (optional)
        """
        self._ensure_bm25()

        # Step 1: Semantic retrieval
        query_embedding = self.embedding_service.embed_query(query)
        semantic_hits = self.vector_store.query(query_embedding, self.config.semantic_top_k)

        # Step 2: BM25 retrieval
        bm25_hits = self.bm25_index.query(query, self.config.bm25_top_k)

        # Step 3: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(semantic_hits, bm25_hits)

        # Step 4: Filter by similarity threshold
        fused = [h for h in fused if h.get("score", 0) >= self.config.similarity_threshold]

        if not fused:
            return []

        # Step 5: Rerank with LLM
        if use_rerank and len(fused) > self.config.rerank_top_k:
            fused = self._llm_rerank(query, fused, self.config.rerank_top_k)
        else:
            fused = fused[: self.config.rerank_top_k]

        return fused

    def _reciprocal_rank_fusion(
        self,
        semantic_hits: list[dict],
        bm25_hits: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Merge two ranked lists using RRF scoring."""
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, dict] = {}

        for rank, hit in enumerate(semantic_hits):
            cid = hit["chunk_id"]
            scores[cid] += self.config.semantic_weight / (k + rank + 1)
            doc_map[cid] = hit

        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            scores[cid] += self.config.bm25_weight / (k + rank + 1)
            if cid not in doc_map:
                doc_map[cid] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for cid, rrf_score in ranked:
            doc = doc_map[cid].copy()
            doc["rrf_score"] = rrf_score
            doc.setdefault("score", 0)
            results.append(doc)

        return results

    def _llm_rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Use an LLM to rerank candidates by relevance to the query."""
        # Prepare numbered passages for the prompt
        passages = []
        for i, hit in enumerate(candidates[: top_k * 3]):  # Send at most 3x top_k
            text_preview = hit["text"][:500]
            source = hit.get("metadata", {}).get("source_file", "unknown")
            section = hit.get("metadata", {}).get("section_hierarchy", "")
            passages.append(f"[{i}] Source: {source} | Section: {section}\n{text_preview}")

        passages_text = "\n\n---\n\n".join(passages)

        prompt = f"""You are a relevance judge. Given a query and a list of text passages, rank the passages by their relevance to the query.

Query: {query}

Passages:
{passages_text}

Return a JSON array of passage indices ordered from most relevant to least relevant.
Only include passages that are actually relevant to answering the query.
Return at most {top_k} indices.

Respond with ONLY a JSON array of integers, e.g. [3, 0, 7, 1]"""

        try:
            response = self.client.chat.completions.create(
                model=self.gen_config.rerank_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()
            # Parse JSON array from response
            match = re.search(r"\[[\d,\s]*\]", content)
            if match:
                indices = json.loads(match.group())
                reranked = []
                for idx in indices[:top_k]:
                    if 0 <= idx < len(candidates):
                        reranked.append(candidates[idx])
                if reranked:
                    return reranked
        except Exception as e:
            logger.warning(f"LLM reranking failed, falling back to RRF order: {e}")

        return candidates[:top_k]
