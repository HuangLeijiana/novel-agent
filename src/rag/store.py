"""ChromaDB vector store for novel memory indexing and retrieval."""

import logging
from typing import Any

from .embedder import Embedder

logger = logging.getLogger(__name__)

# Collection names
COLLECTION_CHAPTERS = "chapter_summaries"
COLLECTION_FACTS = "facts"
COLLECTION_EVENTS = "timeline_events"
COLLECTION_FORESHADOWING = "foreshadowing"


class NovelVectorStore:
    """ChromaDB-backed vector store for novel context.

    Each project gets its own ChromaDB collection set under persist_dir.
    Four collections track different types of memory:
    - chapter_summaries: compressed chapter content
    - facts: established world/character/plot facts
    - timeline_events: major story events with causality
    - foreshadowing: planted/active/paid-off foreshadowing
    """

    def __init__(self, persist_dir: str, embedder: Embedder | None = None):
        self._persist_dir = persist_dir
        self._embedder = embedder or Embedder()
        self._client = None

    @property
    def client(self):
        """Lazy-init ChromaDB persistent client."""
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError(
                    "chromadb is required for RAG. Install with: pip install novel-agent[rag]"
                )
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    def _get_or_create(self, name: str) -> Any:
        """Get or create a ChromaDB collection with the configured embedding function."""
        try:
            # ChromaDB 0.5.x API
            return self.client.get_collection(name=name)
        except Exception:
            pass

        # Create with our custom embedding function
        from chromadb import EmbeddingFunction, Embeddings

        class _NovelEmbeddingFunction(EmbeddingFunction):
            def __init__(self, embedder_ref):
                self._embedder_ref = embedder_ref

            def __call__(self, texts: list[str]) -> Embeddings:
                return self._embedder_ref.embed(texts)

        ef = _NovelEmbeddingFunction(self._embedder)
        return self.client.create_collection(name=name, embedding_function=ef)

    # ============================================================
    # Indexing
    # ============================================================

    def index_chapter_summary(
        self,
        chapter_num: int,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index a chapter summary for semantic retrieval."""
        if not summary.strip():
            return
        meta = metadata or {}
        meta["chapter_num"] = chapter_num
        doc_id = f"ch_{chapter_num}"
        coll = self._get_or_create(COLLECTION_CHAPTERS)
        # Upsert: overwrite if this chapter was already indexed
        coll.upsert(
            ids=[doc_id],
            documents=[summary],
            metadatas=[meta],
        )
        logger.debug(f"Indexed chapter {chapter_num} summary")

    def index_facts(
        self,
        facts: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        """Index multiple facts. Each tuple is (fact_id, fact_text, metadata)."""
        if not facts:
            return
        coll = self._get_or_create(COLLECTION_FACTS)
        ids = [f[0] for f in facts]
        texts = [f[1] for f in facts]
        metas = [f[2] for f in facts]
        coll.upsert(ids=ids, documents=texts, metadatas=metas)
        logger.debug(f"Indexed {len(facts)} facts")

    def index_timeline_events(
        self,
        events: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        """Index multiple timeline events."""
        if not events:
            return
        coll = self._get_or_create(COLLECTION_EVENTS)
        ids = [e[0] for e in events]
        texts = [e[1] for e in events]
        metas = [e[2] for e in events]
        coll.upsert(ids=ids, documents=texts, metadatas=metas)
        logger.debug(f"Indexed {len(events)} timeline events")

    def index_foreshadowing(
        self,
        entries: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        """Index multiple foreshadowing entries."""
        if not entries:
            return
        coll = self._get_or_create(COLLECTION_FORESHADOWING)
        ids = [e[0] for e in entries]
        texts = [e[1] for e in entries]
        metas = [e[2] for e in entries]
        coll.upsert(ids=ids, documents=texts, metadatas=metas)
        logger.debug(f"Indexed {len(entries)} foreshadowing entries")

    # ============================================================
    # Retrieval
    # ============================================================

    def search_chapters(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search chapter summaries by embedding."""
        coll = self._get_or_create(COLLECTION_CHAPTERS)
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, coll.count()),
        )
        return self._format_results(results)

    def search_facts(
        self,
        query_embedding: list[float],
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search facts by embedding."""
        coll = self._get_or_create(COLLECTION_FACTS)
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, coll.count()),
        )
        return self._format_results(results)

    def search_events(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search timeline events by embedding."""
        coll = self._get_or_create(COLLECTION_EVENTS)
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, coll.count()),
        )
        return self._format_results(results)

    def search_foreshadowing(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search foreshadowing entries by embedding."""
        coll = self._get_or_create(COLLECTION_FORESHADOWING)
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, coll.count()),
        )
        return self._format_results(results)

    def search_all(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search all collections and return results keyed by collection."""
        return {
            "chapters": self.search_chapters(query_embedding, n_results),
            "facts": self.search_facts(query_embedding, n_results),
            "events": self.search_events(query_embedding, n_results),
            "foreshadowing": self.search_foreshadowing(query_embedding, n_results),
        }

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _format_results(results: dict) -> list[dict[str, Any]]:
        """Format ChromaDB query results into a clean list."""
        formatted = []
        ids_list = results.get("ids", [[]])[0] if results.get("ids") else []
        docs_list = results.get("documents", [[]])[0] if results.get("documents") else []
        metas_list = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        for i in range(len(ids_list)):
            formatted.append({
                "id": ids_list[i] if i < len(ids_list) else "",
                "document": docs_list[i] if i < len(docs_list) else "",
                "metadata": metas_list[i] if i < len(metas_list) else {},
                "distance": distances[i] if i < len(distances) else 1.0,
            })
        return formatted

    def count_all(self) -> dict[str, int]:
        """Return document counts for all collections."""
        counts = {}
        for name in [COLLECTION_CHAPTERS, COLLECTION_FACTS, COLLECTION_EVENTS, COLLECTION_FORESHADOWING]:
            try:
                coll = self.client.get_collection(name=name)
                counts[name] = coll.count()
            except Exception:
                counts[name] = 0
        return counts
