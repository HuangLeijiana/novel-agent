"""Context retrieval for novel writing — builds queries from chapter plans
and retrieves semantically relevant history from the vector store."""

import logging
from typing import Any

from .embedder import Embedder
from .store import NovelVectorStore

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Builds search queries from chapter context and retrieves relevant history."""

    def __init__(self, store: NovelVectorStore, embedder: Embedder | None = None):
        self._store = store
        self._embedder = embedder or Embedder()

    # ================================================================
    # Query construction
    # ================================================================

    def build_writing_query(self, chapter_plan: Any) -> str:
        """Build a retrieval query for writing a new chapter.

        Combines character names, settings, and plot goals into a
        natural-language query optimized for BGE embedding.
        """
        parts = []

        # Characters involved
        if hasattr(chapter_plan, "characters_involved") and chapter_plan.characters_involved:
            parts.append("角色：" + "、".join(chapter_plan.characters_involved[:5]))

        # Scene settings
        if hasattr(chapter_plan, "scenes") and chapter_plan.scenes:
            settings = [s.setting for s in chapter_plan.scenes[:3] if s.setting]
            if settings:
                parts.append("地点：" + "、".join(settings))

        # Plot goal
        if hasattr(chapter_plan, "goal") and chapter_plan.goal:
            parts.append("情节目标：" + chapter_plan.goal[:100])

        # Conflict
        if hasattr(chapter_plan, "conflict") and chapter_plan.conflict:
            parts.append("冲突：" + chapter_plan.conflict[:100])

        if not parts:
            return "小说情节 角色发展 故事推进"

        return "。".join(parts)

    def build_continuity_query(self, draft_content: str, max_chars: int = 2000) -> str:
        """Build a retrieval query for continuity checking.

        Uses the first portion of the chapter to find potentially
        conflicting facts and events.
        """
        excerpt = draft_content[:max_chars]
        # Use the chapter opening as the query — it contains character
        # intros, setting descriptions, and initial actions that are
        # most likely to reference established facts
        return excerpt

    # ================================================================
    # Retrieval
    # ================================================================

    async def retrieve_for_writing(
        self,
        chapter_plan: Any,
        n_results: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve context relevant to writing the planned chapter.

        Returns a dict with keys: chapters, facts, events, foreshadowing.
        Each value is a list of {id, document, metadata, distance} dicts.
        """
        query = self.build_writing_query(chapter_plan)
        logger.debug(f"RAG query: {query[:120]}...")

        try:
            embedding = self._embedder.embed_query(query)
        except Exception as e:
            logger.warning(f"Embedding failed: {e} — skipping RAG retrieval")
            return {"chapters": [], "facts": [], "events": [], "foreshadowing": []}

        return self._store.search_all(embedding, n_results=n_results)

    async def retrieve_for_continuity(
        self,
        draft_content: str,
        n_results: int = 15,
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve context for continuity checking.

        Uses the chapter content itself as the query to find
        facts and events that might contradict.
        """
        query = self.build_continuity_query(draft_content)
        logger.debug(f"Continuity RAG query: {len(query)} chars")

        try:
            embedding = self._embedder.embed_query(query)
        except Exception as e:
            logger.warning(f"Embedding failed: {e} — skipping continuity RAG")
            return {"chapters": [], "facts": [], "events": [], "foreshadowing": []}

        return self._store.search_all(embedding, n_results=n_results)

    def format_context_for_prompt(
        self,
        retrieval_results: dict[str, list[dict[str, Any]]],
        max_items_per_type: int = 8,
    ) -> str:
        """Format retrieval results into a compact context string for prompt injection.

        Args:
            retrieval_results: Output from retrieve_for_writing().
            max_items_per_type: Max items to include per collection type.

        Returns:
            A formatted string ready for prompt injection, or empty string.
        """
        lines: list[str] = []

        # Chapter summaries
        chapters = retrieval_results.get("chapters", [])
        if chapters:
            lines.append("【语义检索：相关历史章节】")
            for r in chapters[:max_items_per_type]:
                ch = r.get("metadata", {}).get("chapter_num", "?")
                doc = r.get("document", "")
                if doc:
                    lines.append(f"第{ch}章 > {doc[:200]}")

        # Facts
        facts = retrieval_results.get("facts", [])
        if facts:
            lines.append("\n【语义检索：相关历史事实】")
            for r in facts[:max_items_per_type]:
                meta = r.get("metadata", {})
                cat = meta.get("category", "")
                doc = r.get("document", "")
                if doc:
                    prefix = f"[{cat}] " if cat else ""
                    lines.append(f"- {prefix}{doc[:150]}")

        # Events
        events = retrieval_results.get("events", [])
        if events:
            lines.append("\n【语义检索：相关历史事件】")
            for r in events[:max_items_per_type]:
                meta = r.get("metadata", {})
                ch = meta.get("chapter_num", "?")
                doc = r.get("document", "")
                if doc:
                    lines.append(f"- 第{ch}章: {doc[:150]}")

        # Foreshadowing
        fs_entries = retrieval_results.get("foreshadowing", [])
        if fs_entries:
            lines.append("\n【语义检索：相关伏笔】")
            for r in fs_entries[:max_items_per_type]:
                meta = r.get("metadata", {})
                status = meta.get("status", "")
                doc = r.get("document", "")
                if doc:
                    status_tag = f" [{status}]" if status else ""
                    lines.append(f"- {doc[:150]}{status_tag}")

        return "\n".join(lines) if lines else ""


class NovelRAG:
    """High-level RAG interface for the novel writing workflow.

    Usage:
        rag = NovelRAG(persist_dir="workspace/projects/my-novel/rag")

        # After each chapter:
        await rag.index_chapter(chapter_num, summary, facts, timeline_events, foreshadowing)

        # Before writing:
        context = await rag.retrieve_for_writing(chapter_plan)
        formatted = rag.retriever.format_context_for_prompt(context)
        # inject formatted into the writer prompt

        # During review:
        cont_context = await rag.retrieve_for_continuity(draft.content)
    """

    def __init__(self, persist_dir: str, model_name: str | None = None):
        self._embedder = Embedder(model_name)
        self._store = NovelVectorStore(persist_dir, embedder=self._embedder)
        self.retriever = ContextRetriever(self._store, embedder=self._embedder)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def store(self) -> NovelVectorStore:
        return self._store

    async def index_chapter(
        self,
        chapter_num: int,
        summary: str,
        facts: list[Any] | None = None,
        timeline_events: list[Any] | None = None,
        foreshadowing_entries: list[Any] | None = None,
    ) -> None:
        """Index a completed chapter into the vector store.

        Args:
            chapter_num: Chapter number.
            summary: Chapter summary text.
            facts: List of Fact objects from the chapter.
            timeline_events: List of TimelineEvent objects.
            foreshadowing_entries: List of ForeshadowingEntry objects.
        """
        # Index chapter summary
        self._store.index_chapter_summary(
            chapter_num=chapter_num,
            summary=summary,
        )

        # Index facts
        if facts:
            fact_tuples = []
            for f in facts:
                text = f"{f.category}: {f.description}" if hasattr(f, "category") else str(f)
                fact_tuples.append((
                    getattr(f, "id", str(hash(text))),
                    text,
                    {
                        "chapter_num": chapter_num,
                        "category": getattr(f, "category", ""),
                        "certainty": getattr(f, "certainty", 1.0),
                    },
                ))
            self._store.index_facts(fact_tuples)

        # Index timeline events
        if timeline_events:
            event_tuples = []
            for e in timeline_events:
                text = f"{e.in_story_time}: {e.description}" if hasattr(e, "in_story_time") else str(e)
                event_tuples.append((
                    getattr(e, "id", str(hash(text))),
                    text,
                    {
                        "chapter_num": chapter_num,
                        "importance": getattr(e, "importance", "minor"),
                        "location": getattr(e, "location", ""),
                    },
                ))
            self._store.index_timeline_events(event_tuples)

        # Index foreshadowing
        if foreshadowing_entries:
            fs_tuples = []
            for f in foreshadowing_entries:
                text = str(f.description) if hasattr(f, "description") else str(f)
                fs_tuples.append((
                    getattr(f, "id", str(hash(text))),
                    text,
                    {
                        "chapter_num": chapter_num,
                        "status": getattr(f, "status", "active"),
                        "planted_chapter": getattr(f, "planted_chapter", chapter_num),
                    },
                ))
            self._store.index_foreshadowing(fs_tuples)

        logger.info(
            f"RAG indexed chapter {chapter_num}: summary + "
            f"{len(facts) if facts else 0} facts + "
            f"{len(timeline_events) if timeline_events else 0} events + "
            f"{len(foreshadowing_entries) if foreshadowing_entries else 0} foreshadowing"
        )

    async def retrieve_for_writing(self, chapter_plan: Any) -> dict:
        """Retrieve context for writing."""
        return await self.retriever.retrieve_for_writing(chapter_plan)

    async def retrieve_for_continuity(self, draft_content: str) -> dict:
        """Retrieve context for continuity checking."""
        return await self.retriever.retrieve_for_continuity(draft_content)

    def get_stats(self) -> dict[str, int]:
        """Get document counts across all collections."""
        return self._store.count_all()
