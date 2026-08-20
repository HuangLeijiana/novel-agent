"""RAG (Retrieval-Augmented Generation) module for novel writing.

Provides semantic search over chapter history using embeddings + ChromaDB.
All imports are lazy — the module works without RAG deps installed, but
search/index operations will raise clear errors.

Usage:
    from src.rag import NovelRAG

    rag = NovelRAG(persist_dir="workspace/projects/my-novel/rag")
    await rag.index_chapter(chapter_num, summary, facts, events)
    context = await rag.retrieve_for_writing(chapter_plan)
"""

from .retriever import ContextRetriever, NovelRAG

__all__ = ["ContextRetriever", "NovelRAG"]
