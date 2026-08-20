#!/usr/bin/env python3
"""novel-agent — 记忆 RAG 检索评估脚本

用真实项目《回到2015当首富》的长期记忆数据评估语义检索质量：
1. 索引 memory/long_term.yaml 的 facts（42 条）与 memory/timeline.yaml 的事件（30 条），
   走项目自身的 NovelVectorStore + Embedder 代码路径（BGE-small-zh 512 维）。
2. 对每条事实/事件，用本地 Ollama qwen3:8b 改写成一个自然中文检索问题，
   模拟"写作 Agent 拿着当前剧情去问记忆库"的查询形态。
3. 检索 top-k，检查来源条目是否命中（recall@k / MRR / 平均排名）。

运行方式：
    python eval/eval_rag_retrieval.py \
        [--bge-model BAAI/bge-small-zh-v1.5] \
        [--memory-dir eval/fixtures/memory] \
        [--ollama-model qwen3:8b] [--top-k 5]
输出：
    eval/rag_retrieval_results.json

前置条件：
- 安装 rag 依赖：uv sync --extra rag（chromadb + sentence-transformers）
- 本地 Ollama 运行 qwen3:8b（用于查询改写；不可用时自动退化为原文查询）
"""

import argparse
import json
import os
import sys
import urllib.request

import yaml

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from rag.embedder import Embedder  # noqa: E402
from rag.store import NovelVectorStore  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "memory")
DEFAULT_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_store")
DEFAULT_OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_retrieval_results.json")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_BGE_MODEL = "BAAI/bge-small-zh-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG memory retrieval (recall@k / MRR).")
    parser.add_argument(
        "--bge-model", default=DEFAULT_BGE_MODEL, help="BGE 模型名或本地路径（默认 BAAI/bge-small-zh-v1.5）"
    )
    parser.add_argument(
        "--memory-dir", default=DEFAULT_MEMORY_DIR, help="记忆数据目录（含 long_term.yaml / timeline.yaml）"
    )
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR, help="ChromaDB 持久化目录")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API 地址")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama 查询改写模型")
    parser.add_argument("--top-k", type=int, default=5, help="检索 top-k")
    parser.add_argument("--output", default=DEFAULT_OUT_PATH, help="结果 JSON 输出路径")
    return parser.parse_args()


def load_docs(memory_dir: str) -> list[tuple[str, str, dict]]:
    """(id, text, metadata) — 42 facts + 30 timeline events，均为真实生成数据。"""
    docs = []
    lt_path = os.path.join(memory_dir, "long_term.yaml")
    tl_path = os.path.join(memory_dir, "timeline.yaml")
    if not os.path.exists(lt_path) or not os.path.exists(tl_path):
        raise FileNotFoundError(f"memory dir 缺少 long_term.yaml / timeline.yaml: {memory_dir}")
    lt = yaml.safe_load(open(lt_path, encoding="utf-8"))
    for fid, item in lt.get("facts", {}).items():
        docs.append(
            (
                fid,
                item["description"],
                {"type": "fact", "category": item.get("category", ""), "source_chapter": item.get("source_chapter", 0)},
            )
        )
    tl = yaml.safe_load(open(tl_path, encoding="utf-8"))
    for ev in tl:
        docs.append(
            (
                ev["id"],
                ev["description"],
                {"type": "event", "importance": ev.get("importance", ""), "chapter": ev.get("chapter", 0)},
            )
        )
    return docs


def ollama_paraphrase(text: str, ollama_url: str, ollama_model: str, max_chars: int = 120) -> tuple[str, bool]:
    """本地 qwen3:8b 把事实改写为自然检索问题；失败时退化为原文。"""
    prompt = (
        "把下面这句话改写成一个自然的中文检索问题（用于从小说记忆库中检索这条信息），"
        f"只输出问题本身，不要解释，不要加引号：\n{text}"
    )
    body = json.dumps(
        {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 128},
        }
    ).encode("utf-8")
    try:
        req = urllib.request.Request(ollama_url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
        q = resp.get("response", "").strip().strip('"').strip("“”")
        return (q[:max_chars] if q else text[:max_chars]), True
    except Exception:
        return text[:max_chars], False


def main() -> None:
    args = parse_args()
    print(f"loading BGE from {args.bge_model} ...")
    embedder = Embedder(model_name=args.bge_model)
    store = NovelVectorStore(persist_dir=args.store_dir, embedder=embedder)

    docs = load_docs(args.memory_dir)
    facts = [d for d in docs if d[2]["type"] == "fact"]
    events = [d for d in docs if d[2]["type"] == "event"]
    print(f"docs: {len(docs)} (facts={len(facts)}, events={len(events)})")

    # 用项目自身的索引接口入库
    store.index_facts([(i, t, m) for i, t, m in facts])
    store.index_timeline_events([(i, t, m) for i, t, m in events])
    print(f"indexed: {store.count_all()}")

    results = {
        "model": args.bge_model,
        "query_paraphrase": f"Ollama {args.ollama_model} 本地改写（失败时退化为原文）",
        "n_facts": len(facts),
        "n_events": len(events),
        "top_k": args.top_k,
        "collections": {},
    }

    for label, group, search_fn in (
        ("facts", facts, store.search_facts),
        ("events", events, store.search_events),
    ):
        ranks = []
        hits_at = {1: 0, 3: 0, 5: 0}
        n_paraphrase_ok = 0
        for idx, (doc_id, text, _meta) in enumerate(group):
            q, ok = ollama_paraphrase(text, args.ollama_url, args.ollama_model)
            n_paraphrase_ok += int(ok)
            q_emb = embedder.embed([q])[0]
            res = search_fn(q_emb, n_results=args.top_k)
            ids = [r["id"] for r in res]
            rank = ids.index(doc_id) + 1 if doc_id in ids else None
            ranks.append(rank)
            if rank:
                for k in hits_at:
                    if rank <= k:
                        hits_at[k] += 1
            if (idx + 1) % 10 == 0:
                print(f"  {label} {idx + 1}/{len(group)} done")
        n = len(group)
        recall = {f"recall@{k}": round(hits_at[k] / n, 4) for k in hits_at}
        hit_ranks = [r for r in ranks if r]
        mrr = round(sum(1 / r for r in hit_ranks) / n, 4)
        results["collections"][label] = {
            **recall,
            "mrr": mrr,
            "mean_rank": round(sum(hit_ranks) / len(hit_ranks), 2) if hit_ranks else None,
            "missed": n - len(hit_ranks),
            "paraphrase_ok": f"{n_paraphrase_ok}/{n}",
        }
        print(f"{label}: {recall} mrr={mrr} missed={n - len(hit_ranks)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.output}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
