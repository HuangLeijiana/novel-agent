# 长篇上下文一致性方案

## 问题定义

当前 novel-agent 在长篇（>50章）写作中，每章生成只能看到：
- 完整世界观设定（静态，几百章不变）
- 完整角色档案（静态 `current_state`，创建后从不更新）
- 上一章 200 字摘要
- 活跃伏笔 ID 列表（无描述）
- 未解决钩子

**看不到的**：第3章发生了什么、角色50章来的演变轨迹、早期埋下的伏笔内容、已确立的世界事实。

简单说：Writer 写第200章时，和第5章时的「知情程度」几乎一样——它只知道「刚发生的上一章」和「创世之初的设定」。

---

## 方案对比

### 方案 A：层次化摘要 (Hierarchical Summarization)

**核心思路**：每 N 章把内容压缩一层，形成树状结构。写作时把各层最新摘要注入上下文。

```
Chapter 1-10  → Stage Summary 1  ┐
Chapter 11-20 → Stage Summary 2  ├─→ Arc Summary I
Chapter 21-30 → Stage Summary 3  │
Chapter 31-40 → Stage Summary 4  ┘
...
Chapter 91-100 → Stage Summary 10 → Global Summary
```

**注入到每章写作的上下文**：
```
[上一章摘要] 200字
[当前阶段摘要] ~500字（最近10章的压缩）
[当前弧摘要] ~800字（最近50章的压缩）
[全局摘要] ~1000字（全书脉络，每100章更新）
```

**优点**：
- 零新依赖，纯 Python 实现
- LLM 天然擅长摘要（已有 MemoryManagerAgent 的基础）
- 延迟零增加（无额外查询）
- 总上下文预算可控（~2500 字摘要覆盖全书）

**缺点**：
- 信息有损——摘要可能遗漏后来变得重要的细节
- 摘要质量依赖 LLM 能力
- 无法精确回答「第37章那个路人甲后来怎么样了」

**实现量**：~200 行核心逻辑 + model 字段扩展

---

### 方案 B：RAG 检索增强 (Retrieval-Augmented Generation)

**核心思路**：所有章节摘要/事实/事件 → embedding → 向量库；写第N章时，用「本章涉及的角色、地点、情节线」做查询，检索最相关的前K条上下文注入。

```
写作流程：
1. 提取本章关键词（角色名、地点、情节主题）
2. query = "角色A 关系 角色B 重要事件"
3. 向量检索 top-20 最相关记忆
4. 注入：上一章摘要 + 检索到的相关记忆
```

**优点**：
- 精确检索——只注入和本章相关的内容
- 无信息衰减——原始事实存在向量库中
- 天然适应超长篇（1000+章也能检索）
- 可以支持「跨卷检索」：写第500章时检索第50章埋的伏笔

**缺点**：
- **新依赖**：embedding 模型（本地 sentence-transformers 或 API）+ 向量存储（ChromaDB/Qdrant/lance）
- **额外成本**：embedding API 调用费（如果用云端模型），或者本地显存开销（~500MB for all-MiniLM-L6-v2）
- **检索质量依赖查询构造**：如果 query 没写好，检索结果没用
- **无法替代全局感知**：检索只能给你「点」，不能给你「面」——写作还需要知道当前在哪个弧、整体节奏

**实现量**：~500 行 + 新依赖安装

---

### 方案 C：混合方案（推荐）

**层次化摘要提供全局上下文 + RAG 提供精确检索**。两者互补：

| 需求 | 用什么 | 为什么 |
|------|--------|--------|
| 知道当前在哪个故事弧 | 层次化摘要 | 结构信息，压缩不丢失 |
| 知道角色当前状态 | Memory.character_states（已有）| 结构化数据，不需要检索 |
| 查「某个角色的某个细节」 | RAG 检索 | 精确匹配，摘要可能漏掉 |
| 查「某个伏笔的状态」 | ForeshadowingTracker（已有）| 结构化追踪，精确 |
| 知道整体写作风格和规则 | Bible（已有，静态） | 不变，每次都注入 |
| 跨卷事实一致性校验 | RAG 检索 + 矛盾检测 | 两个来源比对 |

**写作时注入上下文的完整构成**：
```
1. [静态] 世界观 + 风格约定                        ~1000 tokens
2. [结构化] 本章涉及角色的档案 + 当前状态            ~500 tokens
3. [层次化摘要] 上一章 + 阶段 + 弧 + 全局摘要        ~1500 tokens
4. [RAG检索] 与本章相关的历史事实 top-10              ~500 tokens
5. [结构化] 活跃伏笔详情                              ~300 tokens
6. [章节规划] 场景+情绪曲线                           ~500 tokens
─────────────────────────────────────────────────────────────
总计                                                ~4300 tokens
```

远低于 8192/16384 的上下文窗口，有充足余量写 3000+ 字正文。

---

## 推荐实施路径

分两个阶段，先做层次化摘要（快速见效），后加 RAG（锦上添花）。

### Phase 1：层次化摘要（1-2 天）

改动范围：

1. **`models/memory.py`** — 新增字段：
```python
class LongTermMemory(BaseModel):
    # ... 现有字段保持不变 ...
    stage_summaries: dict[int, str] = {}   # stage_num -> 500字阶段摘要
    arc_summaries: dict[int, str] = {}     # arc_num -> 800字弧摘要
    global_summary: str = ""               # 全书脉络，每100章更新

class MemoryState(BaseModel):
    # 新增
    stage_boundaries: list[int] = []       # 阶段边界章号，如 [10, 20, 30, ...]
    arc_boundaries: list[int] = []         # 弧边界章号，如 [50, 100, 150, ...]
```

2. **`agents/memory_manager.py`** — 新增 `consolidate_periodically()`：
```python
async def consolidate_periodically(self, memory: MemoryState, chapter_num: int):
    """每10章生成阶段摘要，每50章生成弧摘要，每100章更新全局摘要"""
    if chapter_num % 10 == 0:
        stage_num = chapter_num // 10
        # 取该阶段所有章节摘要 + 时间线事件 → LLM 压缩
        stage_summary = await self._generate_stage_summary(...)
        memory.long_term.stage_summaries[stage_num] = stage_summary
    
    if chapter_num % 50 == 0:
        # 取该弧所有阶段摘要 → LLM 压缩
        arc_summary = await self._generate_arc_summary(...)
        memory.long_term.arc_summaries[arc_num] = arc_summary
    
    if chapter_num % 100 == 0:
        # 取全部弧摘要 → LLM 压缩
        memory.long_term.global_summary = await self._generate_global_summary(...)
```

3. **`agents/base.py`** — `build_context_block` 改用层次化摘要：
```python
if memory:
    # 替换原来只取 short_term 的逻辑
    lt = memory.get("long_term", {})
    parts.append("=== 近期记忆 ===")
    parts.append(f"上一章: {st.get('current_chapter_summary', '')}")
    
    # 层次化摘要
    stage = lt.get("stage_summaries", {})
    if stage:
        latest = stage[max(stage.keys())]
        parts.append(f"当前阶段（最近10章）: {latest}")
    
    arc = lt.get("arc_summaries", {})
    if arc:
        latest = arc[max(arc.keys())]
        parts.append(f"当前弧（最近50章）: {latest}")
    
    if lt.get("global_summary"):
        parts.append(f"全书脉络: {lt['global_summary']}")
```

4. **`agents/writer.py`** — `_build_full_context` 增加相关事实注入：
```python
# 从 long_term.facts 中筛选本章相关事实（基于角色/地点关键词匹配）
relevant = self._filter_relevant_facts(
    chapter_plan=chapter_plan,
    facts=memory.long_term.facts,
    max_facts=20,
)
```

5. **`api/routes.py`** — `start_workflow` 恢复记忆：
```python
# 现有代码加载 bible/characters/outline 但跳过 memory
memory = await file_manager.load_memory()
if memory:
    state.memory = memory
```

6. **`models/state.py`** — 修复角色状态分裂：
```python
# advance_chapter() 中增加：将 memory 中的角色状态写回 character registry
for cid, state_updates in self.memory.character_states.items():
    if cid in self.characters.characters:
        self.characters.characters[cid].current_state.update(state_updates)
```

### Phase 2：RAG 检索（后续，可并行开发）

1. **向量存储**：选择 `ChromaDB`（轻量，零配置，Python 原生，SQLite 后端）
2. **Embedding 模型**：`BAAI/bge-small-zh-v1.5`（中文优化，512维，~100MB，CPU 友好）
3. **索引内容**：每章的 (章节摘要, 新增事实, 时间线事件, 伏笔描述)
4. **检索触发点**：
   - `WriterAgent._build_full_context`：检索本章相关记忆
   - `ContinuityCheckerAgent`：检索可能矛盾的历史事实
   - `PlotPlannerAgent.plan_chapter`：检索相关伏笔和未解决线索

**新增模块**：
```
src/
├── rag/
│   ├── __init__.py
│   ├── embedder.py      # embedding 模型封装
│   ├── store.py          # ChromaDB CRUD
│   └── retriever.py      # 查询构造 + 检索逻辑
```

---

## 成本分析

| 方案 | 新增依赖 | 每章额外 LLM 调用 | 每章额外成本（估算） |
|------|---------|-------------------|---------------------|
| A: 层次化摘要 | 无 | 每10/50/100章 +1次摘要调用 | ~0.001$/章（摊销） |
| B: 纯 RAG | ChromaDB + embedding模型 | 每章 +1次 embedding（~10条） | ~0.0001$/章（本地embedding） |
| C: 混合 | ChromaDB + embedding模型 | A的摘要调用 + 每章embedding | ~0.001$/章 |

本地 embedding 模型（bge-small-zh）在 CPU 上每条约 5-10ms，10条 < 100ms，几乎不影响延迟。

---

## 结论

**推荐立即实施 Phase 1（层次化摘要）**，这是当前最严重的缺口——系统已经存了所有数据（`chapter_summaries`、`facts`、`timeline`），只是没有压缩和注入机制。改动量小，效果立竿见影。

**Phase 2（RAG）** 作为下一步，当层次化摘要本身不够时（比如需要精确检索某个冷门细节），增加 RAG 层。两层互补：摘要给「面」，RAG 给「点」。
