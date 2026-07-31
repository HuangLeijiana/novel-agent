"""Topic Scout Agent — market research and topic selection for commercial web novels.

Covers Steps 1-4 of the commercial writing pipeline:
  1A/1B: Platform scanning (飞卢 + 番茄 real rankings)
  1C: Cross-platform analysis
  2: Benchmark book skeleton analysis
  3A/3B: 12 candidate topics → scoring → top 4 → manual select 2
  4A: Title & synopsis generation
"""

import logging

from pydantic import BaseModel, Field, field_validator

from ..models.topic import (
    BenchmarkReport,
    BenchmarkSkeleton,
    CandidateTopic,
    CandidateTopicsOutput,
    CrossPlatformEntry,
    CrossPlatformReport,
    MiniArcChapter,
    MiniArcOutline,
    ScanEntry,
    ScanReport,
    TitleSynopsisPair,
    TitleSynopsisReport,
    TopicScoreCard,
    TopicScoringReport,
)
from .base import BaseAgent

logger = logging.getLogger(__name__)


# ============================================================
# Structured output schemas for LLM responses
# ============================================================


class ScanReportOutput(BaseModel):
    """LLM output for platform scan analysis."""

    platform: str = Field(default="", description="平台名称")
    list_name: str = Field(default="", description="榜单名称")
    date: str = Field(default="", description="扫描日期")
    entries: list[ScanEntry] = Field(default_factory=list)
    summary: str = Field(default="", description="趋势总结")
    scan_failed: bool = Field(default=False)

    @field_validator("entries", mode="before")
    @classmethod
    def coerce_string_entries(cls, v: list) -> list:
        """Convert plain-string entries to ScanEntry objects on the fly."""
        if not v:
            return v
        result = []
        for i, item in enumerate(v):
            if isinstance(item, str):
                # LLM returned a bare book title → wrap as minimal ScanEntry
                result.append(ScanEntry(rank=i + 1, title=item))
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append(item)
        return result


class CrossPlatformOutput(BaseModel):
    """LLM output for cross-platform comparison."""

    entries: list[CrossPlatformEntry] = Field(default_factory=list)
    selected_directions: list[str] = Field(default_factory=list)


class BenchmarkOutput(BaseModel):
    """LLM output for benchmark skeleton analysis."""

    entries: list[BenchmarkSkeleton] = Field(default_factory=list)
    structural_summary: str = Field(default="")


class CandidateTopicsListOutput(BaseModel):
    """LLM output for 12 candidate topics."""

    topics: list[CandidateTopic] = Field(default_factory=list)


class TopicScoresOutput(BaseModel):
    """LLM output for topic scoring."""

    scores: list[TopicScoreCard] = Field(default_factory=list)
    top_4: list[str] = Field(default_factory=list)


class TitleSynopsisListOutput(BaseModel):
    """LLM output for title/synopsis generation."""

    genre_name: str = Field(default="")
    title_candidates: list[TitleSynopsisPair] = Field(default_factory=list)
    final_title: str = Field(default="")
    final_synopsis: str = Field(default="")


class MiniArcOutlineOutput(BaseModel):
    """LLM output for 10-chapter mini-arc outline."""

    genre_name: str = Field(default="")
    total_words: str = Field(default="20,000-21,000")
    chapters: list[MiniArcChapter] = Field(default_factory=list)
    arc_goal: str = Field(default="")
    next_arc_hook: str = Field(default="")


# ============================================================
# Agent
# ============================================================


class TopicScoutAgent(BaseAgent):
    """Market research and topic selection for commercial web novels.

    Implements the complete upstream pipeline: real platform scanning →
    cross-platform comparison → benchmark skeleton analysis →
    candidate topic generation → scoring → title/synopsis selection.

    Usage:
        agent = TopicScoutAgent(scheduler)
        # Step 1A: scan 飞卢
        feilu = await agent.scan_feilu(page_content=feilu_html)
        # Step 1B: scan 番茄
        fanqie = await agent.scan_fanqie(page_content=fanqie_html)
        # ... etc
    """

    agent_type = "topic_scout"

    # ================================================================
    # Step 1A: 飞卢真实扫榜
    # ================================================================

    async def scan_platform(self, platform: str, page_content: str | None = None) -> ScanReport:
        """Scan a real platform rankings page and extract structured data.

        Tries structured generation first, falls back to plain-text generation
        + regex parsing on failure (needed for ModelScope which lacks JSON mode).

        Args:
            platform: '飞卢' or '番茄'
            page_content: Raw HTML or text content from the platform page.
                          If None/empty, the scan fails gracefully.

        Returns:
            ScanReport with entries and summary, or scan_failed=True.
        """
        if not page_content or not page_content.strip():
            logger.warning(f"{platform} scan: no page content provided → scan failed")
            return ScanReport(
                platform=platform,
                scan_failed=True,
                summary="本次扫榜失败：未提供页面内容",
            )

        system_prompt = self.build_system_prompt(
            role="网文扫榜编辑" if platform == "飞卢" else "网文平台分析编辑",
            expertise=(
                f"你熟悉{platform}小说平台的榜单结构、题材风向和读者口味。"
                f"你能从真实榜单数据中精确提取每本书的排名、书名、题材、爽点和核心设定。"
                f"你的分析必须基于提供的真实页面数据，不允许泛泛而谈或编造信息。"
            ),
        )

        user_prompt = f"""你现在是{platform}平台的专业分析编辑。请基于下方提供的{platform}真实榜单页面内容，完成逐本书的分析。

【平台页面内容】
{page_content[:15000]}

⚠️ 重要：必须逐本提取！不要只写总结！entries 数组里每本书必须是一个独立对象！

步骤如下：

1. 从页面中逐本提取真实书名，至少提取10本。找不到10本就如实说明原因。

2. 每本书在 entries 数组中作为一个独立对象。页面中可直接提取的字段（书名、排名）必须真实提取，不能编造。页面中不直接显示的字段（如题材标签、金手指、爽点模式等），请根据以下规则**合理推断**：
   - 根据书名中的关键词推断题材标签（如"战神"→都市、"修仙"→仙侠、"系统"→游戏异界）
   - 根据书名结构推断书名吸睛点（如身份反差、钩子悬念、金手指展示等）
   - 根据题材惯例推断金手指类型和爽点循环模式
   - 推断结果标注在对应字段中即可，无需特别说明

3. 每本书必须包含以下字段：
   - rank: 排名（数字，必须从页面提取）
   - title: 书名（必须从页面真实提取）
   - genre: 题材标签（根据书名推断）
   - title_appeal: 书名为什么吸睛（根据书名风格推断）
   - one_liner: 一句话卖点（根据书名+题材惯例推断）
   - golden_finger: 金手指/核心设定（根据题材惯例推断）
   - opening_pressure: 开篇压力设计（根据题材惯例推断）
   - pleasure_loop: 爽点循环模式（根据题材惯例推断）

4. 在 summary 字段中总结题材趋势。

5. 如果页面内容确实连书名都无法提取，将 scan_failed 设为 true。
"""

        # Attempt 1: structured generation
        try:
            result = await self.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ScanReportOutput,
                temperature_override=0.3,
            )
            if result.entries and len(result.entries) >= 3:
                return ScanReport(
                    platform=result.platform or platform,
                    list_name=result.list_name,
                    date=result.date,
                    entries=[ScanEntry(**e.model_dump()) for e in result.entries],
                    summary=result.summary,
                    scan_failed=False,
                )
            logger.warning(
                f"{platform} structured scan returned only {len(result.entries or [])} entries → falling back to text"
            )
        except Exception as e:
            logger.warning(f"{platform} structured scan failed: {e} → falling back to text")

        # Attempt 2: plain-text generation + regex parsing
        logger.info(f"{platform}: falling back to text-based scan")
        try:
            text_prompt = (
                user_prompt
                + """

请用以下纯文本格式逐本输出（每本书一行）：
#排名 | 书名 | 题材标签 | 书名吸睛点 | 一句话卖点 | 金手指 | 开篇压力 | 爽点模式

然后在最后用"=== 趋势总结 ==="开头，写2-4句趋势总结。
"""
            )
            text_result = await self.generate(
                system_prompt=system_prompt,
                user_prompt=text_prompt,
                temperature_override=0.3,
            )
            text = text_result.content if hasattr(text_result, "content") else str(text_result)
            entries = self._parse_text_scan_entries(text)

            if entries:
                import re

                summary_match = re.search(r"===?\s*趋势总结\s*===?\s*\n(.+)", text, re.DOTALL)
                summary = summary_match.group(1).strip()[:200] if summary_match else f"{platform}榜单分析完成"
                logger.info(f"{platform} text fallback: parsed {len(entries)} entries")
                return ScanReport(
                    platform=platform,
                    list_name=f"{platform}榜单",
                    date="",
                    entries=entries,
                    summary=summary,
                    scan_failed=False,
                )
        except Exception as e2:
            logger.error(f"{platform} text fallback also failed: {e2}")

        return ScanReport(platform=platform, scan_failed=True, summary="本次扫榜失败：无法从页面提取书名")

    @staticmethod
    def _parse_text_scan_entries(text: str) -> list[ScanEntry]:
        """Parse pipe-delimited scan entries from plain-text LLM output.

        Expected format: #N | title | genre | title_appeal | one_liner | golden_finger | opening_pressure | pleasure_loop
        """

        entries = []
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            try:
                rank = int(parts[0].lstrip("#").strip())
            except ValueError:
                continue

            entries.append(
                ScanEntry(
                    rank=rank,
                    title=parts[1] if len(parts) > 1 else "",
                    genre=parts[2] if len(parts) > 2 else "",
                    title_appeal=parts[3] if len(parts) > 3 else "",
                    one_liner=parts[4] if len(parts) > 4 else "",
                    golden_finger=parts[5] if len(parts) > 5 else "",
                    opening_pressure=parts[6] if len(parts) > 6 else "",
                    pleasure_loop=parts[7] if len(parts) > 7 else "",
                )
            )
        return entries

    async def scan_feilu(self, page_content: str | None = None) -> ScanReport:
        """Step 1A: Scan 飞卢小说 real rankings."""
        return await self.scan_platform("飞卢", page_content)

    async def scan_fanqie(self, page_content: str | None = None) -> ScanReport:
        """Step 1B: Scan 番茄小说 real rankings.

        For Fanqie, the scraper already produces clean structured text.
        We use the direct parser to extract clean book entries (no LLM needed
        for extraction), then enrich with LLM-based genre/market analysis.
        """
        if not page_content or not page_content.strip():
            logger.warning("Fanqie scan: no page content provided → scan failed")
            return ScanReport(
                platform="番茄",
                scan_failed=True,
                summary="本次扫榜失败：未提供页面内容",
            )

        # Step 1: Direct parse to get clean book titles + abstracts
        direct_entries = self._parse_fanqie_scraper_text(page_content)
        if direct_entries:
            logger.info(f"Fanqie direct parse: {len(direct_entries)} books")

            # Step 2: Enrich with LLM analysis (genre, golden_finger, trend summary)
            enriched, trend_summary = await self._enrich_fanqie_with_llm(page_content, direct_entries)
            if enriched:
                summary = trend_summary or f"番茄榜单：共 {len(enriched)} 本上榜书籍，已完成题材分析"
                return ScanReport(
                    platform="番茄",
                    list_name="综合榜单（新书榜+阅读榜，男频+女频）",
                    date="",
                    entries=enriched,
                    summary=summary,
                    scan_failed=False,
                )

            # LLM enrichment failed — return direct-parsed entries
            return ScanReport(
                platform="番茄",
                list_name="综合榜单（新书榜+阅读榜，男频+女频）",
                date="",
                entries=direct_entries,
                summary=f"番茄榜单：共 {len(direct_entries)} 本上榜书籍（书名+简介，题材分析未完成）",
                scan_failed=False,
            )

        # Direct parsing failed — fall back to full LLM scan (like Feilu)
        logger.info("Fanqie direct parse failed, falling back to LLM scan")
        return await self.scan_platform("番茄", page_content)

    async def _enrich_fanqie_with_llm(
        self, page_content: str, direct_entries: list[ScanEntry]
    ) -> tuple[list[ScanEntry] | None, str]:
        """Use LLM to enrich Fanqie entries with genre analysis and trend summary.

        Uses plain-text generation (not structured output) for reliability
        with all API providers. Parses the result with regex.
        """
        system_prompt = self.build_system_prompt(
            role="网文平台分析编辑",
            expertise=(
                "你熟悉番茄小说平台的榜单结构、题材风向和读者口味。"
                "你能从榜单数据中识别每本书的题材标签、书名吸睛点、金手指类型、"
                "开篇压力设计和爽点循环模式，并总结出榜单的整体趋势。"
            ),
        )

        book_list = "\n".join(
            f"#{e.rank} 《{e.title}》" + (f" — {e.one_liner[:80]}" if e.one_liner else "") for e in direct_entries[:15]
        )

        user_prompt = f"""请分析以下番茄小说榜单，为每本书标注题材和核心设定，并总结趋势。

【榜单数据】
{book_list}

请按以下格式输出：

=== 趋势总结 ===
（2-4句话总结榜单整体趋势）

=== 逐本分析 ===
#排名 | 题材 | 书名吸睛点 | 金手指类型 | 开篇压力 | 爽点模式
#1 | 都市 | 反差身份 | 战神归来 | 被退婚羞辱 | 打脸逆袭
#2 | 古言 | ... | ... | ... | ...
（每本书一行，保持原排名和书名）

请确保每本书都有一行分析。"""

        try:
            result = await self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature_override=0.4,
            )
            text = result.content if hasattr(result, "content") else str(result)

            # Parse trend summary
            import re

            summary_match = re.search(r"===?\s*趋势总结\s*===?\s*\n(.+?)(?:\n===|\n\n===|\Z)", text, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else ""

            # Parse per-book analysis
            enriched = []
            for line in text.split("\n"):
                line = line.strip()
                if not line.startswith("#"):
                    continue
                # Format: #N | genre | title_appeal | golden_finger | opening_pressure | pleasure_loop
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 3:
                    continue

                try:
                    rank = int(parts[0].lstrip("#").strip())
                except ValueError:
                    continue

                # Find matching direct entry
                direct = next((d for d in direct_entries if d.rank == rank), None)
                title = direct.title if direct else parts[0]

                enriched.append(
                    ScanEntry(
                        rank=rank,
                        title=title,
                        genre=parts[1] if len(parts) > 1 else "",
                        title_appeal=parts[2] if len(parts) > 2 else "",
                        golden_finger=parts[3] if len(parts) > 3 else "",
                        opening_pressure=parts[4] if len(parts) > 4 else "",
                        pleasure_loop=parts[5] if len(parts) > 5 else "",
                        one_liner=direct.one_liner if direct else "",
                    )
                )

            if enriched:
                return enriched, summary
        except Exception as e:
            logger.warning(f"Fanqie LLM enrichment failed: {e}")

        return None, ""

    @staticmethod
    def _parse_fanqie_scraper_text(text: str) -> list[ScanEntry]:
        """Parse Fanqie scraper output directly into ScanEntry list.

        The scraper produces formatted text like:
            ===== 番茄小说榜单数据（共X本，涵盖...）=====

            #1 《书名》 作者：作者名
                简介：abstract text
            #2 《书名》 作者：作者名
                简介：abstract text

        Multi-line abstracts are handled (new entries start with #N pattern).
        """
        import re

        entries = []
        # Split into entry blocks: each starts with #N pattern
        # Remove the header line
        lines = text.strip().split("\n")

        # Collect entry blocks
        current_block = []
        for line in lines:
            # Strip whitespace but preserve the content
            stripped = line.strip()
            # Detect start of a new entry: #N 《 or #N（
            if re.match(r"^#\d+\s*[《（]", stripped):
                if current_block:
                    entry = TopicScoutAgent._parse_one_fanqie_block(current_block)
                    if entry:
                        entries.append(entry)
                current_block = [stripped]
            elif current_block:  # continuation of current entry (e.g. multi-line abstract)
                current_block.append(stripped)

        # Don't forget the last block
        if current_block:
            entry = TopicScoutAgent._parse_one_fanqie_block(current_block)
            if entry:
                entries.append(entry)

        return entries

    @staticmethod
    def _parse_one_fanqie_block(lines: list[str]) -> ScanEntry | None:
        """Parse a single Fanqie book entry block into a ScanEntry."""
        import re

        if not lines:
            return None

        first_line = lines[0]
        # Extract rank: #1, #2, etc.
        rank_match = re.match(r"^#(\d+)", first_line)
        rank = int(rank_match.group(1)) if rank_match else 0

        # Extract title from 《...》
        title_match = re.search(r"《(.+?)》", first_line)
        title = title_match.group(1).strip() if title_match else ""

        # Extract author from 作者：...
        author_match = re.search(r"作者[：:]\s*(.+?)$", first_line)
        author = author_match.group(1).strip() if author_match else ""

        # Extract abstract from subsequent lines
        abstract = ""
        for line in lines[1:]:
            # Remove 简介：prefix
            clean = re.sub(r"^简介[：:]\s*", "", line)
            if clean:
                abstract += clean

        if not title:
            return None

        return ScanEntry(
            rank=rank,
            title=title,
            genre="",
            title_appeal="",
            one_liner=abstract[:200] if abstract else f"作者：{author}" if author else "",
            golden_finger="",
            opening_pressure="",
            pleasure_loop="",
        )

    # ================================================================
    # Step 1C: 双榜交叉分析
    # ================================================================

    async def cross_platform_analysis(self, feilu: ScanReport | None, fanqie: ScanReport | None) -> CrossPlatformReport:
        """Step 1C: Cross-platform comparison — narrow down topic directions.

        Works with whatever platform data is available. If only one platform
        has data, analyzes that platform alone and produces directions.
        Falls back to extracting genre keywords from scan entries when the
        LLM produces no directions.
        """
        feilu_ok = feilu is not None and not feilu.scan_failed and len(feilu.entries) > 0
        fanqie_ok = fanqie is not None and not fanqie.scan_failed and len(fanqie.entries) > 0

        # If no platform data at all, return empty
        if not feilu_ok and not fanqie_ok:
            logger.warning("Cross-platform analysis: no valid scan data from either platform")
            return CrossPlatformReport()

        feilu_text = self._format_scan_for_prompt(feilu) if feilu else "[无飞卢数据]"
        fanqie_text = self._format_scan_for_prompt(fanqie) if fanqie else "[无番茄数据]"

        # Adjust role and prompt based on data availability
        if feilu_ok and fanqie_ok:
            role = "网文双平台策略分析师"
            expertise = (
                "你精通飞卢和番茄两个平台的内容生态差异。"
                "你能从榜单数据中识别出跨平台的热门题材共性，也能发现每个平台的独特口味。"
                "你的分析基于真实数据，不凭空猜测。"
            )
            prompt_intro = '请同时参考以下两份真实扫榜数据，做"双榜交叉结论"。'
        else:
            role = "网文平台策略分析师"
            platform_name = "飞卢" if feilu_ok else "番茄"
            expertise = (
                f"你熟悉{platform_name}平台的内容生态和题材风向。"
                f"你能从榜单数据中识别热门题材并推荐适合继续深挖的方向。"
                f"你的分析基于真实数据，不凭空猜测。"
            )
            prompt_intro = f"请基于以下{platform_name}真实扫榜数据，分析热门题材并推荐创作方向。"

        system_prompt = self.build_system_prompt(role=role, expertise=expertise)

        user_prompt = f"""{prompt_intro}

=== A. 飞卢扫榜数据 ===
{feilu_text}

=== B. 番茄扫榜数据 ===
{fanqie_text}

请输出表格（每个题材一行）：

| 题材名 | 飞卢是否有明显热度 | 番茄是否有对应读者基础 | 飞卢吸人的点 | 到番茄需要调整什么 | 风险点 | 推荐等级 |

最后保留3-5个适合继续做对标拆解的方向（放在 selected_directions 字段中）。"""

        # Attempt 1: structured generation
        try:
            result = await self.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=CrossPlatformOutput,
                temperature_override=0.5,
            )
            if result.selected_directions:
                return CrossPlatformReport(
                    entries=[CrossPlatformEntry(**e.model_dump()) for e in result.entries] if result.entries else [],
                    selected_directions=result.selected_directions,
                )
            logger.warning("Cross-platform structured output: no selected_directions → falling back to text")
        except Exception as e:
            logger.warning(f"Cross-platform structured generation failed: {e} → falling back to text")

        # Attempt 2: plain-text generation + regex parsing
        try:
            text_prompt = (
                user_prompt
                + """

请用以下格式输出推荐的创作方向（每行一个）：
推荐方向：题材名1
推荐方向：题材名2
推荐方向：题材名3
"""
            )
            text_result = await self.generate(
                system_prompt=system_prompt,
                user_prompt=text_prompt,
                temperature_override=0.5,
            )
            text = text_result.content if hasattr(text_result, "content") else str(text_result)
            import re

            directions = re.findall(r"推荐方向[：:]\s*(.+?)(?:\n|$)", text)
            directions = [d.strip() for d in directions if d.strip()]
            if directions:
                logger.info(f"Cross-platform text fallback: {len(directions)} directions")
                return CrossPlatformReport(selected_directions=directions[:5])
        except Exception as e2:
            logger.error(f"Cross-platform text fallback also failed: {e2}")

        # Attempt 3: extract genres directly from scan data
        scan_entries = []
        if feilu_ok:
            scan_entries.extend(feilu.entries)
        if fanqie_ok:
            scan_entries.extend(fanqie.entries)

        genre_set: set[str] = set()
        for entry in scan_entries[:20]:
            g = (entry.genre or "").strip()
            if g:
                for sub in g.replace("／", "/").split("/"):
                    sub = sub.strip()
                    if sub:
                        genre_set.add(sub)

        if genre_set:
            logger.info(f"Cross-platform genre fallback: {len(genre_set)} genres from scan data")
            return CrossPlatformReport(selected_directions=list(genre_set)[:5])

        return CrossPlatformReport()

    # ================================================================
    # Step 2: 对标书骨架拆解
    # ================================================================

    async def analyze_benchmarks(
        self,
        directions: list[str],
        feilu_scan: ScanReport | None = None,
        fanqie_scan: ScanReport | None = None,
    ) -> BenchmarkReport:
        """Step 2: Analyze benchmark book skeletons.

        For each retained direction, pick 2 representative books and
        extract their structural skeleton (not the content).
        """
        context = ""
        if feilu_scan:
            context += f"\n飞卢参考：{self._format_scan_for_prompt(feilu_scan)}"
        if fanqie_scan:
            context += f"\n番茄参考：{self._format_scan_for_prompt(fanqie_scan)}"

        system_prompt = self.build_system_prompt(
            role="网文结构分析师",
            expertise=(
                "你擅长拆解网文的结构骨架——不是复述正文，不是抄内容，而是提取"
                "一本书的核心结构要素：开局、金手指、压力设计、爽点循环、事件启动方式。"
                "你能识别哪些结构元素值得借鉴，哪些必须替换以避免高仿抄袭。"
            ),
        )

        user_prompt = f"""请从以下保留的题材方向里，每个方向选2本最有代表性的对标书，总共拆6-10本。

【保留的题材方向】
{chr(10).join(f"- {d}" for d in directions)}
{context}

注意：不是复述正文，不是抄内容，而是拆骨架。请按以下字段输出每本书：

| 书名 | 题材 | 书名吸量点 | 简介承诺的爽点 | 主角开局处境 | 金手指或核心底牌 | 开篇压力 | 首个小事件如何启动 | 爽点循环 | 哪些元素值得借 | 哪些元素必须换掉避免高仿 |

最后总结：每个题材最值得借的结构骨架是什么。
"""

        try:
            result = await self.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=BenchmarkOutput,
                temperature_override=0.5,
            )
            return BenchmarkReport(
                entries=[BenchmarkSkeleton(**e.model_dump()) for e in result.entries] if result.entries else [],
                structural_summary=result.structural_summary,
            )
        except Exception as e:
            logger.error(f"Benchmark analysis error: {e}")
            return BenchmarkReport()

    # ================================================================
    # Step 3A: 生成12个候选题材
    # ================================================================

    async def generate_topics(
        self,
        cross_platform: CrossPlatformReport,
        benchmarks: BenchmarkReport,
    ) -> CandidateTopicsOutput:
        """Step 3A: Generate 12 candidate topics from research.

        Generates in 3 batches of 4 to reduce structured-output complexity.
        Falls back to text-based parsing if structured generation fails.
        """
        cross_text = self._format_cross_platform(cross_platform)
        bench_text = self._format_benchmarks(benchmarks)

        system_prompt = self.build_system_prompt(
            role="网文创意策划师",
            expertise=(
                "你擅长基于市场数据和对标分析，批量生成有商业潜力的网文题材。"
                "你的创意方法：借骨架不抄内容——人物、开局、事件、金手指、核心爽点全部替换。"
                "每个题材都要适合番茄免费文平台，并且能让视频观众一眼听懂。"
            ),
        )

        base_prompt = f"""请基于双榜交叉结论 + 对标书骨架，生成候选题材。

【双榜交叉结论】
{cross_text}

【对标书骨架参考】
{bench_text}

要求：
1. 不抄正文，只借骨架。每个题材必须换：人物、开局、事件、金手指、核心爽点。
2. 每个题材都要适合发在番茄小说，且能让视频观众一眼听懂。
3. 每个题材都要能写出10章左右的首个小事件闭环。

请为每个候选题材输出以下字段：
- genre_name: 题材名
- one_line_setting: 一句话设定
- golden_finger: 核心金手指
- chapter1_conflict: 第一章冲突
- first_event_direction: 首个小事件闭环方向
- first_pleasure_wave: 第一波爽点
- tomato_fit: high/medium/low
- video_potential: high/medium/low
- risk: 风险点"""

        all_topics = []
        # Generate in 3 batches of 4 to reduce per-call complexity
        for batch_idx in range(3):
            batch_num = batch_idx + 1
            batch_prompt = (
                base_prompt
                + f"\n\n请生成第{batch_num}批，恰好4个候选题材（题材{batch_idx * 4 + 1}-{batch_idx * 4 + 4}）。"
            )

            batch_topics = await self._generate_topic_batch(
                system_prompt=system_prompt,
                user_prompt=batch_prompt,
                batch_num=batch_num,
            )
            all_topics.extend(batch_topics)
            logger.info(f"Batch {batch_num}: generated {len(batch_topics)} topics")

        logger.info(f"Total topics generated: {len(all_topics)} across 3 batches")
        return CandidateTopicsOutput(topics=all_topics)

    async def _generate_topic_batch(self, system_prompt: str, user_prompt: str, batch_num: int) -> list[CandidateTopic]:
        """Generate one batch of topics. Falls back to text parsing if structured fails."""

        # Attempt 1: structured generation
        try:
            result = await self.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=CandidateTopicsListOutput,
                temperature_override=0.8,
            )
            if result.topics:
                return [CandidateTopic(**t.model_dump()) for t in result.topics]
        except Exception as e:
            logger.warning(f"Batch {batch_num} structured generation failed: {e}")

        # Attempt 2: plain text generation + regex parsing
        logger.info(f"Batch {batch_num}: falling back to text-based generation")
        try:
            text_result = await self.generate(
                system_prompt=system_prompt
                + "\n\n请以markdown格式输出，每个题材用## 标题分隔，字段用- 字段名: 值的形式列出。",
                user_prompt=user_prompt,
                temperature_override=0.8,
            )
            topics = self._parse_text_topics(text_result.content)
            if topics:
                return topics
        except Exception as e2:
            logger.error(f"Batch {batch_num} text fallback also failed: {e2}")

        return []

    @staticmethod
    def _parse_text_topics(text: str) -> list[CandidateTopic]:
        """Parse markdown/text topic descriptions into CandidateTopic objects."""
        import re

        topics = []
        # Split by ## heading or numbered entries
        blocks = re.split(r"\n(?=## |\d+[\.\)]\s*)", text)

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 20:
                continue

            # Extract fields with regex
            def _field(pattern, default=""):
                m = re.search(pattern, block, re.IGNORECASE)
                return m.group(1).strip() if m else default

            genre = _field(r"(?:题材名|genre_name)[：:]\s*(.+)", "")
            if not genre:
                # Try to get from heading
                m = re.match(r"##\s*(.+)", block)
                if m:
                    genre = m.group(1).strip()

            if not genre:
                continue

            topics.append(
                CandidateTopic(
                    genre_name=genre,
                    one_line_setting=_field(r"(?:一句话设定|one_line_setting)[：:]\s*(.+)"),
                    golden_finger=_field(r"(?:核心金手指|golden_finger)[：:]\s*(.+)"),
                    chapter1_conflict=_field(r"(?:第一章冲突|chapter1_conflict)[：:]\s*(.+)"),
                    first_event_direction=_field(r"(?:首个小事件闭环方向|first_event_direction)[：:]\s*(.+)"),
                    first_pleasure_wave=_field(r"(?:第一波爽点|first_pleasure_wave)[：:]\s*(.+)"),
                    tomato_fit=_field(r"(?:番茄适配度|tomato_fit)[：:]\s*(.+)", "medium"),
                    video_potential=_field(r"(?:视频表现力|video_potential)[：:]\s*(.+)", "medium"),
                    risk=_field(r"(?:风险点|risk)[：:]\s*(.+)"),
                )
            )

        return topics

    # ================================================================
    # Step 3B: 评分收缩
    # ================================================================

    async def score_topics(self, candidates: CandidateTopicsOutput) -> TopicScoringReport:
        """Step 3B: Score 12 candidates across 8 dimensions, recommend top 4.

        Does NOT make the final decision — outputs a score table and
        top-4 recommendation for human selection.
        """
        topics_text = self._format_candidates(candidates)

        system_prompt = self.build_system_prompt(
            role="网文选题评估师",
            expertise=(
                "你擅长从商业可行性、创作稳定性和传播潜力三个维度评估网文题材。"
                "你的评分客观、有区分度，能指出每个题材的具体优劣。"
                "你不会替作者做最终决定，而是提供清晰的评分数据和建议。"
            ),
        )

        user_prompt = f"""请对这12个候选题材做评分，但不要替我最终决定。

【候选题材】
{topics_text}

评分维度（每项1-10分）：

1. 书名吸量潜力 — 题材本身能否产生高点击的书名
2. 第一章开篇压力 — 开局冲突是否足够强、足够快
3. 首个小事件闭环是否清楚 — 10章内的完整故事线是否清晰
4. 金手指是否能持续制造爽点 — 核心设定的爽点续航力
5. 番茄适配度 — 是否匹配番茄免费文的读者口味
6. AI辅助稳定性 — 让AI续写时是否容易偏离/崩设定
7. 视频表现力 — 做成推广视频时是否容易讲清楚、吸引人
8. 吃书/崩设定风险 — 长期写作中设定自相矛盾的概率（分数越低越好，即低分=高风险）

请输出：
- 完整的评分表（每项打分+综合得分）
- 推荐前4名（按综合得分排序）
- 每个人工选择建议：每个题材适合什么取舍
"""

        # Attempt 1: structured generation
        try:
            result = await self.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=TopicScoresOutput,
                temperature_override=0.3,
            )
            if result.top_4 and len(result.top_4) >= 2:
                return TopicScoringReport(
                    scores=[TopicScoreCard(**s.model_dump()) for s in result.scores] if result.scores else [],
                    top_4=result.top_4,
                )
            logger.warning(
                f"Topic scoring structured: only {len(result.top_4 or [])} top entries → falling back to text"
            )
        except Exception as e:
            logger.warning(f"Topic scoring structured generation failed: {e} → falling back to text")

        # Attempt 2: plain-text generation + regex parsing
        logger.info("Topic scoring: falling back to text-based generation")
        import re

        try:
            text_prompt = (
                user_prompt
                + """

请用以下格式输出结果：

推荐前4名：
1. 题材名1
2. 题材名2
3. 题材名3
4. 题材名4

评分表：
#排名 | 题材名 | 书名吸量 | 开篇压力 | 事件闭环 | 金手指续航 | 番茄适配 | AI稳定 | 视频表现 | 崩设定风险 | 综合得分
"""
            )
            text_result = await self.generate(
                system_prompt=system_prompt,
                user_prompt=text_prompt,
                temperature_override=0.3,
            )
            text = text_result.content if hasattr(text_result, "content") else str(text_result)

            # Parse top 4 from text
            top_4 = []
            top_match = re.search(r"推荐前4名[：:]?\s*\n(.+?)(?:\n\n|\n(?!\d+\.|\s*\d+\s))", text, re.DOTALL)
            if top_match:
                top_4 = re.findall(r"\d+\.\s*(.+?)(?:\n|$)", top_match.group(1))
                top_4 = [t.strip() for t in top_4 if t.strip()]
            if not top_4:
                # Try alternative formats
                top_4 = re.findall(r"(?:top|推荐)[\s_]*(?:4|四)[：:]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
                if top_4:
                    top_4 = [t.strip() for t in re.split(r"[,，、]", top_4[0]) if t.strip()]

            if top_4:
                logger.info(f"Topic scoring text fallback: top_4={top_4}")
                return TopicScoringReport(top_4=top_4)
        except Exception as e2:
            logger.error(f"Topic scoring text fallback also failed: {e2}")

        # Attempt 3: use first 4 candidate genres directly
        logger.info("Topic scoring: using first 4 candidate genres as fallback")
        if candidates.topics:
            fallback_top_4 = [t.genre_name for t in candidates.topics[:4] if t.genre_name]
            if fallback_top_4:
                return TopicScoringReport(top_4=fallback_top_4)

        return TopicScoringReport()

    # ================================================================
    # Step 4A: 书名与简介前置测试
    # ================================================================

    async def generate_titles(self, genre_name: str, topic: CandidateTopic) -> TitleSynopsisReport:
        """Step 4A: Generate title + synopsis for one topic.

        Uses plain-text generation with regex parsing (not structured output)
        for reliability across all LLM providers including ModelScope.
        """
        system_prompt = self.build_system_prompt(
            role="网文包装策划师",
            expertise=(
                "你擅长为网文设计高点击率的书名和简介。你熟悉番茄免费文的"
                "书名风格（直白、有冲突感、一眼看懂）和简介写法（快速建立期待、"
                "明确爽点、留钩子）。你也了解什么样的书名和简介适合做短视频推广。"
            ),
        )

        user_prompt = f"""现在针对以下题材，设计书名和简介。

【题材信息】
- 题材名：{genre_name}
- 一句话设定：{topic.one_line_setting}
- 核心金手指：{topic.golden_finger}
- 第一章冲突：{topic.chapter1_conflict}
- 首个小事件闭环方向：{topic.first_event_direction}
- 第一波爽点：{topic.first_pleasure_wave}

请按以下格式直接输出：

最终书名：《书名》
简介：2-4句话的简介，80-150字，快速建立期待、明确爽点、留钩子
"""

        # Attempt 1: structured generation (fast path for providers with JSON mode)
        try:
            result = await self.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=TitleSynopsisListOutput,
                temperature_override=0.8,
            )
            if result.final_title and result.final_synopsis:
                return TitleSynopsisReport(
                    genre_name=result.genre_name or genre_name,
                    title_candidates=[TitleSynopsisPair(**t.model_dump()) for t in result.title_candidates]
                    if result.title_candidates
                    else [],
                    final_title=result.final_title,
                    final_synopsis=result.final_synopsis,
                )
            logger.warning("Title structured: empty title/synopsis → falling back to text")
        except Exception as e:
            logger.warning(f"Title structured generation failed: {e} → falling back to text")

        # Attempt 2: plain-text generation + regex parsing
        try:
            text_result = await self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature_override=0.8,
            )
            text = text_result.content if hasattr(text_result, "content") else str(text_result)
            import re

            # Parse: 最终书名：《XXX》 or 书名：XXX
            title_match = re.search(r"(?:最终)?书名[：:]\s*[《]?(.+?)[》]?(?:\n|$)", text)
            final_title = title_match.group(1).strip() if title_match else ""

            # Parse: 简介：XXX
            syn_match = re.search(r"简介[：:]\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
            final_synopsis = syn_match.group(1).strip() if syn_match else ""

            # Fallback: use first line as title if no match
            if not final_title:
                lines = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.startswith("#")]
                if lines:
                    # Clean up common artifacts
                    first = re.sub(r"^[《「]|[》」]$", "", lines[0]).strip()
                    if len(first) >= 4:
                        final_title = first[:50]
                    if len(lines) >= 2:
                        final_synopsis = " ".join(lines[1:3])[:300]

            if final_title:
                logger.info(f"Title text fallback: '{final_title}' for '{genre_name}'")
                return TitleSynopsisReport(
                    genre_name=genre_name,
                    final_title=final_title,
                    final_synopsis=final_synopsis or topic.one_line_setting or f"一部{genre_name}题材的精彩小说",
                )
        except Exception as e2:
            logger.error(f"Title text fallback also failed: {e2}")

        # Attempt 3: build from topic data (no LLM)
        fallback_title = topic.one_line_setting or genre_name
        return TitleSynopsisReport(
            genre_name=genre_name,
            final_title=fallback_title[:50],
            final_synopsis=f"一部关于{genre_name}的精彩小说。{topic.golden_finger or '金手指'}，{topic.chapter1_conflict or '开局冲突'}，{topic.first_pleasure_wave or '爽点不断'}。",
        )

    # ================================================================
    # Step 5: 10章小事件闭环大纲
    # ================================================================

    async def plan_mini_arc(
        self,
        genre_name: str,
        topic: CandidateTopic,
        title_report: TitleSynopsisReport,
    ) -> MiniArcOutline:
        """Step 5: Design a 10-chapter mini-arc with a complete small event loop.

        Each chapter is 2000-2100 words, total ~20,000-21,000 words.
        The first small event must form a complete closed loop.

        Tries structured generation first, falls back to plain-text generation
        + regex parsing on failure (needed for ModelScope which lacks JSON mode).
        """
        system_prompt = self.build_system_prompt(
            role="网文结构设计师",
            expertise=(
                "你擅长设计紧凑的10章小事件闭环。你理解网文的节奏规律："
                "第1章给压力 → 第2-3章展示规则和底牌 → 第4-7章升级阻碍 → "
                "第8-9章布局反击 → 第10章释放爽点、收束事件、抛新钩子。"
                "每章2000-2100字，10章总计约2万字。"
            ),
        )

        user_prompt = f"""请基于以下题材信息，设计"10章小事件闭环大纲"。

【题材信息】
- 题材名：{genre_name}
- 一句话设定：{topic.one_line_setting}
- 核心金手指：{topic.golden_finger}
- 第一章冲突：{topic.chapter1_conflict}
- 首个小事件闭环方向：{topic.first_event_direction}
- 第一波爽点：{topic.first_pleasure_wave}

【书名和简介】
- 书名：{title_report.final_title}
- 简介：{title_report.final_synopsis}

每章按2000-2100字规划，10章总计约20,000-21,000字。首个小事件必须完整闭环，不能只是开头。

请按以下结构输出每章：

| 字段 | 说明 |
|------|------|
| 章数 | 第1-10章 |
| 本章目标 | 这一章要达成什么 |
| 本章冲突 | 核心矛盾是什么 |
| 本章爽点 | 读者的满足感来自哪里 |
| 新增信息 | 读者获得什么新认知 |
| 人物状态变化 | 主角/配角的情感或能力变化 |
| 伏笔推进 | 埋下或推进了什么伏笔 |
| 结尾钩子 | 如何让读者点下一章 |

节奏要求：
- 第1章：给开局压力（冲突立即出现）
- 第2-3章：展示世界观规则和主角底牌
- 第4-7章：升级阻碍和冲突
- 第8-9章：完成布局与反击
- 第10章：释放爽点、收束小事件，并抛出下一事件钩子
"""

        # Attempt 1: structured generation
        try:
            result = await self.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=MiniArcOutlineOutput,
                temperature_override=0.6,
            )
            # Check chapters exist AND have actual content (not just empty shells)
            if result.chapters and len(result.chapters) >= 5:
                filled = sum(1 for c in result.chapters if c.goal or c.conflict or c.pleasure_point)
                if filled >= 5:
                    return MiniArcOutline(
                        genre_name=result.genre_name or genre_name,
                        total_words=result.total_words or "20,000-21,000",
                        chapters=[MiniArcChapter(**c.model_dump()) for c in result.chapters],
                        arc_goal=result.arc_goal,
                        next_arc_hook=result.next_arc_hook,
                    )
                logger.warning(
                    f"Mini-arc structured: {len(result.chapters)} chapters but only {filled} have content → falling back to text"
                )
            else:
                logger.warning(
                    f"Mini-arc structured: only {len(result.chapters or [])} chapters → falling back to text"
                )
        except Exception as e:
            logger.warning(f"Mini-arc structured generation failed: {e} → falling back to text")

        # Attempt 2: plain-text generation + regex parsing
        logger.info(f"Mini-arc for '{genre_name}': falling back to text-based generation")
        try:
            text_prompt = (
                user_prompt
                + """

请用以下纯文本格式输出每章（必须输出全部10章）：

第1章
本章目标：...
本章冲突：...
本章爽点：...
新增信息：...
人物变化：...
伏笔推进：...
结尾钩子：...

第2章
本章目标：...
（以此类推，直到第10章）

闭环总结
弧目标：...
下一弧钩子：...
"""
            )
            text_result = await self.generate(
                system_prompt=system_prompt,
                user_prompt=text_prompt,
                temperature_override=0.6,
            )
            text = text_result.content if hasattr(text_result, "content") else str(text_result)
            chapters, arc_goal, next_hook = self._parse_text_mini_arc(text)

            if chapters and len(chapters) >= 5:
                logger.info(f"Mini-arc text fallback: parsed {len(chapters)} chapters for '{genre_name}'")
                return MiniArcOutline(
                    genre_name=genre_name,
                    total_words="20,000-21,000",
                    chapters=chapters,
                    arc_goal=arc_goal,
                    next_arc_hook=next_hook,
                )
            logger.warning(f"Text fallback produced only {len(chapters)} chapters → building deterministic chapters")
        except Exception as e2:
            logger.error(f"Mini-arc text fallback also failed: {e2}")

        # Attempt 3: deterministic chapter generation from topic data
        logger.info(f"Mini-arc for '{genre_name}': building deterministic chapters from topic data")
        return _build_deterministic_mini_arc(genre_name, topic, title_report)

    @staticmethod
    def _parse_text_mini_arc(text: str) -> tuple[list[MiniArcChapter], str, str]:
        """Parse plain-text mini-arc outline into chapter list.

        Accepts multiple formats:
          === 第N章 ===  /  ## 第N章  /  第N章  /  N.
        With field labels in Chinese or abbreviated form.
        """
        import re

        chapters = []
        # Split by any chapter marker: === 第N章 ===, ## 第N章, 第N章, N.
        chapter_blocks = re.split(
            r"\n(?=(?:===?\s*)?第\s*\d+\s*章|^\d+[\.\)、])",
            text,
            flags=re.MULTILINE,
        )
        # Also try splitting by double-newline if the above yields too few blocks
        if len(chapter_blocks) < 5:
            chapter_blocks = re.split(r"\n\s*\n(?=第)", text)

        for block in chapter_blocks:
            match = re.search(r"第\s*(\d+)\s*章", block)
            if not match:
                # Try numbered format: "1." or "1、"
                match = re.search(r"(?:^|\n)\s*(\d+)\s*[\.\)、]", block)
            if not match:
                continue
            ch_num = int(match.group(1))
            if ch_num < 1 or ch_num > 50:
                continue

            def _f(*patterns, default=""):
                for pat in patterns:
                    m = re.search(pat, block)
                    if m:
                        return m.group(1).strip()
                return default

            chapters.append(
                MiniArcChapter(
                    chapter_number=ch_num,
                    goal=_f(
                        r"(?:本章)?目标[：:]\s*(.+?)(?:\n|$)",
                        r"目标[：:]\s*(.+?)(?:\n|$)",
                    ),
                    conflict=_f(
                        r"(?:本章)?冲突[：:]\s*(.+?)(?:\n|$)",
                        r"冲突[：:]\s*(.+?)(?:\n|$)",
                        r"矛盾[：:]\s*(.+?)(?:\n|$)",
                    ),
                    pleasure_point=_f(
                        r"(?:本章)?爽点[：:]\s*(.+?)(?:\n|$)",
                        r"爽点[：:]\s*(.+?)(?:\n|$)",
                    ),
                    new_info=_f(
                        r"新增?信息[：:]\s*(.+?)(?:\n|$)",
                        r"新信息[：:]\s*(.+?)(?:\n|$)",
                    ),
                    character_change=_f(
                        r"人物[状变][态化][：:]\s*(.+?)(?:\n|$)",
                        r"人物变化[：:]\s*(.+?)(?:\n|$)",
                    ),
                    foreshadowing=_f(
                        r"伏笔[推进][进前][：:]\s*(.+?)(?:\n|$)",
                        r"伏笔[：:]\s*(.+?)(?:\n|$)",
                    ),
                    ending_hook=_f(
                        r"(?:结尾)?钩子[：:]\s*(.+?)(?:\n|$)",
                        r"钩子[：:]\s*(.+?)(?:\n|$)",
                    ),
                )
            )

        chapters.sort(key=lambda c: c.chapter_number)

        # Parse summary section (multiple possible formats)
        arc_goal = ""
        next_hook = ""
        # Try to find summary section with various markers
        summary_match = re.search(
            r"(?:===?\s*)?闭环总结\s*(?:===?\s*)?\n?(.+?)(?:\Z)",
            text,
            re.DOTALL,
        )
        if not summary_match:
            # Also try "总结" or "Summary" section
            summary_match = re.search(
                r"(?:===?\s*)?(?:总结|Summary|概要)\s*(?:===?\s*)?\n?(.+?)(?:\Z)",
                text,
                re.DOTALL,
            )
        if summary_match:
            summary = summary_match.group(1)
            m = re.search(r"弧目标[：:]\s*(.+?)(?:\n|$)", summary)
            if m:
                arc_goal = m.group(1).strip()
            m = re.search(r"(?:下一弧)?钩子[：:]\s*(.+?)(?:\n|$)", summary)
            if m:
                next_hook = m.group(1).strip()
        # Fallback: search for arc_goal/next_hook anywhere in text
        if not arc_goal:
            m = re.search(r"弧目标[：:]\s*(.+?)(?:\n|$)", text)
            if m:
                arc_goal = m.group(1).strip()
        if not next_hook:
            m = re.search(r"(?:下一弧)?钩子[：:]\s*(.+?)(?:\n|$)", text)
            if m:
                next_hook = m.group(1).strip()

        return chapters, arc_goal, next_hook

    # ================================================================
    # Prompt formatters (private)
    # ================================================================

    @staticmethod
    def _format_scan_for_prompt(report: ScanReport) -> str:
        """Format a ScanReport as prompt text."""
        if report.scan_failed:
            return f"[{report.platform}] 本次扫榜失败: {report.summary}"
        lines = [
            f"平台：{report.platform}",
            f"榜单：{report.list_name}",
            f"日期：{report.date}",
            f"共 {len(report.entries)} 条记录",
            "",
        ]
        for e in report.entries:
            lines.append(
                f"#{e.rank} {e.title} | 题材：{e.genre} | 金手指：{e.golden_finger} | 爽点循环：{e.pleasure_loop}"
            )
        lines.append(f"\n趋势总结：{report.summary}")
        return "\n".join(lines)

    @staticmethod
    def _format_cross_platform(report: CrossPlatformReport) -> str:
        """Format a CrossPlatformReport as prompt text."""
        lines = []
        for e in report.entries:
            lines.append(
                f"- {e.genre}: 飞卢热度={'是' if e.feilu_heat else '否'}, "
                f"番茄基础={'是' if e.tomato_audience else '否'}, "
                f"飞卢吸人点={e.feilu_appeal}, "
                f"番茄调整={e.tomato_adjustment}, "
                f"风险={e.risk}, 推荐={e.recommendation}"
            )
        if report.selected_directions:
            lines.append(f"\n保留方向：{', '.join(report.selected_directions)}")
        return "\n".join(lines)

    @staticmethod
    def _format_benchmarks(report: BenchmarkReport) -> str:
        """Format a BenchmarkReport as prompt text."""
        lines = []
        for e in report.entries:
            lines.append(
                f"- {e.title} ({e.genre}): "
                f"开局={e.protagonist_opening}, 金手指={e.golden_finger}, "
                f"爽点循环={e.pleasure_loop}"
            )
        lines.append(f"\n结构总结：{report.structural_summary}")
        return "\n".join(lines)

    @staticmethod
    def _format_candidates(output: CandidateTopicsOutput) -> str:
        """Format CandidateTopicsOutput as prompt text."""
        lines = []
        for i, t in enumerate(output.topics, 1):
            lines.append(
                f"{i}. {t.genre_name}: {t.one_line_setting}\n"
                f"   金手指={t.golden_finger}, 首章冲突={t.chapter1_conflict}, "
                f"番茄适配={t.tomato_fit}, 视频表现={t.video_potential}"
            )
        return "\n".join(lines)


# ============================================================
# Deterministic mini-arc builder (final fallback, no LLM)
# ============================================================


def _build_deterministic_mini_arc(
    genre_name: str,
    topic: "CandidateTopic",
    title_report: "TitleSynopsisReport",
) -> "MiniArcOutline":
    """Build a basic 10-chapter mini-arc outline from topic data with no LLM.

    This is the ultimate fallback when both structured and text generation fail.
    Uses standard web novel pacing: pressure → rules → obstacles → counterattack → payoff.
    """
    golden = topic.golden_finger or "金手指"
    conflict = topic.chapter1_conflict or "开篇冲突"
    pleasure = topic.first_pleasure_wave or "打脸逆袭"
    title = title_report.final_title or genre_name

    chapter_template = [
        {
            "goal": f"开局建立冲突：{conflict}",
            "conflict": f"主角面临{conflict}的困境",
            "pleasure_point": "开篇压力制造悬念，读者期待反转",
            "new_info": "引入世界观基本设定，展示主角处境",
            "character_change": "主角从被动承受转向主动应对",
            "foreshadowing": f"暗示{golden}的存在",
            "ending_hook": f"主角发现{golden}的线索，读者期待下一章",
        },
        {
            "goal": f"展示{golden}的初步能力",
            "conflict": "主角尝试使用金手指但遭遇阻力",
            "pleasure_point": f"{golden}首次展现，给读者爽感",
            "new_info": f"揭示{golden}的基本规则和限制",
            "character_change": "主角开始建立自信",
            "foreshadowing": "埋下更大的世界观伏笔",
            "ending_hook": f"主角用{golden}获得第一次小胜利",
        },
        {
            "goal": "建立主角的第一个小目标",
            "conflict": "外部势力开始注意到主角的变化",
            "pleasure_point": "主角用能力解决一个小麻烦",
            "new_info": "展示这个世界的力量体系或社会规则",
            "character_change": "主角开始主动规划行动",
            "foreshadowing": "引入第一个小反派或竞争对手",
            "ending_hook": "反派开始注意到主角",
        },
        {
            "goal": "主角遭遇第一次重大挫折",
            "conflict": "对手利用规则或势力压制主角",
            "pleasure_point": "主角在逆境中发现金手指的新用法",
            "new_info": "揭示对手的背景和动机",
            "character_change": "主角认识到需要变强",
            "foreshadowing": "暗示背后更大的阴谋",
            "ending_hook": "主角制定反击计划",
        },
        {
            "goal": "主角开始布局反击",
            "conflict": "资源不足，需要争取盟友或资源",
            "pleasure_point": "主角用智慧获得关键资源或盟友",
            "new_info": "引入新角色（盟友或导师）",
            "character_change": "主角从独行侠变为有支持者",
            "foreshadowing": "埋下盟友身份的伏笔",
            "ending_hook": "准备阶段完成，行动即将开始",
        },
        {
            "goal": "第一次正面交锋",
            "conflict": f"主角用{golden}与对手正面碰撞",
            "pleasure_point": "战斗或较量中的高光时刻",
            "new_info": "揭示对手的弱点和秘密",
            "character_change": "主角实战经验增长",
            "foreshadowing": "发现对手背后有更大的势力",
            "ending_hook": "暂时击退对手但发现更大的威胁",
        },
        {
            "goal": "主角消化战斗收获，准备更大行动",
            "conflict": "新威胁出现，主角需要升级",
            "pleasure_point": f"{golden}升级或获得新能力",
            "new_info": "揭示世界观的更深层次",
            "character_change": "主角实力和心智双重成长",
            "foreshadowing": "暗示最终boss的存在",
            "ending_hook": "主角下定决心面对最终挑战",
        },
        {
            "goal": "主角主动出击",
            "conflict": "深入对手地盘或势力范围",
            "pleasure_point": "连续打脸，层层推进的爽感",
            "new_info": "揭示事件背后的真相",
            "character_change": "主角从被动变为主动掌控局面",
            "foreshadowing": "回收之前埋下的伏笔",
            "ending_hook": "即将迎来最终对决",
        },
        {
            "goal": "决战时刻",
            "conflict": "主角用全力与最终对手决战",
            "pleasure_point": "最高潮的爽点释放",
            "new_info": "揭示事件的全部真相",
            "character_change": "主角完成阶段性成长",
            "foreshadowing": "埋下下一卷的伏笔",
            "ending_hook": "胜利在望",
        },
        {
            "goal": "收束第一条故事线",
            "conflict": "处理决战后遗留的问题",
            "pleasure_point": f"{pleasure}的完整释放",
            "new_info": "展示主角的新身份或新处境",
            "character_change": "主角完成本卷的角色弧光",
            "foreshadowing": "为下一卷埋下新钩子",
            "ending_hook": "新的挑战或机遇出现，读者期待下一卷",
        },
    ]

    chapters = []
    for i, tmpl in enumerate(chapter_template, 1):
        chapters.append(
            MiniArcChapter(
                chapter_number=i,
                goal=tmpl["goal"],
                conflict=tmpl["conflict"],
                pleasure_point=tmpl["pleasure_point"],
                new_info=tmpl["new_info"],
                character_change=tmpl["character_change"],
                foreshadowing=tmpl["foreshadowing"],
                ending_hook=tmpl["ending_hook"],
            )
        )

    return MiniArcOutline(
        genre_name=genre_name,
        total_words="20,000-21,000",
        chapters=chapters,
        arc_goal=f"完成{conflict}的第一个小闭环，实现{pleasure}",
        next_arc_hook=f"第一事件收束后，更大的世界向主角展开。《{title}》的第二卷即将开始。",
    )
