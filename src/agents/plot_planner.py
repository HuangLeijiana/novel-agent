"""Plot Planner Agent — master outline, volume structure, chapter planning."""

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from ..llm.scheduler import ModelScheduler
from ..models.bible import NovelBible
from ..models.characters import CharacterRegistry
from ..models.memory import MemoryState
from ..models.outline import (
    ChapterPlan,
    EmotionalBeat,
    Hook,
    MasterOutline,
    PlotArc,
    Scene,
    TurningPoint,
    Volume,
)
from ..models.project import ProjectConfig
from .base import BaseAgent

logger = logging.getLogger(__name__)


# ============================================================
# Structured output schemas
# ============================================================


class MasterOutlineOutput(BaseModel):
    """LLM output for master outline."""

    title: str = Field(default="", description="小说标题")
    subtitle: Optional[str] = Field(default=None, description="副标题")
    logline: str = Field(default="", description="一句话简介")
    main_plot: list[PlotArc] = Field(default_factory=list)
    subplots: list[PlotArc] = Field(default_factory=list)
    volumes: list[Volume] = Field(default_factory=list)
    major_turning_points: list[TurningPoint] = Field(default_factory=list)
    chapter_count: int = Field(default=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Coerce Chinese LLM quirks: string turning points → TurningPoint."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "major_turning_points" in d:
            tps = d["major_turning_points"]
            if isinstance(tps, list):
                coerced = []
                for i, tp in enumerate(tps):
                    if isinstance(tp, str):
                        coerced.append({"description": tp, "chapter": i + 1})
                    elif isinstance(tp, dict):
                        coerced.append(tp)
                    else:
                        coerced.append(tp)
                d["major_turning_points"] = coerced
        # Also coerce string plot arcs
        for field in ("main_plot", "subplots"):
            if field in d and isinstance(d[field], list):
                d[field] = [{"description": x} if isinstance(x, str) else x for x in d[field]]
        # Coerce string volumes (Qwen3 may output "第X卷：标题...描述")
        if "volumes" in d and isinstance(d["volumes"], list):
            coerced_vols = []
            for i, v in enumerate(d["volumes"]):
                if isinstance(v, str):
                    # Parse "第X卷：标题" or "第X卷 标题" or just "标题"
                    import re as _re

                    vol_num = i + 1
                    vol_title = v
                    # Try to extract volume number from string
                    m = _re.match(r"第\s*([\d一二三四五六七八九十]+)\s*[卷部篇]\s*[：:\s]\s*(.+)", v)
                    if m:
                        # Convert Chinese number if needed (simplified: use index-based fallback)
                        try:
                            vol_num = int(m.group(1))
                        except ValueError:
                            vol_num = i + 1
                        vol_title = m.group(2).strip()
                    else:
                        # Fallback: use the whole string as title, auto-number
                        vol_title = v
                    coerced_vols.append(
                        {
                            "title": vol_title,
                            "number": vol_num,
                            "start_chapter": (i * 10) + 1,
                            "end_chapter": (i + 1) * 10,
                        }
                    )
                elif isinstance(v, dict):
                    coerced_vols.append(v)
                else:
                    coerced_vols.append(v)
            d["volumes"] = coerced_vols
        return d


def _parse_emotional_beat_string(raw: str, index: int = 0) -> dict:
    """Parse a Chinese LLM emotional beat string into a dict.

    Handles formats like:
      "压抑，强度0.9"
      "愤怒, 强度0.7"
      "紧张 0.8"
      "激动：0.95"
      "压抑（强度0.9）"   ← parenthesized intensity
    """
    import re

    s = raw.strip()

    # Try parenthesized intensity: "压抑（强度0.9）" or "压抑(强度0.9)"
    m = re.search(r"[（(]\s*强度\s*([\d.]+)\s*[）)]", s)
    if m:
        emotion = s[: m.start()].strip()
        intensity = float(m.group(1))
        if intensity > 1.0:
            intensity /= 10.0
        return {"position": index * 0.25, "emotion": emotion, "intensity": intensity}

    # Try "情绪，强度N" / "情绪,强度N" / "情绪, 强度N"
    m = re.search(r"[，,]\s*强度\s*([\d.]+)", s)
    if m:
        emotion = s[: m.start()].strip()
        intensity = float(m.group(1))
        if intensity > 1.0:
            intensity /= 10.0
        return {"position": index * 0.25, "emotion": emotion, "intensity": intensity}

    # Try "情绪 N" or "情绪：N" (emotion followed by number)
    m = re.search(r"[：:]\s*([\d.]+)$", s)
    if m:
        emotion = s[: m.start()].strip()
        intensity = float(m.group(1))
        if intensity > 1.0:
            intensity /= 10.0
        return {"position": index * 0.25, "emotion": emotion, "intensity": intensity}

    m = re.search(r"\s+([\d.]+)$", s)
    if m:
        emotion = s[: m.start()].strip()
        intensity = float(m.group(1))
        if intensity > 1.0:
            intensity /= 10.0
        return {"position": index * 0.25, "emotion": emotion, "intensity": intensity}

    # Fallback: treat entire string as emotion name
    return {"position": index * 0.25, "emotion": s, "intensity": 0.5}


def _parse_emotional_curve_string(raw: str) -> list[dict]:
    """Parse a comma/line-separated emotional curve string into a list of dicts.

    Handles: "压抑强度0.9，愤怒强度0.7，紧张强度0.95"
    """
    import re

    # Split on Chinese/English commas, newlines, or "强度" runs
    # Strategy: split on "，" or "," and parse each fragment
    parts = re.split(r"[，,]\s*", raw.strip())
    if len(parts) <= 1:
        parts = re.split(r"\n+", raw.strip())

    result = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        beat = _parse_emotional_beat_string(part, i)
        result.append(beat)
    return result


class ChapterPlanOutput(BaseModel):
    """LLM output for chapter plan."""

    chapter_number: int = Field(default=0)
    title: str = Field(default="")
    goal: str = Field(default="")
    scenes: list[Scene] = Field(default_factory=list)
    pov_character: str = Field(default="")
    conflict: str = Field(default="")
    hooks: list[Hook] = Field(default_factory=list)
    characters_involved: list[str] = Field(default_factory=list)
    information_increment: str = Field(default="")
    reveals: list[str] = Field(default_factory=list)
    foreshadowing: list[str] = Field(default_factory=list)
    emotional_curve: list[EmotionalBeat] = Field(default_factory=list)
    ending_hook: str = Field(default="")
    word_count_target: int = Field(default=4000)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        """Pre-normalize nested fields before model validation."""
        if not isinstance(data, dict):
            return data
        d = dict(data)

        # ── emotional_curve: coerce string items & clamp intensities ──
        if "emotional_curve" in d:
            curve = d["emotional_curve"]
            if isinstance(curve, list):
                coerced = []
                for i, beat in enumerate(curve):
                    if isinstance(beat, str):
                        beat = _parse_emotional_beat_string(beat, i)
                    if isinstance(beat, dict):
                        # If emotion field contains parenthesized intensity, extract it
                        if "emotion" in beat and isinstance(beat["emotion"], str):
                            raw_emotion = beat["emotion"]
                            parsed = _parse_emotional_beat_string(raw_emotion, i)
                            # Use parsed emotion if it's cleaner (shorter, no parens)
                            if "（" not in parsed["emotion"] and "(" not in parsed["emotion"]:
                                beat["emotion"] = parsed["emotion"]
                            else:
                                # Strip leading position number if present (e.g. "0.0 压抑" → "压抑")
                                import re

                                cleaned = re.sub(r"^[\d.]+\s+", "", raw_emotion, count=1)
                                if cleaned != raw_emotion:
                                    beat["emotion"] = cleaned
                            # Use parsed intensity if current is missing or default
                            if "intensity" not in beat or beat.get("intensity") in (0, 0.5):
                                beat["intensity"] = parsed["intensity"]
                        # Clamp intensity: model often uses 0–10 scale
                        if "intensity" in beat:
                            v = beat["intensity"]
                            if isinstance(v, (int, float)) and v > 1.0:
                                beat["intensity"] = v / 10.0
                    coerced.append(beat)
                d["emotional_curve"] = coerced
            elif isinstance(curve, str):
                # Single string — try to parse as comma-separated list
                d["emotional_curve"] = _parse_emotional_curve_string(curve)

        return d


# ============================================================
# Agent
# ============================================================


class PlotPlannerAgent(BaseAgent):
    """Plans the complete story structure — from master outline to individual chapters."""

    agent_type = "plot_planner"

    # ================================================================
    # Master Outline
    # ================================================================

    async def create_master_outline(
        self,
        config: ProjectConfig,
        bible: NovelBible,
        characters: CharacterRegistry,
    ) -> MasterOutline:
        """Create the complete master outline with volumes and turning points."""
        logger.info("Creating master outline...")

        system = self.build_system_prompt(
            role="故事架构师",
            expertise="设计引人入胜的故事结构。精通三幕剧、英雄之旅、起承转合等叙事结构，"
            "擅长规划主线、支线、转折点和章节节奏。尤其精通{genre}类型的故事设计。".format(
                genre="、".join(config.genre)
            ),
        )

        # Build character summary
        char_lines = []
        for cid, char in characters.characters.items():
            char_lines.append(f"- [{cid}] {char.name}（{char.role}）: {char.motivation[:80]}")
        char_summary = "\n".join(char_lines)

        world = bible.world
        conflicts = bible.core_conflicts
        themes = bible.themes

        conflict_lines = "\n".join(f"- [{c.id}] {c.description}" for c in conflicts)
        theme_lines = ", ".join(t.name for t in themes)

        # Estimate chapter count
        avg_words_per_chapter = 4000
        total_chapters = max(10, config.target_word_count // avg_words_per_chapter)

        user = f"""请为以下小说设计完整的故事结构：

【灵感】{config.inspiration}
【题材】{", ".join(config.genre)}
【目标读者】{config.target_readers}
【篇幅】约{config.target_word_count}字（预估{total_chapters}章，每章约{avg_words_per_chapter}字）

【世界观】{world.name}（{world.world_type}）
【主题】{theme_lines}
【核心冲突】
{conflict_lines}

【角色】
{char_summary}

请设计：

### 1. 基本信息
- 小说标题
- 一句话简介（logline）

### 2. 主线剧情（main_plot）
- 分配 3-5 个主要剧情弧线（arc）
- 每个弧线包含：起止章、关键节拍（beats）

### 3. 支线剧情（subplots）
- 2-3 条支线
- 每条标注类型（subplot / b_plot）

### 4. 分卷结构（volumes）
- 将故事分为 {max(2, total_chapters // 30)} - {max(3, total_chapters // 20)} 卷
- 每卷：标题、简介、起止章节、主要事件、情感弧线

### 5. 关键转折点（major_turning_points）
- 至少包括以下类型节点各一个：
  - inciting_incident（激励事件）
  - first_plot_point（第一转折点）
  - midpoint（中点转折）
  - all_is_lost（一切尽失）
  - climax（高潮）
  - denouement（结局）

设计原则：
- 确保每一章都有推动剧情的信息增量
- 爽点分布在关键章节处
- 主线与支线有机交织
- 为角色弧线留出展示空间"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=MasterOutlineOutput,
            temperature_override=0.7,
            max_tokens_override=8192,
        )

        outline = MasterOutline(
            title=result.title or "",
            subtitle=result.subtitle,
            logline=result.logline,
            main_plot=result.main_plot,
            subplots=result.subplots,
            volumes=result.volumes,
            major_turning_points=result.major_turning_points,
            chapter_count=result.chapter_count or total_chapters,
        )

        logger.info(
            f"Master outline created: {outline.title}, {len(outline.volumes)} volumes, {outline.chapter_count} chapters"
        )
        return outline

    # ================================================================
    # Chapter Planning
    # ================================================================

    async def plan_chapter(
        self,
        chapter_number: int,
        config: ProjectConfig,
        bible: NovelBible,
        characters: CharacterRegistry,
        outline: MasterOutline,
        memory: Optional[MemoryState] = None,
    ) -> ChapterPlan:
        """Plan a single chapter in detail."""
        logger.info(f"Planning chapter {chapter_number}...")

        system = self.build_system_prompt(
            role="章节规划师",
            expertise="将故事大纲拆解为可执行的章节计划。确保每章有明确的目标、冲突、"
            "信息增量和章末钩子，让读者欲罢不能。",
        )

        # Determine which volume this chapter belongs to
        current_volume = None
        for vol in outline.volumes:
            if vol.start_chapter <= chapter_number <= vol.end_chapter:
                current_volume = vol
                break

        # Find turning points in this chapter
        chapter_tps = [tp for tp in outline.major_turning_points if tp.chapter == chapter_number]

        # Build context
        outline_json = outline.model_dump_json(indent=2)

        prev_chapter_summary = ""
        if memory and memory.short_term:
            prev_chapter_summary = memory.short_term.current_chapter_summary

        user = f"""请为以下小说规划第 {chapter_number} 章的详细方案：

【大纲】
{outline_json}

【当前卷】{current_volume.title if current_volume else "未分配"} — {current_volume.logline if current_volume else ""}

【本章转折点】{[(tp.turning_type, tp.description) for tp in chapter_tps] if chapter_tps else "无特定转折点"}

【上一章摘要】{prev_chapter_summary or "这是第一章"}

【角色当前状态】
{self._format_character_states(characters, memory)}

请设计：

1. **章节标题**：有吸引力的标题
2. **本章目标**：这一章必须完成什么（推动哪条线、达成什么结果）
3. **场景拆分**（2-5 个场景）：
   - 每个场景设定地点、POV角色、在场角色、场景目标、冲突、结果
4. **核心冲突**：本章的主要矛盾
5. **信息增量**：读者在这一章能获得什么新信息
6. **揭示**（reveals）：是否有反转或揭示
7. **伏笔**（foreshadowing）：本章埋什么伏笔
8. **情绪曲线**：5个情绪节点（位置0.0/0.25/0.5/0.75/1.0），标注情绪和强度
9. **章末钩子**：如何让读者迫不及待看下一章
10. **目标字数**：{bible.rules.chapter_length_range[0]}-{bible.rules.chapter_length_range[1]}字

设计原则：
- 场景之间要有因果链
- 情绪有起伏，不要平铺直叙
- 章末必须有钩子（悬念/情绪/揭示）
- 符合{config.tone}的语调"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ChapterPlanOutput,
            temperature_override=0.7,
        )

        plan = ChapterPlan(
            chapter_number=chapter_number,
            title=result.title,
            goal=result.goal,
            scenes=result.scenes,
            pov_character=result.pov_character,
            conflict=result.conflict,
            hooks=result.hooks,
            characters_involved=result.characters_involved,
            information_increment=result.information_increment,
            reveals=result.reveals,
            foreshadowing=result.foreshadowing,
            emotional_curve=result.emotional_curve,
            ending_hook=result.ending_hook,
            word_count_target=result.word_count_target or 4000,
            status="planned",
        )

        logger.info(f"Chapter {chapter_number} plan ready: {len(plan.scenes)} scenes")
        return plan

    def _format_character_states(
        self,
        characters: CharacterRegistry,
        memory: Optional[MemoryState],
    ) -> str:
        """Format current character states for the prompt."""
        lines = []
        for cid, char in characters.characters.items():
            state = char.current_state
            # Override with memory state if available
            if memory and cid in memory.character_states:
                state = memory.character_states[cid]

            # Normalise: CharacterState model → dict, memory state may already be dict
            if not isinstance(state, dict):
                state = state.model_dump()

            lines.append(
                f"- [{cid}] {char.name}: 位置={state.get('physical', {}).get('location', '未知')}, "
                f"情绪={state.get('emotional', {}).get('mood', '正常')}, "
                f"弧线进度={state.get('arc_progress', 0.0):.0%}"
            )
        return "\n".join(lines)
