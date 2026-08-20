"""Topic research and commercial web novel pipeline models.

Covers the upstream workflow: platform scanning → benchmark analysis →
topic generation → scoring → title/synopsis → mini-arc outline.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Step 1A/1B: Platform Scanning
# ============================================================


class ScanEntry(BaseModel):
    """A single entry from a platform rankings scan."""

    rank: int = Field(default=0, description="Rank position on the list")
    title: str = Field(default="", description="Book title")
    genre: str = Field(default="", description="Genre / category tags")
    title_appeal: str = Field(default="", description="What makes the title attention-grabbing")
    one_liner: str = Field(default="", description="One-sentence selling point")
    golden_finger: str = Field(default="", description="Core cheat / unique ability")
    opening_pressure: str = Field(default="", description="Chapter 1 opening pressure / conflict")
    pleasure_loop: str = Field(default="", description="Core pleasure-point loop pattern")
    tomato_adaptable: bool = Field(default=False, description="Whether suitable for Tomato (free) platform")

    @field_validator("tomato_adaptable", mode="before")
    @classmethod
    def coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "是", "1", "y")
        if isinstance(v, (int, float)):
            return bool(v)
        return False


class ScanReport(BaseModel):
    """Complete platform scan report (Step 1A or 1B)."""

    platform: str = Field(default="", description="Platform name: 飞卢 / 番茄")
    list_name: str = Field(default="", description="Name of the ranking list consulted")
    date: str = Field(default="", description="Date of scan (YYYY-MM-DD)")
    entries: list[ScanEntry] = Field(default_factory=list, description="Top 10-20 entries")
    summary: str = Field(default="", description="Overall trends and observations")
    scan_failed: bool = Field(default=False, description="True if scan could not access real data")

    @model_validator(mode="before")
    @classmethod
    def coerce_from_dict(cls, data: Any) -> Any:
        if isinstance(data, str):
            if data.strip() == "本次扫榜失败":
                return {"scan_failed": True, "summary": data}
            return {"summary": data}
        return data


# ============================================================
# Step 1C: Cross-Platform Analysis
# ============================================================


class CrossPlatformEntry(BaseModel):
    """A single genre/direction in the cross-platform comparison table."""

    genre: str = Field(default="", description="Genre / topic direction name")
    feilu_heat: bool = Field(default=False, description="是否有飞卢明显热度")
    tomato_audience: bool = Field(default=False, description="番茄是否有对应读者基础")
    feilu_appeal: str = Field(default="", description="飞卢吸引读者的点")
    tomato_adjustment: str = Field(default="", description="到番茄需要调整什么")
    risk: str = Field(default="", description="Risk points")
    recommendation: str = Field(default="medium", description="推荐等级: high / medium / low")

    @field_validator("feilu_heat", "tomato_audience", mode="before")
    @classmethod
    def coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "是", "1", "y")
        return False


class CrossPlatformReport(BaseModel):
    """Complete cross-platform comparison report (Step 1C)."""

    entries: list[CrossPlatformEntry] = Field(default_factory=list)
    selected_directions: list[str] = Field(
        default_factory=list,
        description="3-5 recommended directions with rationale",
    )


# ============================================================
# Step 2: Benchmark Book Skeleton Analysis
# ============================================================


class BenchmarkSkeleton(BaseModel):
    """Structural skeleton of a single benchmark book (Step 2)."""

    title: str = Field(default="", description="书名")
    genre: str = Field(default="", description="题材")
    title_appeal: str = Field(default="", description="书名吸量点")
    synopsis_promise: str = Field(default="", description="简介承诺的爽点")
    protagonist_opening: str = Field(default="", description="主角开局处境")
    golden_finger: str = Field(default="", description="金手指或核心底牌")
    opening_pressure: str = Field(default="", description="开篇压力")
    first_event: str = Field(default="", description="首个小事件如何启动")
    pleasure_loop: str = Field(default="", description="爽点循环")
    borrow_elements: list[str] = Field(default_factory=list, description="哪些元素值得借")
    replace_elements: list[str] = Field(default_factory=list, description="哪些元素必须换掉避免高仿")


class BenchmarkReport(BaseModel):
    """Complete benchmark skeleton analysis (Step 2)."""

    entries: list[BenchmarkSkeleton] = Field(default_factory=list)
    structural_summary: str = Field(default="", description="每个题材最值得借的结构骨架总结")


# ============================================================
# Step 3A: Candidate Topics
# ============================================================


class CandidateTopic(BaseModel):
    """A single candidate topic generated from research (Step 3A)."""

    genre_name: str = Field(default="", description="题材名称")
    one_line_setting: str = Field(default="", description="一句话设定")
    golden_finger: str = Field(default="", description="核心金手指")
    chapter1_conflict: str = Field(default="", description="第一章冲突")
    first_event_direction: str = Field(default="", description="首个小事件闭环方向")
    first_pleasure_wave: str = Field(default="", description="第一波爽点")
    tomato_fit: str = Field(default="medium", description="番茄适配度: high / medium / low")
    video_potential: str = Field(default="medium", description="视频表现力: high / medium / low")
    risk: str = Field(default="", description="风险点")


class CandidateTopicsOutput(BaseModel):
    """12 candidate topics output (Step 3A)."""

    topics: list[CandidateTopic] = Field(default_factory=list, description="12候选题材")


# ============================================================
# Step 3B: Topic Scoring
# ============================================================


class TopicScoreCard(BaseModel):
    """Score card for a single topic (Step 3B)."""

    genre_name: str = Field(default="", description="题材名称")
    score_title_appeal: float = Field(default=5.0, ge=0.0, le=10.0, description="书名吸量潜力")
    score_opening_pressure: float = Field(default=5.0, ge=0.0, le=10.0, description="第一章开篇压力")
    score_event_clarity: float = Field(default=5.0, ge=0.0, le=10.0, description="首个小事件闭环是否清楚")
    score_golden_finger: float = Field(default=5.0, ge=0.0, le=10.0, description="金手指是否能持续制造爽点")
    score_tomato_fit: float = Field(default=5.0, ge=0.0, le=10.0, description="番茄适配度")
    score_ai_stability: float = Field(default=5.0, ge=0.0, le=10.0, description="AI辅助稳定性")
    score_video_potential: float = Field(default=5.0, ge=0.0, le=10.0, description="视频表现力")
    score_breakdown_risk: float = Field(default=5.0, ge=0.0, le=10.0, description="吃书/崩设定风险（越低越好）")
    total_score: float = Field(default=0.0, description="综合得分（自动计算）")
    human_note: str = Field(default="", description="人工选择建议：每个题材适合什么取舍")


class TopicScoringReport(BaseModel):
    """Complete topic scoring report (Step 3B)."""

    scores: list[TopicScoreCard] = Field(default_factory=list)
    top_4: list[str] = Field(default_factory=list, description="推荐前4名题材名称")


# ============================================================
# Step 4A: Title & Synopsis
# ============================================================


class TitleSynopsisPair(BaseModel):
    """A single title + synopsis candidate (Step 4A)."""

    title: str = Field(default="", description="书名")
    synopsis: str = Field(default="", description="简介")
    score_clarity: float = Field(default=5.0, ge=0.0, le=10.0, description="一眼能否看懂")
    score_click_appeal: float = Field(default=5.0, ge=0.0, le=10.0, description="吸点击感")
    score_tomato_taste: float = Field(default=5.0, ge=0.0, le=10.0, description="番茄免费文口味")
    score_video_spread: float = Field(default=5.0, ge=0.0, le=10.0, description="视频传播性")
    score_ch1_conflict: float = Field(default=5.0, ge=0.0, le=10.0, description="是否容易引出第一章冲突")
    score_10chapter_viable: float = Field(default=5.0, ge=0.0, le=10.0, description="是否方便写出10章小事件闭环")


class TitleSynopsisReport(BaseModel):
    """Title & synopsis report for one topic (Step 4A)."""

    genre_name: str = Field(default="", description="题材名称")
    title_candidates: list[TitleSynopsisPair] = Field(default_factory=list, description="10个书名+3版简介候选")
    final_title: str = Field(default="", description="最终选定的书名")
    final_synopsis: str = Field(default="", description="最终选定的简介")


# ============================================================
# Step 5: Mini-Arc Outline (10-Chapter Small Event Loop)
# ============================================================


class MiniArcChapter(BaseModel):
    """A single chapter in the 10-chapter mini-arc outline (Step 5)."""

    chapter_number: int = Field(default=0, description="章节号 (1-10)")
    goal: str = Field(default="", description="本章目标")
    conflict: str = Field(default="", description="本章冲突")
    pleasure_point: str = Field(default="", description="本章爽点")
    new_info: str = Field(default="", description="新增信息")
    character_change: str = Field(default="", description="人物状态变化")
    foreshadowing: str = Field(default="", description="伏笔推进")
    ending_hook: str = Field(default="", description="结尾钩子")


class MiniArcOutline(BaseModel):
    """Complete 10-chapter mini-arc outline for one topic (Step 5)."""

    genre_name: str = Field(default="", description="题材名称")
    total_words: str = Field(default="20,000-21,000", description="预计总字数")
    chapters: list[MiniArcChapter] = Field(default_factory=list, description="10章大纲（每章2000-2100字）")
    arc_goal: str = Field(default="", description="小事件闭环总目标")
    next_arc_hook: str = Field(default="", description="下一事件的钩子")


# ============================================================
# Aggregated Pipeline State
# ============================================================


class TopicResearchState(BaseModel):
    """Aggregated state for the entire topic research pipeline (Steps 1-4).

    Stored in MainState.topic_research.
    """

    feilu_scan: Optional[ScanReport] = Field(default=None, description="飞卢扫榜结果")
    fanqie_scan: Optional[ScanReport] = Field(default=None, description="番茄扫榜结果")
    cross_platform: Optional[CrossPlatformReport] = Field(default=None, description="双榜交叉结论")
    benchmarks: Optional[BenchmarkReport] = Field(default=None, description="对标书骨架拆解")
    candidates: Optional[CandidateTopicsOutput] = Field(default=None, description="12候选题材")
    scores: Optional[TopicScoringReport] = Field(default=None, description="题材评分")
    title_synopsis: list[TitleSynopsisReport] = Field(
        default_factory=list,
        description="最终2个题材的书名和简介",
    )
    selected_genres: list[str] = Field(default_factory=list, description="人工选定的最终题材名称")
