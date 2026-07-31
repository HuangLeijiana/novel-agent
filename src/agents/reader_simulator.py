"""Reader Simulator Agent — simulates reader experience and emotional response."""

import logging

from pydantic import BaseModel, Field

from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft
from ..models.review import Issue
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReaderFeedbackOutput(BaseModel):
    """LLM output for reader simulation."""

    engagement_score: float = Field(default=7.0, ge=0.0, le=10.0, description="沉浸感评分")
    emotional_impact: str = Field(default="", description="情感冲击描述")
    boring_sections: list[str] = Field(default_factory=list, description="可能让读者感到无聊的段落")
    exciting_sections: list[str] = Field(default_factory=list, description="让读者兴奋的段落")
    continuation_likelihood: float = Field(
        default=7.0,
        ge=0.0,
        le=10.0,
        description="读者想继续读下一章的可能性",
    )
    reader_questions: list[str] = Field(
        default_factory=list,
        description="读者在阅读时可能会产生的问题",
    )
    issues: list[Issue] = Field(default_factory=list, description="从读者视角发现的问题")


class ReaderSimulatorAgent(BaseAgent):
    """Simulates a target reader's experience to provide reader-centric feedback.

    This agent is distinct from the Editor — it doesn't judge technical quality,
    it judges reading enjoyment. It answers: "Would the target reader enjoy this?"
    """

    agent_type = "reader_simulator"

    async def simulate_reading(
        self,
        draft: ChapterDraft,
        bible: NovelBible,
        target_reader: str = "",
    ) -> dict:
        """Simulate a target reader reading this chapter.

        Returns a dict with engagement metrics and reader-centric feedback.
        """
        system = self.build_system_prompt(
            role="读者体验模拟器",
            expertise=f"""你能够完全代入{target_reader or "目标读者"}的视角来阅读小说。
你像真正的读者一样：会被钩子吸引、会在无聊处走神、会对角色产生情感投射、
会在反转处感到震撼。你不会用编辑的眼光分析——你用的是读者的心。""",
        )

        user = f"""请以目标读者的身份阅读以下章节：

【读者画像】{target_reader or "普通小说读者"}
【文风】{bible.style_contract.tone}
【题材】{bible.world.world_type}

【章节正文】
{draft.content[:1500]}{"...(因篇幅截断)" if len(draft.content) > 1500 else ""}

请以读者视角回答：
1. **沉浸感评分**（0-10）：你有多沉浸在故事中？
2. **情感冲击**：读完有什么感觉？兴奋？感动？期待？平淡？
3. **无聊段落**：有没有让你想跳过的部分？
4. **兴奋段落**：哪些部分让你读得最投入？
5. **继续阅读意愿**（0-10）：多想立刻读下一章？
6. **读者疑问**：你在阅读中产生了什么困惑或好奇？
7. **章末钩子效果**：结尾让你多想看下一章？"""

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=ReaderFeedbackOutput,
            temperature_override=0.5,
        )

        return {
            "engagement_score": result.engagement_score,
            "emotional_impact": result.emotional_impact,
            "boring_sections": result.boring_sections,
            "exciting_sections": result.exciting_sections,
            "continuation_likelihood": result.continuation_likelihood,
            "reader_questions": result.reader_questions,
            "issues": [i for i in result.issues if i.category not in ("consistency", "style")],
        }
