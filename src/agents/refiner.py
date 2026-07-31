"""Refiner Agent — polishes chapter drafts based on review feedback."""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from ..llm.scheduler import ModelScheduler
from ..models.bible import NovelBible
from ..models.chapter import ChapterDraft, PolishedChapter
from ..models.review import ReviewReport
from .base import BaseAgent

logger = logging.getLogger(__name__)


class PolishedOutput(BaseModel):
    """LLM output for polished chapter."""
    title: str = Field(default="")
    content: str = Field(..., description="润色后的章节正文")
    revision_notes: str = Field(default="", description="修改说明")


class RefinerAgent(BaseAgent):
    """Polishes and revises chapter drafts.

    Handles:
    - Language rhythm and flow improvements
    - Duplicate sentence pattern removal
    - Cliché expression elimination
    - Dialogue naturalization
    - Chapter-end hook strengthening
    - AI-flavor removal
    """

    agent_type = "refiner"

    async def polish_chapter(
        self,
        draft: ChapterDraft,
        review: ReviewReport,
        bible: NovelBible,
        human_feedback: Optional[str] = None,
    ) -> PolishedChapter:
        """Polish a chapter draft based on review feedback.

        Args:
            draft: The chapter draft to polish.
            review: Review report with issues and suggestions.
            bible: Novel bible with style contract.
            human_feedback: Optional additional feedback from the human author.

        Returns:
            PolishedChapter with the refined content.
        """
        logger.info(f"Polishing chapter {draft.chapter_number}...")

        style = bible.style_contract

        system = self.build_system_prompt(
            role="文字打磨师",
            expertise="你是一位追求极致的文字编辑。你能让一段普通的文字变得精准、有力、"
                      "有韵味。你擅长调整句式节奏、删除冗余、增强画面感、让对话更自然。"
                      "你对AI生成的套路表达有天然的敏感。",
            constraints=f"""文风要求：
- 语调：{style.tone}
- 句式风格：{style.sentence_style}
- 禁用表达：{style.forbidden_phrases}
- 推荐技法：{style.preferred_techniques}""",
        )

        # Compile issues into actionable feedback
        issues_text = self._format_issues(review)
        suggestions_text = "\n".join(f"- {s}" for s in review.suggestions)

        user = f"""请对以下章节进行润色和修改：

【审阅报告】
总分：{review.overall_score}/10
各维度：{review.dimension_scores}

【需要修改的问题】
{issues_text}

【改进建议】
{suggestions_text}

【原文章节标题】{draft.title}

【原文章节正文】
{draft.content}

修改重点：
1. **句式变化**：消除重复句式，调整长短句比例
2. **删AI味**：删除「综上所述」「值得注意的是」「在...的过程中」等机械表达
3. **对话优化**：让每句对话更符合角色性格，删除过于功能化的台词
4. **画面感**：增强关键场景的感官细节（视觉、听觉、触觉）
5. **章末钩子**：确保结尾有足够的吸引力
6. **节奏**：调整场景推进速度，避免均分笔墨"""

        if human_feedback:
            user += f"\n\n【作者反馈】\n{human_feedback}"

        result = await self.generate_structured(
            system_prompt=system,
            user_prompt=user,
            response_model=PolishedOutput,
            temperature_override=0.4,
            max_tokens_override=8192,
        )

        word_count = len(result.content.replace(" ", "").replace("\n", ""))

        polished = PolishedChapter(
            chapter_number=draft.chapter_number,
            title=result.title or draft.title,
            content=result.content,
            word_count=word_count,
            revision_notes=result.revision_notes,
        )

        logger.info(f"Chapter {draft.chapter_number} polished: {word_count} chars")
        return polished

    def _format_issues(self, review: ReviewReport) -> str:
        """Format review issues for the polish prompt."""
        if not review.issues:
            return "（无特定问题）"

        by_severity = {"critical": [], "major": [], "minor": [], "suggestion": []}
        for issue in review.issues:
            sev = issue.severity if isinstance(issue.severity, str) else issue.severity.value
            by_severity.setdefault(sev, []).append(issue)

        lines = []
        for sev in ["critical", "major", "minor", "suggestion"]:
            items = by_severity.get(sev, [])
            if items:
                lines.append(f"\n### {sev.upper()} ({len(items)}个)")
                for item in items:
                    lines.append(f"- [{item.category}] {item.description}")
                    if item.suggestion:
                        lines.append(f"  建议: {item.suggestion}")

        return "\n".join(lines)
