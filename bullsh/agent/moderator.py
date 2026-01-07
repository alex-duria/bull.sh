"""Moderator agent for debate synthesis."""

from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from bullsh.agent.base import AgentResult
from bullsh.config import Config, get_config
from bullsh.logging import log


@dataclass
class SynthesisResult:
    """Structured result from moderator synthesis."""
    contentions: list[str]  # 2-5 key points of disagreement
    conviction: int  # 1-10 scale
    conviction_direction: str  # "bull" or "bear"
    conviction_reasoning: str  # Why this score
    thesis_breaker: str  # Conditional: "thesis breaks if..."
    evidence_summary: str  # Which evidence buckets favor each side
    full_synthesis: str  # Complete moderator output
    consensus_points: list[str]  # Points both agents agreed on
    factual_discrepancies: list[str]  # Conflicting facts noted


class ModeratorAgent:
    """
    Neutral moderator that synthesizes bull vs. bear debate.

    Unlike other agents, the moderator:
    - Has NO tools (synthesis only)
    - Receives only arguments, not raw data
    - Must remain impartial
    - Cannot be coached by the user
    """

    def __init__(
        self,
        config: Config | None = None,
        ticker: str | None = None,
    ):
        self.config = config or get_config()
        self.ticker = ticker
        self.client = AsyncAnthropic(api_key=self.config.anthropic_api_key)

    @property
    def system_prompt(self) -> str:
        return f"""You are a neutral moderator synthesizing a bull vs. bear debate on {self.ticker or 'a stock'}.

You receive the opening arguments and rebuttals from both sides.
You do NOT have access to raw data - judge based on arguments presented.

Your job:

1. IDENTIFY KEY CONTENTIONS (2-5 points)
   Find points where bull and bear fundamentally disagree.
   Format: "**[Topic]**: Bull says X; Bear says Y"

2. NOTE CONSENSUS (if any)
   If both agents agree on something, note it as consensus, not contention.

3. CHECK FOR FACTUAL DISCREPANCIES
   If bull and bear cite different numbers for the same metric, flag it:
   "Note: Factual discrepancy - Bull cited X%, Bear cited Y%"

4. SUMMARIZE EVIDENCE BUCKETS
   "Most [financial/growth/valuation] metrics favor [bull/bear]"

5. PRODUCE CONVICTION SCORE
   Scale:
   - 1-3: Strong Bear
   - 4: Lean Bear
   - 5: Neutral
   - 6: Lean Bull
   - 7-10: Strong Bull

   Format your score as:
   "**Conviction: X/10 [LEAN/STRONG] [BULL/BEAR]**

   X/10 because [reasoning]. Would be Y/10 if [condition]."

6. SURFACE THE THESIS-BREAKER
   Use conditional framing for the winning side:
   "**[Bull/Bear] thesis breaks if:** [specific condition]"

Be impartial. Weight evidence over rhetoric. The score should reflect
what percentage of the presented evidence supports each side.

OUTPUT FORMAT:
Start with contentions, then consensus (if any), then evidence summary,
then conviction with reasoning, then thesis-breaker."""

    async def synthesize(
        self,
        bull_opening: str,
        bear_opening: str,
        bull_rebuttals: list[str],
        bear_rebuttals: list[str],
    ) -> AgentResult:
        """
        Synthesize the debate into a verdict.

        Args:
            bull_opening: Bull's opening argument
            bear_opening: Bear's opening argument
            bull_rebuttals: List of bull's rebuttals
            bear_rebuttals: List of bear's rebuttals

        Returns:
            AgentResult with moderator's synthesis
        """
        log("agent", f"ModeratorAgent synthesizing debate for {self.ticker}")

        # Build the debate transcript
        bull_rebuttals_text = "\n\n".join(bull_rebuttals) if bull_rebuttals else "(none)"
        bear_rebuttals_text = "\n\n".join(bear_rebuttals) if bear_rebuttals else "(none)"

        task = f"""BULL OPENING ARGUMENT:
{bull_opening}

BULL REBUTTALS:
{bull_rebuttals_text}

---

BEAR OPENING ARGUMENT:
{bear_opening}

BEAR REBUTTALS:
{bear_rebuttals_text}

---

Synthesize this debate. Identify contentions, assess evidence, and provide your conviction score with reasoning."""

        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=2048,
                system=self.system_prompt,
                messages=[{"role": "user", "content": task}],
            )

            text_content = "".join(
                block.text for block in response.content if block.type == "text"
            )
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            log("agent", f"ModeratorAgent completed synthesis")

            return AgentResult(
                content=text_content,
                tool_results=[],
                ticker=self.ticker,
                success=True,
                tokens_used=tokens_used,
            )

        except Exception as e:
            log("agent", f"ModeratorAgent error: {e}", level="error")
            return AgentResult(
                content="",
                ticker=self.ticker,
                success=False,
                error=str(e),
            )

    def parse_synthesis(self, synthesis_text: str) -> SynthesisResult:
        """
        Parse the moderator's synthesis into structured data.

        This is best-effort parsing - the synthesis text is authoritative.
        """
        import re

        contentions: list[str] = []
        consensus_points: list[str] = []
        factual_discrepancies: list[str] = []
        conviction = 5
        conviction_direction = "neutral"
        conviction_reasoning = ""
        thesis_breaker = ""
        evidence_summary = ""

        # Extract conviction score
        conviction_match = re.search(
            r"\*?\*?Conviction:?\*?\*?\s*(\d+)/10\s*(?:\*?\*?)?\s*(LEAN|STRONG)?\s*(BULL|BEAR)?",
            synthesis_text,
            re.IGNORECASE,
        )
        if conviction_match:
            conviction = int(conviction_match.group(1))
            direction = conviction_match.group(3)
            if direction:
                conviction_direction = direction.lower()
            elif conviction > 5:
                conviction_direction = "bull"
            elif conviction < 5:
                conviction_direction = "bear"

        # Extract reasoning (text after conviction score)
        reasoning_match = re.search(
            r"(\d+)/10 because (.+?)(?:\n\n|\*\*|$)",
            synthesis_text,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            conviction_reasoning = reasoning_match.group(2).strip()

        # Extract thesis breaker
        breaker_match = re.search(
            r"thesis breaks if:?\s*(.+?)(?:\n\n|\*\*|$)",
            synthesis_text,
            re.IGNORECASE,
        )
        if breaker_match:
            thesis_breaker = breaker_match.group(1).strip()

        # Extract contentions (lines with "Bull says" and "Bear says")
        contention_pattern = re.compile(
            r"\*?\*?\[?([^\]]+?)\]?\*?\*?:?\s*Bull says?.+?;?\s*Bear says?",
            re.IGNORECASE,
        )
        for match in contention_pattern.finditer(synthesis_text):
            contentions.append(match.group(0))

        # Extract evidence summary
        evidence_match = re.search(
            r"(Most .+? metrics favor .+?)(?:\n|$)",
            synthesis_text,
            re.IGNORECASE,
        )
        if evidence_match:
            evidence_summary = evidence_match.group(1).strip()

        return SynthesisResult(
            contentions=contentions[:5],  # Max 5
            conviction=conviction,
            conviction_direction=conviction_direction,
            conviction_reasoning=conviction_reasoning,
            thesis_breaker=thesis_breaker,
            evidence_summary=evidence_summary,
            full_synthesis=synthesis_text,
            consensus_points=consensus_points,
            factual_discrepancies=factual_discrepancies,
        )
