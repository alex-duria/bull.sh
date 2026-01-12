"""
Understanding phase for Stampede agent architecture.

Extracts intent, entities, depth, and context from user queries.
Returns clarification questions when confidence is low.
"""

import json
from typing import Any

from anthropic import AsyncAnthropic

from bullsh.config import Config, get_config
from bullsh.logging import log

from .schemas import (
    ExportIntent,
    InferredDepth,
    Understanding,
)

UNDERSTANDING_PROMPT = """You are analyzing a user's investment research query to extract structured information.

Your job is to understand what the user is asking and extract:
1. **Intent**: What type of research are they requesting?
2. **Tickers**: Stock symbols mentioned (uppercase, 1-5 letters)
   - IMPORTANT: Always resolve company names to their actual ticker symbols!
   - Examples: "Tesla" → "TSLA", "Apple" → "AAPL", "Microsoft" → "MSFT", "Google" → "GOOGL", "Amazon" → "AMZN"
   - If user writes "TESLA", return "TSLA" (the actual ticker, not the company name)
3. **Depth**: How thorough should the analysis be?
4. **Focus areas**: What metrics or aspects do they care about?
5. **Export intent**: Do they want to export results?

## Intent Types
- "research": General company research, financials, fundamentals
- "valuation": Price targets, fair value, buy/sell analysis
- "quick_lookup": Simple fact (P/E ratio, price, single metric)
- "thesis": Investment thesis generation
- "sentiment": Social sentiment, crowd opinion
- "factors": Quantitative factor analysis (value, momentum, etc.)
- "comparison": Compare multiple companies (handled separately)

## Depth Inference Rules
- **quick**: Single data point questions ("What's NVDA's P/E?", "AAPL price?")
- **standard**: Normal research ("Tell me about NVDA", "Research AMD")
- **deep**: Thorough analysis ("Full thesis on NVDA", "Deep dive into AMD's financials", "comprehensive analysis")

## Export Detection
Look for keywords like "excel", "export", "spreadsheet", "pdf", "save"

## Session Context
Consider what has already been researched in this session:
{session_context}

## Confidence Scoring
- 1.0: Clear ticker + clear intent + all info present
- 0.8-0.99: Minor ambiguity but can proceed
- 0.5-0.79: Significant ambiguity, should ask clarification
- <0.5: Very unclear, definitely ask clarification

If confidence < 0.8, provide a clarification_question.

## Examples

Query: "What's NVDA's P/E ratio?"
→ intent: "quick_lookup", tickers: ["NVDA"], inferred_depth: "quick", confidence: 1.0

Query: "Tell me about Apple"
→ intent: "research", tickers: ["AAPL"], inferred_depth: "standard", confidence: 0.95
(Note: "Apple" resolved to ticker "AAPL")

Query: "Research Tesla"
→ intent: "research", tickers: ["TSLA"], inferred_depth: "standard", confidence: 0.95
(Note: "Tesla" resolved to ticker "TSLA", NOT "TESLA")

Query: "Deep dive into semiconductor companies"
→ intent: "research", tickers: [], inferred_depth: "deep", confidence: 0.6
→ clarification_question: "Which semiconductor companies would you like me to research? (e.g., NVDA, AMD, INTC)"

Query: "Research NVDA and export to excel"
→ intent: "research", tickers: ["NVDA"], inferred_depth: "standard", export_intent: "excel", confidence: 1.0

Now analyze this query:
{query}

Respond with a JSON object matching the Understanding schema."""


UNDERSTANDING_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "description": "Primary intent type",
        },
        "tickers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Stock ticker symbols",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in understanding (0-1)",
        },
        "inferred_depth": {
            "type": "string",
            "enum": ["quick", "standard", "deep"],
            "description": "Inferred research depth",
        },
        "timeframe": {
            "type": "string",
            "description": "Timeframe mentioned (null if not specified)",
        },
        "metrics_focus": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific metrics to focus on",
        },
        "wants_sentiment": {
            "type": "boolean",
            "description": "User wants social sentiment",
        },
        "wants_factors": {
            "type": "boolean",
            "description": "User wants factor analysis",
        },
        "export_intent": {
            "type": "string",
            "enum": ["none", "excel", "pdf", "docx"],
            "description": "Export intention",
        },
        "clarification_question": {
            "type": "string",
            "description": "Question to ask if confidence < 0.8",
        },
    },
    "required": ["intent", "tickers", "confidence", "inferred_depth"],
}


class UnderstandingAgent:
    """
    Agent that extracts structured understanding from user queries.

    Uses Claude to analyze queries and return Understanding objects.
    If confidence is low, returns a clarification question.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.client = AsyncAnthropic(api_key=self.config.anthropic_api_key)

    async def understand(
        self,
        query: str,
        prior_tickers: list[str] | None = None,
        prior_frameworks: list[str] | None = None,
    ) -> Understanding:
        """
        Analyze a user query and extract structured understanding.

        Args:
            query: The user's query
            prior_tickers: Tickers already researched in this session
            prior_frameworks: Frameworks already used in this session

        Returns:
            Understanding object with extracted information
        """
        log("stampede", f"Understanding query: {query[:100]}...")

        # Build session context
        session_context = self._build_session_context(
            prior_tickers or [],
            prior_frameworks or [],
        )

        # Build the prompt
        prompt = UNDERSTANDING_PROMPT.format(
            session_context=session_context,
            query=query,
        )

        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=1024,
                system="You are a query analysis assistant. Respond only with valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract JSON from response
            response_text = response.content[0].text
            understanding_dict = self._parse_json_response(response_text)

            # Build Understanding object
            understanding = self._build_understanding(
                understanding_dict,
                prior_tickers or [],
                prior_frameworks or [],
            )

            log(
                "stampede",
                f"Understanding: intent={understanding.intent}, "
                f"tickers={understanding.tickers}, "
                f"depth={understanding.inferred_depth.value}, "
                f"confidence={understanding.confidence:.2f}",
            )

            return understanding

        except Exception as e:
            log("stampede", f"Understanding error: {e}", level="error")
            # Return a low-confidence understanding that will trigger clarification
            return Understanding(
                intent="unknown",
                tickers=[],
                confidence=0.3,
                inferred_depth=InferredDepth.STANDARD,
                clarification_question="I had trouble understanding your request. Could you please rephrase it?",
            )

    def _build_session_context(
        self,
        prior_tickers: list[str],
        prior_frameworks: list[str],
    ) -> str:
        """Build session context string for the prompt."""
        parts = []

        if prior_tickers:
            parts.append(f"Previously researched tickers: {', '.join(prior_tickers)}")

        if prior_frameworks:
            parts.append(f"Previously used frameworks: {', '.join(prior_frameworks)}")

        if not parts:
            return "No prior research in this session."

        return "\n".join(parts)

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse JSON from Claude's response, handling markdown code blocks."""
        text = response_text.strip()

        # Try to extract from markdown code block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        return json.loads(text)

    def _build_understanding(
        self,
        data: dict[str, Any],
        prior_tickers: list[str],
        prior_frameworks: list[str],
    ) -> Understanding:
        """Build Understanding object from parsed JSON."""
        # Parse inferred depth
        depth_str = data.get("inferred_depth", "standard")
        try:
            inferred_depth = InferredDepth(depth_str)
        except ValueError:
            inferred_depth = InferredDepth.STANDARD

        # Parse export intent
        export_str = data.get("export_intent", "none")
        try:
            export_intent = ExportIntent(export_str)
        except ValueError:
            export_intent = ExportIntent.NONE

        return Understanding(
            intent=data.get("intent", "research"),
            tickers=data.get("tickers", []),
            confidence=float(data.get("confidence", 0.5)),
            inferred_depth=inferred_depth,
            timeframe=data.get("timeframe"),
            metrics_focus=data.get("metrics_focus", []),
            wants_sentiment=data.get("wants_sentiment", False),
            wants_factors=data.get("wants_factors", False),
            export_intent=export_intent,
            prior_tickers_researched=prior_tickers,
            prior_frameworks_used=prior_frameworks,
            clarification_question=data.get("clarification_question"),
        )


async def quick_understand(
    query: str,
    config: Config | None = None,
) -> Understanding:
    """
    Convenience function for quick understanding extraction.

    Args:
        query: The user's query
        config: Optional config override

    Returns:
        Understanding object
    """
    agent = UnderstandingAgent(config)
    return await agent.understand(query)
