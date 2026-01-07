# Bull vs. Bear Agent Debate - Feature Specification

## Overview

Spawn two adversarial agents—one arguing the bull case, one arguing the bear case—that critique each other's assumptions. A moderator agent then synthesizes a balanced thesis, surfacing the strongest counterarguments automatically.

---

## Design Decisions

| Area | Decision |
|------|----------|
| **Modes** | Quick (default, 1 rebuttal) and Deep (`--deep`, 2 rebuttals) |
| **Streaming** | Real-time token-by-token for each agent |
| **User coaching** | Optional hints, queued until phase pause, unlimited |
| **Moderator coaching** | Not allowed (must remain impartial) |
| **Concessions** | Allowed - agents can acknowledge opponent's strong points |
| **Lopsided debates** | Force balance - even weak cases present 3-5 points |
| **Fact disputes** | Moderator requests agents to reconcile before synthesizing |
| **Score meaning** | Data-weight verdict (e.g., "7/10 = 70% of evidence supports bull") |
| **Score presentation** | Show reasoning ("6/10 because X outweighs Y, would be 8 if Z confirmed") |
| **Post-debate** | Follow-ups allowed to any agent |
| **Session saving** | Yes, saved like research sessions |
| **Repeat debates** | Use cached tool data (same day) |
| **Multi-ticker** | Single ticker only (use /compare for multi-stock) |
| **Thin data** | Refuse to debate if insufficient SEC filings |
| **Config** | CLI flags only (no config.toml) |

---

## User Experience

### Entry Points

```bash
# CLI commands
bullsh debate NVDA                    # Quick mode (default)
bullsh debate NVDA --deep             # Deep mode (2 rebuttal rounds)
bullsh debate NVDA --framework piotroski  # With framework context

# REPL commands
/debate NVDA
/debate NVDA --deep

# Natural language triggers
"Give me both sides of NVDA"
"What's the bull and bear case for Tesla?"
"Debate AMD"
```

### Output Structure

```
🐂 BULL CASE
────────────────
• Data center dominance with 80%+ GPU market share (10-K 2024)
• AI training demand has multi-year runway
• CUDA ecosystem creates high switching costs

🐻 BEAR CASE
────────────────
• Valuation assumes perfect execution at 70x P/E (Yahoo Finance)
• Customer concentration risk - top 5 hyperscalers = 40% revenue
• AMD/Intel competition intensifying with MI300X

⚔️  KEY CONTENTIONS (3)
────────────────
1. **Moat durability**: Bull cites CUDA lock-in; Bear cites ROCm progress
2. **Valuation**: Bull says growth justifies; Bear says priced for perfection
3. **Competition**: Bull minimizes AMD threat; Bear sees share erosion

⚖️  MODERATOR SYNTHESIS
────────────────
**Conviction: 6/10 LEAN BULL**

Most financial metrics favor bull (revenue growth, margins, cash flow).
Valuation metrics favor bear (P/E, EV/EBITDA vs. sector).

6/10 because growth execution has been exceptional, but would be 8/10
if competition concerns prove unfounded in next earnings.

**Thesis breaks if:** AMD achieves training performance parity at lower price point.
```

---

## Architecture

### Agent Roles

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                            │
│  _detect_debate() → Spawns DebateCoordinator                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DEBATE COORDINATOR                         │
│  Manages 4-phase flow, context passing, state persistence   │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │   BULL    │        │   BEAR    │        │ MODERATOR │
    │   AGENT   │◄──────►│   AGENT   │───────►│   AGENT   │
    └───────────┘        └───────────┘        └───────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    [Opening]            [Opening]            [Synthesis]
    [Rebuttals]          [Rebuttals]          [Conviction]
    [Citations]          [Citations]          [Reasoning]
```

### Debate Flow

```
Phase 1: RESEARCH (parallel)
├── Bull Agent: Gather data (5 tool calls max)
├── Bear Agent: Gather data (5 tool calls max, cache hits from Bull)
└── Tool status shown: "◐ Fetching SEC filing..."

Phase 2: OPENING ARGUMENTS (parallel, streamed)
├── Bull Agent: Present 3-5 strongest bull points with inline citations
├── Bear Agent: Present 3-5 strongest bear points with inline citations
└── [User hint injection point - queued until phase complete]

Phase 3: REBUTTAL (sequential, streamed)
├── Bull Agent: Direct-quote bear's points, then counter
├── Bear Agent: Direct-quote bull's points, then counter
├── [User hint injection point]
└── [Deep mode: Round 2 rebuttals]

Phase 4: SYNTHESIS (streamed)
├── Identify 2-5 key contentions
├── Summarize evidence buckets
├── Score with reasoning
└── Surface conditional thesis-breaker
```

### State Persistence

```python
@dataclass
class DebateState:
    """Persisted for resume and session saving."""
    ticker: str
    phase: DebatePhase  # RESEARCH, OPENING, REBUTTAL, SYNTHESIS, COMPLETE
    deep_mode: bool
    framework: str | None

    # Phase outputs
    bull_research: dict[str, ToolResult] | None
    bear_research: dict[str, ToolResult] | None
    bull_opening: str | None
    bear_opening: str | None
    bull_rebuttals: list[str]
    bear_rebuttals: list[str]
    user_hints: list[tuple[str, str]]  # (target_agent, hint)

    # Synthesis
    contentions: list[str] | None
    conviction: int | None  # 1-10
    conviction_reasoning: str | None
    thesis_breaker: str | None

    # Metadata
    started_at: datetime
    completed_at: datetime | None
    tokens_used: int
```

---

## Agent Specifications

### BullAgent

**System Prompt:**
```
You are an investment analyst arguing the BULL CASE for {ticker}.
Your job is to find the strongest reasons to be optimistic.

Focus on:
- Competitive advantages and moats
- Growth catalysts and tailwinds
- Management quality and execution
- Underappreciated strengths
- Why bears are wrong

RULES:
- You MAY concede valid points ("I acknowledge X, but...")
- You MUST cite sources inline: "Revenue grew 40% (10-K 2024)"
- You MUST present 3-5 points even if the case is weak
- Be rigorous but advocatory

{framework_context if active}

You will receive the bear's arguments. When rebutting:
- DIRECTLY QUOTE the bear's point you're addressing
- Then explain why it's overstated or incorrect
```

**Tools:** sec_search, sec_fetch, rag_search, scrape_yahoo, compute_ratios, search_news, web_search

**Iteration Limit:** 5 tool calls

### BearAgent

**System Prompt:**
```
You are an investment analyst arguing the BEAR CASE for {ticker}.
Your job is to find the strongest reasons to be cautious.

Focus on:
- Valuation concerns and downside scenarios
- Competitive threats and disruption risk
- Execution risks and management issues
- Macro/regulatory headwinds
- Why bulls are wrong

RULES:
- You MAY concede valid points ("I acknowledge X, but...")
- You MUST cite sources inline: "P/E of 70x (Yahoo Finance)"
- You MUST present 3-5 points even if the case is weak
- Be rigorous but skeptical

{framework_context if active}

You will receive the bull's arguments. When rebutting:
- DIRECTLY QUOTE the bull's point you're addressing
- Then explain why it's overstated or incorrect
```

**Tools:** Same as BullAgent

**Iteration Limit:** 5 tool calls

### ModeratorAgent

**System Prompt:**
```
You are a neutral moderator synthesizing a bull vs. bear debate on {ticker}.

You receive the opening arguments and rebuttals from both sides.
You do NOT have access to raw data - judge based on arguments presented.

Your job:
1. Identify 2-5 KEY CONTENTIONS - points of fundamental disagreement
2. Summarize which evidence buckets favor each side
3. Produce a CONVICTION SCORE with reasoning:
   - 1-3: Strong Bear
   - 4: Lean Bear
   - 5: Neutral
   - 6: Lean Bull
   - 7-10: Strong Bull

   Format: "X/10 because [reasoning]. Would be Y/10 if [condition]."

4. Surface the THESIS-BREAKER using conditional framing:
   "Bull thesis breaks if: [specific condition]"
   OR
   "Bear thesis breaks if: [specific condition]"

If agents AGREE on a point, note it as consensus rather than contention.

If agents cite conflicting facts, note the discrepancy and request
clarification before finalizing your synthesis.

Be impartial. Weight evidence over rhetoric.
```

**Tools:** None (synthesis only)

**Iteration Limit:** 1 (single pass)

---

## Data Flow & Context

### Shared Research Cache

Both agents query same tools. HTTP cache ensures identical data:

```python
# Bull agent calls first
await scrape_yahoo("NVDA")  # Fetches fresh

# Bear agent calls (parallel or after)
await scrape_yahoo("NVDA")  # Cache hit - identical data guaranteed
```

### Framework Integration

When framework is active (via session or `--framework` flag):

```python
def _build_agent_context(self, agent_type: str) -> str:
    context = f"Present your {agent_type} case for {self.ticker}."

    if self.framework:
        summary = self.session.get_framework_summary()
        context += f"""

Framework Analysis ({self.framework}):
{summary}

This framework data is available to both agents. You may reference it
to support your argument. If framework findings contradict your position,
you MUST reconcile (explain why despite the data, your thesis holds).
"""
    return context
```

### Context Passing Between Phases

```python
# Phase 2: Opening
bull_prompt = f"""
Research data summary:
{self._summarize_tool_results(bull_research)}

{self._build_agent_context("bull")}
"""

# Phase 3: Rebuttals
bull_rebuttal_prompt = f"""
Your opening argument:
{bull_opening}

Bear's opening argument:
{bear_opening}

{user_hints_for_bull if any}

Rebut the bear's weakest points. DIRECTLY QUOTE each point before countering.
"""

# Phase 4: Synthesis
moderator_prompt = f"""
BULL OPENING:
{bull_opening}

BULL REBUTTALS:
{bull_rebuttals}

BEAR OPENING:
{bear_opening}

BEAR REBUTTALS:
{bear_rebuttals}

Synthesize and score.
"""
```

---

## User Coaching

### Hint Injection Flow

```
User types: "Bull, mention their new AI chip"

System:
1. Detect non-command input during debate
2. Parse target agent ("Bull")
3. Queue hint for next phase
4. Display: "💡 Hint queued for Bull"
5. At next phase start, inject into prompt:

   USER HINT: Also mention their new AI chip
```

### Hint Timing

- Hints typed mid-stream are **queued until current agent finishes**
- Hints apply to the **next phase** the target agent participates in
- Opponent **naturally addresses** new points in subsequent rebuttals

### Hint Limits

- **Unlimited** - user's money, user's debate
- Each hint adds ~200-500 tokens
- Token warnings still apply at 80% threshold

---

## Interruption & Resume

### State Checkpoints

State saved after each phase completes:

```python
async def _run_phase(self, phase: DebatePhase) -> None:
    # ... execute phase ...

    self.state.phase = phase
    await self._save_state()  # Checkpoint
```

### Resume Flow

```
User: /debate NVDA
[Ctrl+C during Bear's opening]

[Later]
User: /debate NVDA
System: "Resume debate from Phase 2 (Bear Opening)? [Y/n]"
User: Y
[Continues from last checkpoint]
```

---

## Edge Cases

### Thin Data Stocks

```python
async def _validate_ticker(self, ticker: str) -> bool:
    """Refuse debate if insufficient data."""
    filings = await sec.sec_search(ticker)

    if not filings.data.get("10-K"):
        raise DebateRefused(
            f"Cannot debate {ticker}: No 10-K filings found. "
            "Debates require at least one annual filing. "
            "Try /research for preliminary analysis."
        )
    return True
```

### Agent Agreement (Consensus)

If agents agree on a point, moderator handles:

```
⚖️  MODERATOR SYNTHESIS

**Unusual consensus detected**

Both agents agree:
- Competition is intensifying (cited by both Bull and Bear)
- Management execution has been strong (acknowledged by Bear)

Key remaining uncertainty:
- Valuation sustainability given growth deceleration
```

### Factual Discrepancies

If Bull says "40% growth" and Bear says "35% growth":

```
⚖️  MODERATOR SYNTHESIS

**Note:** Factual discrepancy detected.
Bull cited 40% revenue growth; Bear cited 35%.
Per 10-K, actual figure is 38% YoY.

[Synthesis continues with corrected figure]
```

---

## CLI Integration

```python
@app.command()
def debate(
    ticker: Annotated[str, typer.Argument(help="Stock ticker to debate")],
    deep: Annotated[bool, typer.Option("--deep", help="Two rebuttal rounds")] = False,
    framework: Annotated[Optional[str], typer.Option(help="Framework context")] = None,
) -> None:
    """Run adversarial bull vs. bear debate on a stock."""
    ...
```

### REPL Integration

```python
# Slash command
"/debate NVDA"
"/debate NVDA --deep"
"/debate NVDA --framework piotroski"

# Natural language detection
def _detect_debate(self, message: str) -> tuple[str, bool] | None:
    """Returns (ticker, deep_mode) if debate detected."""
    patterns = [
        r"debate\s+([A-Z]{1,5})",
        r"bull\s+(?:and|vs\.?|versus)\s+bear\s+(?:for\s+)?([A-Z]{1,5})",
        r"both\s+sides\s+(?:of\s+)?([A-Z]{1,5})",
        r"(?:give me|what's|what are)\s+(?:the\s+)?(?:bull|bear)\s+(?:and\s+(?:bull|bear)\s+)?case\s+(?:for\s+)?([A-Z]{1,5})",
    ]
    ...
```

---

## Export

### Formats

```bash
/export debate.md              # Markdown (default)
/export debate.pdf             # PDF
/export debate.md --full       # Full transcript
/export debate.md --summary    # Summary only
```

### Summary Export Structure

```markdown
# NVDA Bull vs. Bear Debate
*2026-01-06 | Quick Mode | Piotroski Framework*

## Bull Case (3 points)
1. Data center dominance...
2. AI training runway...
3. CUDA ecosystem...

## Bear Case (3 points)
1. Valuation risk...
2. Customer concentration...
3. Competition...

## Key Contentions
1. Moat durability
2. Valuation justification
3. Competitive threat severity

## Verdict
**6/10 LEAN BULL**
Most financial metrics favor bull. Would be 8/10 if competition threat diminishes.

**Thesis breaks if:** AMD achieves training parity at lower price.
```

### Full Export Structure

Same as summary, plus:
- Complete opening arguments
- All rebuttals
- User hints injected
- Tool citations appendix

---

## Token Budget

| Mode | Phases | Est. Tokens | Est. Cost |
|------|--------|-------------|-----------|
| Quick | 4 (1 rebuttal) | ~25K | $0.12-0.20 |
| Deep | 5 (2 rebuttals) | ~40K | $0.20-0.35 |

With prompt caching: 30-40% savings on repeated system prompts.

---

## Files to Create

```
bullsh/agent/
├── debate.py           # DebateCoordinator, DebateState, DebatePhase
├── bull.py             # BullAgent(SubAgent)
├── bear.py             # BearAgent(SubAgent)
└── moderator.py        # ModeratorAgent

bullsh/ui/
└── debate_display.py   # Rich formatting, inline headers
```

## Files to Modify

| File | Changes |
|------|---------|
| `bullsh/cli.py` | Add `debate` command |
| `bullsh/ui/repl.py` | Add `/debate`, hint detection |
| `bullsh/agent/orchestrator.py` | Add `_detect_debate()`, `_run_debate()` |
| `bullsh/storage/sessions.py` | Add debate state serialization |

---

## Implementation Phases

### Phase 1: Core Agents
- [ ] `BullAgent` extending SubAgent
- [ ] `BearAgent` extending SubAgent
- [ ] `ModeratorAgent` (no tools)
- [ ] Unit tests with mocked responses

### Phase 2: Coordination
- [ ] `DebateCoordinator` with 4-phase flow
- [ ] `DebateState` for persistence
- [ ] Context passing between phases
- [ ] Orchestrator integration

### Phase 3: CLI/REPL
- [ ] `bullsh debate TICKER [--deep] [--framework]`
- [ ] `/debate` slash command
- [ ] Natural language detection
- [ ] Hint injection handling

### Phase 4: Polish
- [ ] Real-time streaming per phase
- [ ] Inline headers (🐂 BULL CASE, etc.)
- [ ] Resume from interruption
- [ ] Session saving (`/save`)
- [ ] Export (markdown, PDF)
- [ ] Framework context injection

---

*Spec Version: 1.0 (Finalized)*
*Last Updated: 2026-01-06*
