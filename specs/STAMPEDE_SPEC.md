# STAMPEDE: Bull.sh Next-Gen Agent Architecture

## Overview

**Stampede** is Bull.sh's new agent architecture inspired by Dexter's Plan→Execute→Reflect loop, but enhanced with Bull.sh's unique strengths: richer data sources, session memory, framework-guided analysis, and smart tool selection.

**Philosophy**: Not copying Dexter, but taking the best of both worlds - Dexter's structured autonomy with Bull.sh's financial depth.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      STAMPEDE LOOP                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Query                                                 │
│      ▼                                                      │
│  ┌──────────────────┐                                       │
│  │    UNDERSTAND    │  ← Extract intent, tickers, depth,    │
│  │                  │    export intent, confidence score    │
│  │   confidence<80% │──► ASK CLARIFYING QUESTION            │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  CLASSIFY DEPTH  │  ← Implicit: "P/E?" = quick,         │
│  │                  │    "thesis" = deep                    │
│  └────────┬─────────┘                                       │
│           ▼                                                 │
│      ┌────┴────┐                                           │
│      │ SIMPLE? │                                           │
│      └────┬────┘                                           │
│      YES  │  NO                                            │
│       ▼   │   ▼                                            │
│  ┌────────┐  ┌─────────────────────────────────────┐       │
│  │ 0-TASK │  │      ITERATION LOOP (max 5)         │       │
│  │ DIRECT │  │                                     │       │
│  │ ANSWER │  │  ┌──────────┐                       │       │
│  └───┬────┘  │  │   PLAN   │ ← Select frameworks,  │       │
│      │       │  │          │   create 1-10 tasks,  │       │
│      │       │  │          │   uses prior guidance │       │
│      │       │  └────┬─────┘                       │       │
│      │       │       ▼        [SHOW PLAN TO USER]  │       │
│      │       │  ┌──────────┐                       │       │
│      │       │  │ EXECUTE  │ ← Dynamic tool select,│       │
│      │       │  │          │   RAG-first policy,   │       │
│      │       │  │          │   retry once on fail  │       │
│      │       │  └────┬─────┘                       │       │
│      │       │       ▼                             │       │
│      │       │  ┌──────────┐                       │       │
│      │       │  │ REFLECT  │ ← isComplete? visible │       │
│      │       │  │          │   guidance to user    │       │
│      │       │  └────┬─────┘                       │       │
│      │       │       │                             │       │
│      │       │  complete? ─YES─► EXIT LOOP         │       │
│      │       │       │                             │       │
│      │       │      NO ─► fresh plan with guidance │       │
│      │       │                                     │       │
│      │       └─────────────────────────────────────┘       │
│      │                      │                               │
│      └──────────────────────┼───────────────────────────────┤
│                             ▼                               │
│                    ┌──────────────┐                         │
│                    │  SYNTHESIZE  │  ← Streaming answer,    │
│                    │              │    sources at end       │
│                    └──────────────┘                         │
│                             ▼                               │
│                    Streamed Response                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### From Interview

| Topic | Decision | Rationale |
|-------|----------|-----------|
| **Reflection Aggressiveness** | Smart tool selection, trust once comprehensive | More tools ≠ more iteration; be selective about WHICH tools |
| **Streaming UX** | Claude Code-style task progress | Keep user engaged with fancy terminal updates |
| **Task Count** | Dynamic 1-10 | Simple = 1, thesis = many |
| **Tool Selection** | Execution time (dynamic) | Executor decides in-the-moment based on data |
| **Framework Planning** | Framework-aware (e.g., 9 tasks for Piotroski) | Planner knows Piotroski = 9 signals, creates targeted tasks |
| **Session Memory** | Full context from prior research | "Already researched AMD" = skip AMD deep-dive |
| **Simple Queries** | 0-task shortcut | "P/E ratio?" goes straight to synthesis |
| **Research Depth** | Implicit inference | Agent infers quick vs deep from query |
| **Framework Selection** | Planner picks | Planning prompt selects appropriate frameworks |
| **Reflection Visibility** | Visible to user | Show guidance: "Missing margin data, replanning..." |
| **RAG Strategy** | RAG-first always | Try semantic search before fresh SEC fetch |
| **Understanding Confidence** | Threshold with clarification | Below 80% → ask user to clarify |
| **Model Selection** | Same model throughout | Sonnet for all phases, consistency matters |
| **Plan Visibility** | Show full plan before execution | User sees: [1] Fetch 10-K [2] Extract revenue... |
| **Source Attribution** | Summary at end | Answer first, then "Sources: task_1, task_2, task_3" |
| **Export Awareness** | Yes | Understanding extracts export intent, adjusts data gathering |
| **Failure Handling** | Retry once, then skip | Auto-retry, if still fails, note and continue |

### Features Kept Separate (Not in Stampede Loop)

| Feature | Reason |
|---------|--------|
| **Bull vs Bear Debate** | Stays as standalone `/debate` command - different UX paradigm |
| **Compare** | Stays as-is with parallel ResearchAgents - working well |

These features may USE Stampede internally in the future, but are not integrated as "task types".

---

## Phase Specifications

### Phase 1: UNDERSTAND

**Input**: User query, session context

**Output**: `Understanding` object

```python
class Understanding(BaseModel):
    # Core
    intent: str                      # "research", "valuation", "quick_lookup"
    tickers: list[str]               # ["NVDA", "AMD"]
    confidence: float                # 0.0-1.0, if <0.8 ask clarification

    # Depth inference
    inferred_depth: Literal["quick", "standard", "deep"]

    # Bull.sh specific
    timeframe: str | None            # "last 3 years", "Q3 2024"
    metrics_focus: list[str]         # ["revenue", "margins", "growth"]
    wants_sentiment: bool            # Inferred from query
    wants_factors: bool              # Inferred from query
    export_intent: Literal["none", "excel", "pdf"] | None

    # Session context
    prior_tickers_researched: list[str]  # From session memory
```

**Confidence Threshold**: If `confidence < 0.8`, pause and ask clarifying question before proceeding.

**Prompt Key Points**:
- Extract ALL relevant entities from query
- Consider session context (what was already researched)
- Be explicit about uncertainty

---

### Phase 2: PLAN

**Input**: Understanding, prior task results (if iteration > 1), guidance from reflection

**Output**: `TaskPlan` object

```python
class TaskType(str, Enum):
    USE_TOOLS = "use_tools"      # Fetch data with tools
    REASON = "reason"            # LLM analysis/synthesis only

class Task(BaseModel):
    id: str                          # "task_1"
    description: str                 # "Fetch NVDA 10-K filing"
    task_type: TaskType
    depends_on: list[str] = []       # ["task_1"] - must complete first
    rationale: str | None            # WHY this task (for advanced tools only)

class TaskPlan(BaseModel):
    summary: str                     # "Analyzing NVDA revenue trends with Piotroski framework"
    selected_frameworks: list[str]   # ["piotroski"] - planner picks
    tasks: list[Task]                # 1-10 tasks
    is_simple_query: bool            # If True and 0 tasks, go straight to synthesis
```

**Task Count Rules**:
- Simple lookup: 0 tasks (direct answer)
- Standard research: 2-5 tasks
- Deep analysis: 5-10 tasks
- Framework-specific: Match framework needs (9 for Piotroski)

**Framework-Aware Planning**:
```
If framework == "piotroski":
    Create tasks for each of 9 signals:
    - task_1: Fetch financials for ROA calculation
    - task_2: Fetch cash flow statement
    - task_3: Fetch prior year for YoY comparison
    ... etc
```

**Fresh Plan Each Iteration**:
- Planner sees ALL prior task results
- Creates entirely NEW plan (not appending)
- Guidance from reflection shapes priorities

**Display to User**:
```
Planning research...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tasks:
  [1] Fetch NVDA 10-K filing
  [2] Extract revenue data (depends: 1)
  [3] Calculate growth rates (depends: 2)
  [4] Fetch analyst consensus
  [5] Synthesize findings (depends: 3, 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Phase 3: EXECUTE

**Input**: TaskPlan, Understanding, abort signal

**Output**: `dict[str, TaskResult]`

**Execution Strategy**:
1. Dependency-aware parallelism
2. Dynamic tool selection at execution time
3. RAG-first policy for SEC data
4. Retry once on failure, then skip

```python
class TaskExecutor:
    async def execute_plan(self, plan: TaskPlan, understanding: Understanding):
        task_results = {}

        while has_pending_tasks(plan):
            # Get tasks with all dependencies satisfied
            ready = get_ready_tasks(plan, task_results)

            # Execute in parallel
            results = await asyncio.gather(*[
                self.execute_task(task, task_results, understanding)
                for task in ready
            ])

            # Store results
            for task, result in zip(ready, results):
                task_results[task.id] = result

        return task_results

    async def execute_task(self, task: Task, prior_results: dict, understanding: Understanding):
        if task.task_type == TaskType.USE_TOOLS:
            return await self.execute_with_tools(task, prior_results)
        else:
            return await self.execute_reasoning(task, prior_results)
```

**Tool Selection Policy**:
- Executor has access to ALL 14 tools
- Decides dynamically based on task description and data gaps
- RAG-first: Try `rag_search` before `sec_fetch`
- Smart selection: Don't query all 14 tools, pick what's needed

**Failure Handling**:
```python
async def execute_with_retry(self, task, ...):
    try:
        return await self._execute_once(task, ...)
    except ToolError:
        # Retry once
        try:
            return await self._execute_once(task, ...)
        except ToolError as e:
            # Skip and note
            return TaskResult(
                status="failed",
                data={"error": str(e)},
                note="Task failed after retry, skipped"
            )
```

**Progress Display** (Claude Code-style):
```
Executing tasks...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ task_1: Fetch NVDA 10-K filing
  ✓ task_2: Extract revenue data
  ◐ task_3: Calculate growth rates
  ○ task_4: Fetch analyst consensus
  ○ task_5: Synthesize findings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Phase 4: REFLECT

**Input**: Query, Understanding, task_results, iteration number

**Output**: `ReflectionResult`

```python
class ReflectionResult(BaseModel):
    is_complete: bool                # Can we answer the query?
    reasoning: str                   # Why complete/incomplete
    missing_info: list[str]          # What's missing (if incomplete)
    guidance: str                    # For next planning iteration
```

**Reflection Prompt Philosophy**:
```
DEFAULT TO COMPLETE.

Mark incomplete ONLY if:
- Critical data for PRIMARY entity is missing
- Cannot answer the CORE question

"Nice-to-have" enrichment does NOT justify another iteration.
Prefer pragmatic answers over perfect ones.

Remember: We have 14 tools. More tools ≠ use all tools.
Trust data quality once you have what's needed.
```

**Visible to User**:
```
Reflecting on results...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: Incomplete
Reason: Missing Q3 2024 margin data for trend analysis
Guidance: Focus next iteration on quarterly data extraction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Replanning (iteration 2/5)...
```

---

### Phase 5: SYNTHESIZE

**Input**: Query, Understanding, all task_results

**Output**: Streaming response

**Synthesis Prompt**:
```
You have completed research with the following task results:
{task_results}

User query: {query}
Understanding: {understanding}

Generate a comprehensive answer that:
1. Leads with the KEY FINDING in the first sentence
2. Uses SPECIFIC NUMBERS with context
3. Cites sources at the end (not inline)
4. Matches the depth the user expects (quick/standard/deep)
5. If export_intent is set, structure data for that format

Format sources at the end:
---
Sources: [task_1: 10-K filing], [task_4: analyst consensus]
```

---

## Session Memory Integration

Stampede is session-aware:

```python
class Understanding(BaseModel):
    # Session context
    prior_tickers_researched: list[str]  # ["AMD", "INTC"]
    prior_frameworks_used: list[str]     # ["piotroski"]
    prior_task_results: dict | None      # Cached results from session

# In planning
if "AMD" in prior_tickers_researched and query mentions AMD:
    # Skip AMD data gathering, use cached results
    guidance += "AMD was researched earlier in session, reuse that data"
```

---

## Export Awareness

Understanding extracts export intent:

```python
# Query: "Research NVDA, I want to export to Excel"
understanding.export_intent = "excel"

# In planning, this influences task creation:
if export_intent == "excel":
    # Add task to gather structured numerical data
    tasks.append(Task(
        description="Gather numerical metrics in structured format for Excel export",
        rationale="User intends to export to Excel"
    ))
```

---

## Features Intentionally Excluded

### Bull vs Bear Debate
- Stays as standalone `/debate` command
- May use Stampede internally in future
- Different UX paradigm (interactive rounds, user coaching)

### Compare
- Stays as-is with parallel ResearchAgents
- Already working well
- May be enhanced to use Stampede per-company later

---

## UI/UX Specifications

### Current: Rich + prompt_toolkit (Python)

For now, enhance current stack:
- Rich Live display for task progress
- Status updates visible to user
- Plan shown before execution
- Reflection reasoning visible

### Future: React + Ink (TypeScript)

**Note for spec**: Future UI rewrite to React+Ink like Dexter for:
- Smoother animations
- Better progress panels
- More sophisticated terminal UI

This is a **separate project**, not part of initial Stampede implementation.

---

## New Data Layer: Financial Datasets API

### Overview

**Financial Datasets API** (https://financialdatasets.ai/) becomes the **primary source** for structured financial data:
- Income statements
- Balance sheets
- Cash flow statements
- Insider transactions
- 30,000+ tickers, 30+ years history

### Integration Strategy

| Priority | Source | Use Case |
|----------|--------|----------|
| **1st** | Financial Datasets API | Structured financials (income, balance, cash flow) |
| **2nd** | SEC EDGAR (edgartools) | Full 10-K/10-Q text, RAG indexing |
| **3rd** | Yahoo Finance | Real-time price, analyst ratings |
| **Fallback** | Web search | Gaps, recent news |

### Configuration

```toml
# ~/.bullsh/config.toml or .env
FINANCIAL_DATASETS_API_KEY=fd_...  # User provides their own key
```

**Requirement**: User must configure their own Financial Datasets API key. Free tier available.

### Unified Financial Tool

Instead of separate tools, create a **smart unified tool**:

```python
# bullsh/tools/financials.py

class UnifiedFinancialsTool:
    """
    Single tool that fetches financial data from best available source.

    Priority:
    1. Financial Datasets API (if key configured)
    2. Compute from SEC filings
    3. Yahoo Finance fallback
    """

    async def get_financials(
        self,
        ticker: str,
        statement_type: Literal["income", "balance", "cashflow", "all"],
        period: Literal["annual", "quarterly"] = "annual",
        years: int = 3
    ) -> ToolResult:

        # Try Financial Datasets first
        if self.fd_api_key:
            try:
                return await self._fetch_from_fd(ticker, statement_type, period, years)
            except FDAPIError:
                pass  # Fall through to next source

        # Try computing from SEC filings
        try:
            return await self._compute_from_sec(ticker, statement_type, period, years)
        except SECDataError:
            pass

        # Yahoo Finance fallback
        return await self._fetch_from_yahoo(ticker, statement_type)
```

### Financial Datasets Endpoints

Based on Dexter's implementation:

```python
# Income Statement
GET /api/v1/income-statements/{ticker}
    ?period=annual|quarterly
    &limit=10

# Balance Sheet
GET /api/v1/balance-sheets/{ticker}
    ?period=annual|quarterly
    &limit=10

# Cash Flow Statement
GET /api/v1/cash-flow-statements/{ticker}
    ?period=annual|quarterly
    &limit=10

# Insider Transactions (new capability)
GET /api/v1/insider-transactions/{ticker}
    ?limit=100
```

### Tool Result Schema

```python
class FinancialsResult(BaseModel):
    ticker: str
    source: Literal["financial_datasets", "sec", "yahoo"]
    period: str
    statements: dict  # Raw statement data

    # Income statement fields
    revenue: float | None
    gross_profit: float | None
    operating_income: float | None
    net_income: float | None

    # Balance sheet fields
    total_assets: float | None
    total_liabilities: float | None
    shareholders_equity: float | None
    cash_and_equivalents: float | None

    # Cash flow fields
    operating_cash_flow: float | None
    investing_cash_flow: float | None
    financing_cash_flow: float | None
    free_cash_flow: float | None

    # Computed ratios
    roa: float | None
    roe: float | None
    current_ratio: float | None
    debt_to_equity: float | None
```

### Insider Transactions (New Capability)

Financial Datasets provides insider transaction data that Bull.sh doesn't currently have:

```python
class InsiderTransaction(BaseModel):
    filing_date: str
    insider_name: str
    insider_title: str
    transaction_type: Literal["buy", "sell", "gift", "exercise"]
    shares: int
    price_per_share: float
    total_value: float

# New tool
async def get_insider_transactions(ticker: str, limit: int = 50) -> list[InsiderTransaction]:
    """Fetch recent insider buy/sell activity."""
```

This enables new analysis: "Are insiders buying or selling?"

---

## Implementation Files

### New Files to Create

| File | Purpose |
|------|---------|
| `bullsh/agent/stampede/understanding.py` | Understanding phase + confidence |
| `bullsh/agent/stampede/planner.py` | Task planning with framework awareness |
| `bullsh/agent/stampede/executor.py` | Dependency-aware task execution |
| `bullsh/agent/stampede/reflector.py` | Reflection with visible guidance |
| `bullsh/agent/stampede/synthesizer.py` | Final answer synthesis |
| `bullsh/agent/stampede/schemas.py` | Pydantic models for all phases |
| `bullsh/agent/stampede/loop.py` | Main Stampede orchestrator loop |
| `bullsh/tools/financials.py` | Unified financial data tool (FD API + fallbacks) |
| `bullsh/tools/insiders.py` | Insider transaction tool (new capability) |

### Files to Modify

| File | Changes |
|------|---------|
| `bullsh/agent/orchestrator.py` | Replace with Stampede loop |
| `bullsh/ui/repl.py` | Handle Stampede progress messages |
| `bullsh/agent/__init__.py` | Export Stampede classes |
| `bullsh/config.py` | Add FINANCIAL_DATASETS_API_KEY config |
| `bullsh/tools/base.py` | Register new financial tools |

---

## Summary: Best of Both Worlds

| From Dexter | From Bull.sh | Stampede Synthesis |
|-------------|--------------|-------------------|
| Plan→Execute→Reflect loop | 14 rich data tools | Structured loop with smart tool selection |
| Task decomposition | RAG over SEC filings | Framework-aware task planning |
| Self-validation | Session persistence | Session-aware reflection |
| Bounded iteration | Export to Excel/PDF | Export-aware planning |
| Progress visibility | Debate feature | Visible plans + reflection (debate separate) |
| 0-task shortcut | Factor analysis | Keep all Bull.sh tools, use wisely |
| Confidence scoring | Framework guidance | Planner selects frameworks |
| **Financial Datasets API** | Yahoo + EDGAR | **Unified financials tool with FD as primary** |
| Insider transactions | - | **New insider activity analysis** |

**Stampede = Dexter's brain + Bull.sh's muscle + Financial Datasets' data**

---

## Data Source Priority

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCE HIERARCHY                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STRUCTURED FINANCIALS (income, balance, cash flow)         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Financial Datasets API  ← PRIMARY (if key set)   │  │
│  │  2. SEC EDGAR + compute     ← FALLBACK               │  │
│  │  3. Yahoo Finance           ← LAST RESORT            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  FULL SEC FILINGS (10-K text, RAG search)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SEC EDGAR (edgartools)     ← ONLY SOURCE            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  REAL-TIME PRICE + ANALYST                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Yahoo Finance              ← PRIMARY                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  INSIDER TRANSACTIONS                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Financial Datasets API     ← ONLY SOURCE (NEW!)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  SENTIMENT                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  StockTwits + Reddit        ← EXISTING               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
