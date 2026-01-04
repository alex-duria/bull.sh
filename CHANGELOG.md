# Changelog

All notable changes to bullsh will be documented in this file. Written in plain English for maintainers.

---

## 2026-01-04 - Terminal UI Overhaul & Polish

**Developer**: Alexander Duria

### Added

- **Animated Intro Sequence** (`src/bullsh/ui/intro.py`):
  - Full-terminal candlestick chart animation on startup
  - Realistic market data generation with trending behavior
  - Chart builds progressively over ~4 seconds
  - Ticker header with price, change %, and "LIVE" indicator
  - Volume bars at bottom of chart
  - Scrolling ticker tape (AAPL +2.3% • NVDA +5.1% • ...)
  - Smooth transition to ASCII logo with tagline
  - "Made with ❤ by Alexander Duria" credit
  - Skip with `--no-intro` flag

- **Theme System** (`src/bullsh/ui/theme.py`):
  - Consistent color palette (bull green, bear red, cyan accents)
  - Rich theme integration for styled console output
  - Helper formatters: `format_percent()`, `format_price()`, `format_ticker()`
  - Progress bar characters and box-drawing utilities

- **Response Formatter** (`src/bullsh/ui/formatter.py`):
  - Beautiful markdown-to-terminal rendering
  - Underlined section headers
  - Numbered lists with gold accent numbers
  - Source citations styled in cyan underline
  - Percentages color-coded (+green, -red)
  - Dollar amounts in bold white
  - Cleans tool status noise from output

- **`/format` Command**: Re-displays last response with beautiful formatting

- **Command Autocomplete**: Tab completion for all slash commands with descriptions

### Changed

- **Welcome Screen**: ASCII logo, tagline, compact quick-start guide
- **CLI**: Added `--no-intro` flag to skip animation

### Files Created

```
src/bullsh/ui/intro.py      - Animated intro sequence
src/bullsh/ui/theme.py      - Color theme and formatters
src/bullsh/ui/formatter.py  - Beautiful response formatter
```

---

## 2026-01-04 - Valuation Framework & Smart Exports

**Developer**: Alexander Duria

### Added

- **Valuation Framework** (`src/bullsh/frameworks/valuation.py`):
  - Multi-method price target generation
  - P/E Multiple, Forward P/E, EV/EBITDA, Analyst Consensus, PEG-based
  - Bear/Base/Bull case targets with confidence ratings
  - `/framework valuation` to activate

- **Session-Aware Exports**:
  - `/excel` with no args exports ALL session tickers
  - `/export` aggregates ALL assistant messages
  - Warns if exporting ticker not researched in session
  - Auto-comparison if 2-3 tickers in session

- **Export Reminders**: Agent reminds user to export when framework completes

### Files Created/Modified

```
src/bullsh/frameworks/valuation.py  - NEW - Valuation framework
src/bullsh/frameworks/base.py       - Added VALUATION_FRAMEWORK
src/bullsh/agent/orchestrator.py    - Valuation prompt + export reminders
src/bullsh/ui/repl.py               - Session-aware exports
```

---

## 2026-01-04 - Agent Efficiency & Bug Fixes

**Developer**: Alexander Duria

### Changed

- **Max Iterations Increased**:
  - Orchestrator: 6 → **15** iterations
  - ResearchAgent: 5 → **10** iterations
  - CompareAgent: 3 → **5** (synthesis), 4 → **8** (per company)

- **Parallel Tool Guidance**: Agent now makes multiple tool calls together

### Fixed

- **Empty Message Error**: Fixed API error on follow-up after max iterations
- **MergedCell Excel Error**: `_auto_column_width()` now skips merged cells
- **Blocking API in Subagents**: Changed to `AsyncAnthropic` with `await`
- **Missing 'pitch' Framework**: Restored accidentally removed framework
- **Dividend Yield Display**: Fixed Excel formatting (decimal → percentage)

### Files Modified

```
src/bullsh/agent/orchestrator.py  - Max iterations, parallel guidance
src/bullsh/agent/base.py          - AsyncAnthropic fix
src/bullsh/agent/research.py      - Increased iterations
src/bullsh/agent/compare.py       - Increased iterations
src/bullsh/tools/excel.py         - MergedCell + dividend fixes
src/bullsh/ui/repl.py             - Empty message fix
```

---

## 2026-01-03 - Excel Financial Model Export

**Developer**: Alexander Duria

### Added

- **Excel Spreadsheet Generation** (`/excel` command):
  - Generate multi-sheet Excel workbooks with financial data
  - Key Metrics sheet: Price, P/E, market cap, 52-week range, volume, sector
  - Financial Ratios sheet: Valuation ratios, analyst price targets
  - Comparison sheet: Side-by-side comparison for multiple tickers
  - Valuation Analysis sheet: Price vs targets, 52-week range position
  - Professional formatting with headers, borders, color coding

- **Agent Tool** (`generate_excel`):
  - Agent can create Excel files after gathering market data
  - Supports single ticker or multi-ticker comparison
  - Returns path to generated file

- **REPL Commands**:
  - `/excel <TICKER>` - Generate Excel for single company
  - `/excel compare T1 T2 [T3]` - Generate comparison spreadsheet

### Files Created/Modified

```
pyproject.toml                    - Added openpyxl dependency
src/bullsh/tools/excel.py         - NEW - Excel generation module
src/bullsh/tools/base.py          - Added GENERATE_EXCEL_TOOL
src/bullsh/agent/orchestrator.py  - Wire up generate_excel handler
src/bullsh/ui/repl.py             - /excel command + help text
src/bullsh/ui/status.py           - Excel status description
```

### Output Location

Excel files saved to: `~/.bullsh/exports/{TICKER}_financial_model_{timestamp}.xlsx`

---

## 2026-01-03 - Web Search for Data Gaps

**Developer**: Alexander Duria

### Added

- **Web Search Tool** (`web_search`):
  - General DuckDuckGo web search for current information
  - Fills gaps when Yahoo Finance or social tools fail
  - Used to find real-time prices, earnings, announcements
  - Cached for 1 hour

- **System Prompt Update**:
  - Agent now automatically uses web search when other tools fail
  - Prioritizes always presenting complete research

### Files Modified

```
src/bullsh/tools/news.py          - Added web_search function
src/bullsh/tools/base.py          - Added WEB_SEARCH_TOOL
src/bullsh/agent/orchestrator.py  - Wire up web_search + updated prompts
src/bullsh/ui/status.py           - Web search status description
```

---

## 2026-01-03 - RAG for SEC Filings

**Developer**: Claude (AI) with Alex

### Added

- **Vector Database**: DuckDB-based vector storage for SEC filings
  - Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
  - 3000 char chunks with 500 char overlap (based on tenk approach)
  - Cosine similarity search for semantic matching
  - Automatic indexing when filings are fetched

- **RAG Tools**: Two new tools for semantic search
  - `rag_search`: Find specific information in indexed filings
  - `rag_list`: Show what filings are available for search
  - Auto-indexed when sec_fetch is called

- **Optional Dependencies**: RAG features require extra install
  - `pip install bullsh[rag]` for DuckDB + sentence-transformers

### Changed

- **SEC Tool**: Now auto-indexes filings into vector database after fetch
- **System Prompt**: Updated to guide agent to use RAG for specific queries
- **Tool Count**: Now 10 tools available (added rag_search, rag_list)

### Why This Helps

Before: 10-K filings are ~500k chars, we truncate to 8k (95%+ data loss)
After: Semantic search finds the most relevant chunks for any query

### Files Modified

```
pyproject.toml                      - Added [rag] optional deps
src/bullsh/storage/vectordb.py      - NEW - Vector database
src/bullsh/storage/__init__.py      - Export vectordb
src/bullsh/tools/rag.py             - NEW - RAG search tools
src/bullsh/tools/base.py            - RAG tool definitions
src/bullsh/tools/__init__.py        - Export rag module
src/bullsh/tools/sec.py             - Auto-index after fetch
src/bullsh/agent/orchestrator.py    - Wire up RAG tools
```

---

## 2026-01-03 - Token Optimization

**Developer**: Claude (AI) with Alex

### Added

- **Anthropic Prompt Caching**: System prompt cached across API calls
  - Uses `cache_control: ephemeral` for system prompt
  - 90% cost reduction on cached input tokens
  - Cache hit rate tracked and displayed in `/usage`
  - Configurable via `enable_prompt_caching` setting

- **History Sliding Window**: Prevents unbounded context growth
  - Keeps first message + last N messages (default: 20)
  - Inserts truncation note when window applied
  - Prevents exponential token growth in long sessions
  - Configurable via `max_history_messages` setting

- **Enhanced `/usage` Display**:
  - Shows cache read/write tokens
  - Displays cache hit rate percentage
  - Shows history message count and window status

### Changed

- **TokenUsage Tracking**: Now includes cache metrics
  - `cache_read_tokens`: Tokens read from cache
  - `cache_creation_tokens`: Tokens written to cache
  - `cache_hit_rate`: Percentage of input from cache
  - Cost calculation accounts for cache discounts

### Config Options

```toml
[cost_controls]
max_history_messages = 20      # Sliding window size
enable_prompt_caching = true   # Use Anthropic caching
tool_result_max_chars = 8000   # Max chars per tool result
```

### Files Modified

```
src/bullsh/config.py             - Added optimization settings
src/bullsh/agent/orchestrator.py - Caching + sliding window
src/bullsh/ui/repl.py            - Enhanced /usage display
```

---

## 2026-01-03 - Token Tracking and Cost Controls

**Developer**: Claude (AI) with Alex

### Added

- **Token Usage Tracking**: Every API call now tracks input and output tokens
  - Session-level accumulator for total usage
  - Per-turn tracking for individual requests
  - `TokenUsage` dataclass with estimated cost calculation

- **Cost Warnings**: Automatic warnings when usage approaches limits
  - Configurable warning threshold (default: 80% of limit)
  - Warning displayed inline during response
  - Different warnings for session vs turn limits

- **Hard Limits (Kill Switch)**: Prevents runaway costs
  - Session limit: 500,000 tokens (~$7.50 with Sonnet)
  - Turn limit: 50,000 tokens per conversation turn
  - `TokenLimitExceeded` exception stops execution
  - Session auto-saved before limit enforcement

- **`/usage` Command**: View current token usage and costs
  - Shows input/output token breakdown
  - Estimated cost in USD
  - Percentage of limits used
  - Color-coded (green/yellow/red) based on usage

- **Configurable Limits**: All limits can be overridden in `config.toml`
  ```toml
  [cost_controls]
  max_tokens_per_session = 500000
  max_tokens_per_turn = 50000
  warn_at_token_pct = 0.8
  cost_per_1k_input = 0.003
  cost_per_1k_output = 0.015
  ```

### Files Modified

```
src/bullsh/config.py             - Added cost control settings
src/bullsh/agent/orchestrator.py - Token tracking and limit enforcement
src/bullsh/agent/__init__.py     - Export TokenUsage, TokenLimitExceeded
src/bullsh/ui/repl.py            - /usage command, exception handling
```

---

## 2026-01-03 - Interactive Environment UX

**Developer**: Claude (AI) with Alex

### Changed

- **REPL is now the primary interface**: Type `bullsh` to enter a full interactive environment
- **Commands work directly in REPL**: No need to prefix with `bullsh`
  - `research NVDA` - research a company
  - `compare AMD NVDA INTC` - compare companies
  - `thesis AAPL` - generate thesis
  - `summary MSFT` - quick overview
  - `frameworks list` - show frameworks
- **Flags work inline**: `research NVDA -f piotroski` for framework-guided research
- **Natural language still works**: Just type questions and the agent responds
- **New welcome banner**: Shows available commands on startup
- **Improved help**: `/help` or `help` shows comprehensive command reference
- **Prompt changed**: From `You` to `>` for cleaner interface

### UX Flow

```
$ bullsh                     # Enter environment
> research NVDA              # Commands work directly
> What are the risks?        # Natural language works too
> compare AMD NVDA -f piotroski   # With framework
> exit                       # Leave environment
```

---

## 2026-01-03 - CLI Commands and Cache Integration

**Developer**: Claude (AI) with Alex

### Added

- **Compare Command** (`bullsh compare AMD NVDA INTC`):
  - Side-by-side comparison of up to 3 companies
  - Valuation metrics, growth rates, financial health comparison
  - Analyst sentiment comparison across companies
  - Framework support (Piotroski/Porter) for comparative analysis
  - Auto-enters interactive mode for follow-up questions

- **Thesis Command** (`bullsh thesis NVDA`):
  - Full investment thesis generation in Hedge Fund Pitch format
  - Sections: Thesis, Overview, Catalysts, Valuation, Financial Health, Risks, Conclusion
  - Auto-saves to markdown file with provenance
  - Professional hedge fund pitch formatting

- **Summary Command** (`bullsh summary NVDA`):
  - Quick 300-word company overview
  - Business description, valuation, performance, sentiment, key risk
  - Fast execution for quick research checks

- **Framework Creation Wizard** (`bullsh frameworks create`):
  - Interactive wizard to create custom TOML frameworks
  - Define criteria with questions and data sources
  - Optional scoring configuration
  - Saves to `~/.bullsh/frameworks/custom/`

- **Framework Edit Command** (`bullsh frameworks edit custom:name`):
  - Opens custom framework in system editor
  - Auto-detects editor from $EDITOR, $VISUAL, or common editors

### Changed

- **Cache Integration with All Tools**:
  - SEC tools: 7-day cache for filings
  - Yahoo Finance: 1-hour cache for prices/ratings
  - StockTwits: 1-hour cache for sentiment
  - Reddit: 2-hour cache for discussions
  - News: 4-hour cache for articles
  - All tools now check cache before making API calls
  - Successful results automatically cached

- **Tool Results**: Added `cached` flag to ToolResult to indicate when data came from cache

### Files Modified

```
src/bullsh/cli.py           - Added compare, thesis, summary, frameworks create/edit
src/bullsh/tools/sec.py     - Cache integration
src/bullsh/tools/yahoo.py   - Cache integration
src/bullsh/tools/social.py  - Cache integration
src/bullsh/tools/news.py    - Cache integration
```

---

## 2026-01-01 - Analysis Frameworks and Storage Layer

**Developer**: Claude (AI) with Alex

### Added

- **Piotroski F-Score Framework** (`src/bullsh/frameworks/piotroski.py`):
  - `FinancialData` dataclass with computed properties (ROA, current ratio, gross margin, asset turnover)
  - `FScoreResult` with 9 binary signals and scoring (Strong/Neutral/Weak ratings)
  - `compute_fscore()` function implementing all 9 Piotroski signals:
    - Profitability (4 points): ROA > 0, CFO > 0, ROA increasing, CFO > Net Income
    - Leverage & Liquidity (3 points): Debt decreasing, current ratio increasing, no dilution
    - Efficiency (2 points): Gross margin increasing, asset turnover increasing
  - `extract_financial_data_from_filing()` for basic text extraction from SEC filings

- **Porter's Five Forces Framework** (`src/bullsh/frameworks/porter.py`):
  - `ForceStrength` enum (LOW, MODERATE, HIGH, UNKNOWN)
  - `ForceAnalysis` dataclass with evidence extraction and user overrides
  - `FivesForcesResult` with overall industry attractiveness assessment
  - `FORCE_KEYWORDS` dictionary for keyword-based evidence extraction from 10-K filings
  - `analyze_five_forces()` function analyzing all 5 competitive forces

- **Framework Base Classes** (`src/bullsh/frameworks/base.py`):
  - `Framework` dataclass with progress tracking, scoring, and checklist display
  - `Criterion` dataclass for individual framework criteria
  - `FrameworkType` enum (QUANTITATIVE, QUALITATIVE, OUTPUT)
  - Built-in framework definitions: PIOTROSKI_FRAMEWORK, PORTER_FRAMEWORK, PITCH_FRAMEWORK
  - `load_framework()` for loading built-in and custom TOML frameworks
  - `list_frameworks()` for listing all available frameworks

- **Caching Layer** (`src/bullsh/storage/cache.py`):
  - File-based cache with JSON storage
  - TTL-based expiration (different defaults per source: SEC 7 days, Yahoo 1 hour, etc.)
  - Cache statistics and entry listing
  - Per-ticker and per-source invalidation
  - Global cache instance via `get_cache()`

- **Session Management** (`src/bullsh/storage/sessions.py`):
  - `Session` and `Message` dataclasses for conversation persistence
  - `SessionManager` for save, load, list, search, and rename operations
  - Topic-inferred session naming from conversation content
  - Auto-save on exit and periodic saves during conversation
  - Session search by content, ticker, or framework

### Changed

- **CLI Research Command** (`src/bullsh/cli.py`):
  - Now fully functional with framework support
  - Creates session automatically and saves on completion
  - Supports `--interactive` flag (default: true) to enter REPL after initial research
  - Framework-specific initial prompts for Piotroski and Porter

- **CLI Resume Command**: Now fully implemented with session loading and history restoration

- **CLI Frameworks Subcommands**:
  - `frameworks list` now shows table with all built-in and custom frameworks
  - `frameworks show <name>` displays full framework details and criteria

- **REPL** (`src/bullsh/ui/repl.py`):
  - Integrated session management with auto-save
  - `/save` command now works - saves current session
  - `/sessions` command - lists recent sessions in a table
  - `/resume <id>` command - loads and displays session info
  - `/cache` commands fully implemented:
    - `/cache` or `/cache stats` - shows cache statistics
    - `/cache list` - shows cached entries in a table
    - `/cache clear` - clears all cached data
    - `/cache refresh <ticker>` - invalidates cache for a specific ticker
  - `/sources` - shows data sources used for each ticker in session
  - `/export` - exports last response to markdown file
  - `/checklist` - shows framework progress when a framework is active
  - Improved help text with better command grouping

### Files Created

```
src/bullsh/frameworks/
├── __init__.py      - Module exports
├── base.py          - Framework loader and base classes
├── piotroski.py     - F-Score computation logic
└── porter.py        - Five Forces analysis logic

src/bullsh/storage/
├── __init__.py      - Module exports
├── cache.py         - File-based caching with TTL
└── sessions.py      - Session persistence and management
```

---

## 2026-01-01 - Initial Project Scaffolding

**Developer**: Claude (AI) with Alex

### Added

- **Project foundation**: Complete Python package structure with `pyproject.toml`, configured for Python 3.12+
- **CLI interface**: Typer-based CLI with subcommands (`research`, `compare`, `thesis`, `summary`, `resume`, `frameworks`)
- **Configuration system**:
  - `.env` loading for secrets (API keys)
  - `config.toml` support for user preferences
  - First-run setup wizard that prompts for required keys
- **Tool infrastructure**:
  - `ToolResult` dataclass with confidence scores (0-1) for data quality assessment
  - `ToolStatus` enum (SUCCESS, PARTIAL, FAILED, RATE_LIMITED, CACHED)
  - Tool definitions in Claude API format for all 8 tools
- **SEC EDGAR integration**: Search and fetch 10-K/10-Q filings via edgartools library
- **Yahoo Finance scraper**: Analyst ratings, price, P/E with confidence-based extraction
- **Social sentiment tools**:
  - StockTwits API integration (primary)
  - Reddit scraping fallback (old.reddit.com)
- **News search**: DuckDuckGo integration for financial news
- **Thesis export**: Markdown output with YAML frontmatter for provenance tracking
- **Agent orchestrator**:
  - Streaming responses with tool call status display
  - Framework-aware system prompts (Piotroski, Porter)
  - Conversation history management
- **Interactive REPL**:
  - Rich console formatting
  - Slash commands (`/help`, `/framework`, `/config`, `/export`, etc.)
  - Framework switching mid-session
- **Test foundation**: pytest setup with fixtures for mocked API responses

### Architecture Decisions Documented

- Hybrid orchestrator pattern for subagent parallelism
- Task-based agents (research, compare, thesis) rather than data-source-based
- Selective context passing to subagents (not full conversation)
- Max 3 tool iterations per subagent to control costs
- Graceful degradation when tools fail

### Analysis Frameworks Specified

- **Piotroski F-Score**: 9-point quantitative health scoring
- **Porter's Five Forces**: Competitive moat analysis
- **Hedge Fund Pitch**: Thesis output format (Thesis → Catalysts → Valuation → Risks)
- Custom framework support via TOML files

### Files Created

```
pyproject.toml                    - Package config, dependencies, CLI entry point
README.md                         - Quick start guide
.env.example                      - Required environment variables template
spec.md                           - Full product specification
CLAUDE.md                         - AI assistant guidance

src/bullsh/
├── __init__.py                   - Package version
├── __main__.py                   - python -m bullsh entry
├── cli.py                        - Typer CLI with all subcommands
├── config.py                     - Configuration management
├── agent/
│   ├── __init__.py
│   └── orchestrator.py           - Main agent loop with streaming
├── tools/
│   ├── __init__.py               - Tool exports
│   ├── base.py                   - ToolResult, ToolStatus, definitions
│   ├── sec.py                    - SEC EDGAR tools
│   ├── yahoo.py                  - Yahoo Finance scraping
│   ├── social.py                 - StockTwits + Reddit
│   ├── news.py                   - DuckDuckGo news
│   └── thesis.py                 - Thesis export
├── ui/
│   ├── __init__.py
│   └── repl.py                   - Interactive REPL
├── frameworks/
│   └── __init__.py
└── storage/
    └── __init__.py

tests/
├── __init__.py
├── conftest.py                   - Pytest fixtures
├── test_config.py                - Config tests
└── test_tools_base.py            - Tool infrastructure tests
```

### Dependencies

```
anthropic>=0.40.0      - Claude API
typer>=0.12.0          - CLI framework
rich>=13.0.0           - Terminal formatting
httpx>=0.27.0          - Async HTTP
beautifulsoup4>=4.12.0 - HTML parsing
edgartools>=3.0.0      - SEC EDGAR
duckduckgo-search>=6.0.0 - News search
python-dotenv>=1.0.0   - Env loading
pydantic>=2.0.0        - Data validation
```

### What's Not Yet Implemented

- Session save/resume persistence
- Caching layer
- Piotroski F-Score computation logic
- Porter's Five Forces extraction logic
- Custom framework creation wizard
- Full subagent architecture (research, compare, thesis agents)
- Keybindings beyond readline defaults

---
