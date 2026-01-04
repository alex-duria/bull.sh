# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**bullsh** is an agentic CLI for investment research. Users prompt naturally, and an AI agent autonomously gathers SEC filings, social sentiment, analyst data, and news to build investment theses.

## Build & Development Commands

```bash
# Install dependencies
pip install -e .

# Run the CLI
bullsh                          # Interactive REPL (freestyle)
bullsh research NVDA            # Freestyle research
bullsh research NVDA --framework piotroski  # Framework-guided
bullsh research NVDA --framework porter
bullsh compare AMD NVDA INTC    # Compare up to 3 companies
bullsh thesis AAPL --output thesis.md
bullsh frameworks list          # Show available frameworks
bullsh frameworks create        # Interactive custom framework builder

# Run tests
pytest
pytest --cov=src/bullsh

# Type checking (if configured)
mypy src/bullsh
```

## Architecture

Uses a **hybrid orchestrator pattern** with task-based subagents for parallelism and context efficiency.

```
src/bullsh/
├── cli.py              # Typer app, subcommands, REPL entry
├── config.py           # Load .env (secrets) + config.toml (prefs)
├── agent/
│   ├── orchestrator.py # Lightweight dispatcher, selective context passing
│   ├── base.py         # Base agent class with iteration bounds (max 3)
│   ├── research.py     # Single-company deep dive subagent
│   ├── compare.py      # Multi-company parallel research subagent
│   ├── thesis.py       # Thesis structuring subagent
│   ├── context.py      # Smart compression, selective passing
│   ├── tools.py        # Tool definitions (JSON schema for Claude)
│   └── prompts.py      # System prompts for each agent type
├── frameworks/
│   ├── base.py         # Framework loader, base class
│   ├── builtin/        # Ships with package (piotroski.toml, porter.toml, pitch.toml)
│   ├── piotroski.py    # F-Score computation (9-point quantitative)
│   ├── porter.py       # Five Forces extraction (qualitative)
│   └── custom.py       # Custom framework parser/validator
├── tools/
│   ├── base.py         # ToolResult dataclass with confidence scores
│   ├── sec.py          # SEC EDGAR via edgartools
│   ├── stocktwits.py   # Primary social sentiment (scraping)
│   ├── reddit.py       # Fallback social sentiment (scraping)
│   ├── yahoo.py        # Analyst ratings (scraping, returns confidence)
│   ├── news.py         # DuckDuckGo news search
│   ├── ratios.py       # P/E, EV/EBITDA computation
│   └── thesis.py       # Export with YAML frontmatter provenance
├── ui/
│   ├── repl.py         # Interactive loop with Rich
│   ├── display.py      # Tables, streaming output
│   ├── commands.py     # Slash command handlers (/cache, /sources, etc.)
│   └── keybindings.py  # Ctrl+S, Ctrl+L, etc.
└── storage/
    ├── cache.py        # User-controlled cache refresh
    └── sessions.py     # Topic-inferred session naming, summarized history
```

## Analysis Frameworks

Frameworks provide **structure, not constraints**. They guide research while preserving user creativity.

**Built-in:**
- **Piotroski F-Score**: 9-point quantitative financial health (computed from 10-K/10-Q)
- **Porter's Five Forces**: Qualitative competitive analysis (extracted from 10-K text)
- **Hedge Fund Pitch**: Output format for thesis (Thesis → Catalysts → Valuation → Risks)

**Custom frameworks**: Users can define their own criteria in TOML files (`~/.bullsh/frameworks/custom/`)

**Philosophy**: The thesis is the user's - shaped by their questions, interpretations, and follow-ups. Two users researching the same stock produce different theses.

## Key Design Decisions

- **Frameworks as compass**: Guide conversation without limiting exploration; users can override assessments
- **Hybrid orchestrator**: Lightweight dispatcher routes to task-based subagents (research, compare, thesis)
- **Selective context**: Orchestrator decides what context each subagent needs, not full conversation
- **Bounded iterations**: Max 3 tool calls per subagent to prevent runaway cost
- **Parallel in compare**: Compare agent spawns research subagents in parallel (one per company)
- **Graceful degradation**: Failed subagents noted in response, never blocks entire flow
- **Single API key**: Only Anthropic API required; all data sources are free/scraped
- **Confidence scores**: All scraped tools return `ToolResult` with confidence 0-1
- **Minimum data threshold**: Requires at least one 10-K filing; refuses private companies
- **Tone enforcement**: System prompt prohibits "buy/sell" recommendations

## Testing

Unit tests with mocked responses only - no external network calls in CI. Fixtures in `tests/` provide sample SEC filings, Yahoo HTML, etc.

## Planning & Roadmap

See `PLANNING.md` for detailed implementation plans for remaining work:
- Subagent architecture (parallel research)
- Keybindings (Ctrl+S, etc.)
- Test coverage expansion

## Changelog Maintenance

**IMPORTANT**: After completing any significant work, you MUST update `CHANGELOG.md` with a plain-English summary of changes.

### What to log:
- New features or commands added
- Bug fixes
- Architecture changes
- Breaking changes
- Dependencies added/removed
- Files created or significantly modified

### Format:
```markdown
## [Date] - Brief Title

**Developer**: [Name or "Claude (AI)"]

### Added
- Description of new feature

### Changed
- Description of modification

### Fixed
- Description of bug fix

### Files Modified
- `path/to/file.py` - what changed
```

### Guidelines:
- Write for human maintainers, not machines
- Be specific: "Added SEC 10-K parsing" not "Updated tools"
- Include the WHY when non-obvious
- Group related changes together
- Most recent entries at the top
