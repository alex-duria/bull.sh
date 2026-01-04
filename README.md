<p align="center">
  <img src="static/bullsh-banner.png" alt="bullsh banner" width="800"/>
</p>

<p align="center">
  <strong>Agentic CLI for Investment Research</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#frameworks">Frameworks</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/powered%20by-Claude-orange.svg" alt="Powered by Claude"/>
</p>

---

**bullsh** is an AI-powered command-line tool that transforms investment research. Ask questions in natural language, and get comprehensive analysis synthesized from SEC filings, market data, analyst ratings, and social sentiment—all in your terminal.

## Features

### Core Research Capabilities

| Feature | Description |
|---------|-------------|
| **Natural Language Queries** | Ask questions like "What are NVIDIA's competitive advantages?" and get synthesized answers |
| **SEC EDGAR Integration** | Automatic fetching and parsing of 10-K, 10-Q, and 8-K filings |
| **Market Data** | Real-time prices, P/E ratios, analyst ratings, and price targets from Yahoo Finance |
| **Sentiment Analysis** | Social sentiment from StockTwits and Reddit discussions |
| **News Search** | Recent financial news via DuckDuckGo |
| **Web Search Fallback** | General web search when specialized sources lack data |

### Analysis Frameworks

| Framework | Type | Description |
|-----------|------|-------------|
| **Piotroski F-Score** | Quantitative | 9-point financial health scoring (ROA, cash flow, leverage, efficiency) |
| **Porter's Five Forces** | Qualitative | Competitive positioning and moat analysis |
| **Valuation Analysis** | Quantitative | Multi-method price targets with bear/base/bull cases |
| **Hedge Fund Pitch** | Output | Professional thesis format with catalysts and risks |
| **Custom Frameworks** | User-defined | Create your own analysis frameworks via TOML |

### Advanced Features

| Feature | Description |
|---------|-------------|
| **RAG/Vector Search** | Semantic search over indexed SEC filings using sentence-transformers |
| **Parallel Research** | Subagent architecture for comparing multiple companies simultaneously |
| **Session Management** | Save, resume, and search through research sessions |
| **Excel Export** | Generate financial models with multiple sheets (metrics, ratios, comparisons) |
| **PDF/DOCX Export** | Export research to professional documents |
| **Token Optimization** | Prompt caching and cost tracking to minimize API usage |
| **Animated Terminal UI** | Beautiful candlestick chart intro and themed output |

### Data Sources (All Free)

- **SEC EDGAR** — Official filings via edgartools
- **Yahoo Finance** — Market data, analyst ratings (scraped)
- **StockTwits** — Social sentiment API
- **Reddit** — Community discussions (fallback)
- **DuckDuckGo** — News and web search

## Installation

### Prerequisites

- Python 3.12 or higher
- [Anthropic API key](https://console.anthropic.com/) (for Claude)
- SEC EDGAR identity (name + email for compliance)

### Install from PyPI

```bash
pip install bullsh
```

### Install from Source

```bash
git clone https://github.com/aduria/bullsh.git
cd bullsh
pip install -e .
```

### Optional Dependencies

```bash
# For RAG/semantic search over SEC filings
pip install bullsh[rag]

# For PDF/DOCX export
pip install bullsh[export]

# For development
pip install bullsh[dev]

# All optional features
pip install bullsh[rag,export]
```

## Quick Start

### First Run Setup

```bash
bullsh
```

On first run, you'll be prompted to configure:

1. **Anthropic API Key** — Get yours at [console.anthropic.com](https://console.anthropic.com/)
2. **SEC EDGAR Identity** — Required by SEC: `"Your Name your@email.com"`

Or set environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export EDGAR_IDENTITY="Your Name your@email.com"
```

### Your First Research Session

```bash
bullsh
```

```
╭──────────────────────────────────────────────────────────────╮
│                         BULL.SH                              │
│              AI-Powered Investment Research                  │
╰──────────────────────────────────────────────────────────────╯

You: Research NVDA and tell me about their AI moat

Agent: I'll research NVIDIA for you...
  ◐ Searching SEC filings (NVDA)...
  ✓ Fetching SEC filing
  ◐ Getting market data (NVDA)...
  ✓ done
  ◐ Checking social sentiment (NVDA)...
  ✓ done

Based on my research, NVIDIA has several competitive advantages...
```

## Usage

### Command Line Interface

```bash
# Start interactive REPL
bullsh

# Start with specific ticker context
bullsh research NVDA

# Compare multiple companies
bullsh compare AMD NVDA INTC

# Generate investment thesis
bullsh thesis AAPL

# Use specific framework
bullsh research NVDA --framework piotroski

# Skip animated intro
bullsh --no-intro

# Enable debug logging
bullsh --debug
bullsh --debug --debug-filter "tools,api"
```

### Interactive Commands (REPL)

#### Research Commands

| Command | Description |
|---------|-------------|
| `/research TICKER` | Start researching a company |
| `/compare TICKER1 TICKER2 [TICKER3]` | Compare 2-3 companies side by side |
| `/thesis [TICKER]` | Generate full investment thesis |

#### Framework Commands

| Command | Description |
|---------|-------------|
| `/framework piotroski` | Switch to Piotroski F-Score analysis |
| `/framework porter` | Switch to Porter's Five Forces |
| `/framework valuation` | Switch to Valuation Analysis |
| `/framework pitch` | Switch to Hedge Fund Pitch format |
| `/framework off` | Return to freestyle research mode |
| `/frameworks` | List all available frameworks |
| `/checklist` | Show current framework progress |

#### Session Commands

| Command | Description |
|---------|-------------|
| `/save [name]` | Save current session |
| `/sessions` | List saved sessions |
| `/resume SESSION_ID` | Resume a previous session |
| `/clear` | Clear current session |

#### Export Commands

| Command | Description |
|---------|-------------|
| `/export [filename]` | Export to markdown (default) |
| `/export report.pdf` | Export to PDF |
| `/export report.docx` | Export to Word document |
| `/excel [TICKER]` | Generate Excel financial model |
| `/excel compare T1 T2 T3` | Generate comparison spreadsheet |

#### Information Commands

| Command | Description |
|---------|-------------|
| `/sources` | Show all data sources used |
| `/cost` | Show token usage and estimated cost |
| `/format` | Re-display last response with formatting |
| `/help` | Show all available commands |
| `/quit` or `/exit` | Exit the REPL |

#### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save session |
| `Ctrl+L` | Clear screen |
| `Ctrl+E` | Export current research |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit |

## Frameworks

### Piotroski F-Score

A 9-point quantitative scoring system for financial health:

**Profitability (4 points)**
- Positive Return on Assets (ROA)
- Positive Operating Cash Flow
- ROA increasing year-over-year
- Cash Flow > Net Income (quality of earnings)

**Leverage & Liquidity (3 points)**
- Long-term debt decreasing
- Current ratio increasing
- No share dilution

**Efficiency (2 points)**
- Gross margin increasing
- Asset turnover increasing

**Scoring:** 0-2 = Weak, 3-5 = Average, 6-7 = Strong, 8-9 = Excellent

```bash
bullsh research NVDA --framework piotroski
```

### Porter's Five Forces

Qualitative competitive analysis:

- **Threat of New Entrants** — Barriers to entry
- **Supplier Power** — Bargaining power of suppliers
- **Buyer Power** — Bargaining power of customers
- **Threat of Substitutes** — Alternative products/services
- **Competitive Rivalry** — Industry competition intensity

```bash
bullsh research NVDA --framework porter
```

### Valuation Analysis

Multi-method price target generation:

- P/E Multiple vs Sector Average
- Forward P/E Valuation
- EV/EBITDA Multiple
- Analyst Consensus Targets
- Growth-Adjusted (PEG-based)

Outputs bear/base/bull case price targets.

```bash
bullsh research NVDA --framework valuation
```

### Custom Frameworks

Create your own frameworks in `~/.bullsh/frameworks/custom/`:

```toml
# ~/.bullsh/frameworks/custom/myframework.toml
[meta]
name = "My Custom Framework"
description = "Custom analysis criteria"
author = "Your Name"

[scoring]
enabled = true
pass_threshold = 7

[[criteria.items]]
id = "criterion_1"
name = "Revenue Growth"
question = "Is revenue growing >10% YoY?"
source = "sec"
scoring = "binary"

[[criteria.items]]
id = "criterion_2"
name = "Market Position"
question = "Is the company a market leader?"
source = "sec"
scoring = "scale"
```

Use with: `/framework custom:myframework`

## Configuration

### Config File

Located at `~/.bullsh/config.toml`:

```toml
[general]
verbosity = "full"          # "summary" or "full"
default_model = "claude-sonnet-4-20250514"
log_level = "info"

[display]
verbose_tools = false       # Show detailed tool output

[keybindings]
save_session = "ctrl+s"
clear_screen = "ctrl+l"
show_sources = "ctrl+o"
export_thesis = "ctrl+e"

[cost_controls]
max_tokens_per_session = 1000000
max_tokens_per_turn = 150000
warn_at_token_pct = 0.8
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Claude API key |
| `EDGAR_IDENTITY` | Yes | SEC compliance: `"Name email@example.com"` |
| `MODEL` | No | Override default Claude model |
| `LOG_LEVEL` | No | Logging level (debug, info, warning, error) |

## Architecture

```
bull.sh/
├── bullsh/                   # Main package
│   ├── agent/
│   │   ├── orchestrator.py   # Main agent loop with tool dispatch
│   │   ├── base.py           # SubAgent base class
│   │   ├── research.py       # Single-company research agent
│   │   └── compare.py        # Parallel comparison agent
│   ├── tools/
│   │   ├── base.py           # Tool definitions for Claude
│   │   ├── sec.py            # SEC EDGAR integration
│   │   ├── yahoo.py          # Yahoo Finance scraping
│   │   ├── social.py         # StockTwits, Reddit
│   │   ├── news.py           # DuckDuckGo news/web search
│   │   ├── rag.py            # Vector search over filings
│   │   ├── excel.py          # Excel spreadsheet generation
│   │   └── export.py         # PDF/DOCX export
│   ├── frameworks/
│   │   ├── base.py           # Framework definitions
│   │   └── valuation.py      # Valuation analysis
│   ├── storage/
│   │   ├── cache.py          # HTTP response caching
│   │   └── sessions.py       # Session persistence
│   ├── ui/
│   │   ├── repl.py           # Interactive REPL
│   │   ├── intro.py          # Animated intro sequence
│   │   ├── theme.py          # Color theme
│   │   ├── formatter.py      # Response formatting
│   │   └── status.py         # Tool status indicators
│   ├── cli.py                # Typer CLI entry point
│   ├── config.py             # Configuration management
│   └── logging.py            # Debug logging
└── static/                   # Static assets
    └── bullsh-banner.png
```

## API Costs

bullsh uses Claude's API. Estimated costs per research session:

| Task | Est. Tokens | Est. Cost |
|------|-------------|-----------|
| Single company research | ~20K-50K | $0.10-0.25 |
| Framework analysis | ~30K-80K | $0.15-0.40 |
| Multi-company comparison | ~60K-150K | $0.30-0.75 |

Token usage is displayed with `/cost` command. Prompt caching reduces repeat queries significantly.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Start for Contributors

```bash
# Clone and install dev dependencies
git clone https://github.com/aduria/bullsh.git
cd bullsh
pip install -e ".[dev,rag,export]"

# Run tests
pytest

# Run linter
ruff check src/

# Run type checker
mypy src/
```

### Areas We Need Help

- Additional data source integrations
- More analysis frameworks
- Improved RAG/vector search
- Test coverage
- Documentation
- Bug fixes and performance improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Anthropic](https://anthropic.com/) for Claude
- [edgartools](https://github.com/dgunning/edgartools) for SEC EDGAR access
- [yfinance](https://github.com/ranaroussi/yfinance) for Yahoo Finance data
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) for the REPL

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/aduria">Alexander Duria</a>
</p>
