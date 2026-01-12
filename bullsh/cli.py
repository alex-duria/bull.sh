"""CLI entry point using Typer."""

import logging
import os
import warnings

# Suppress HuggingFace progress bars and downloads (must be before any HF imports)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress noisy warnings from dependencies (must be before imports)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*")
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*")
warnings.filterwarnings("ignore", message=".*falling back to legacy parser.*")
warnings.filterwarnings("ignore", message=".*fallback will be removed.*")
warnings.filterwarnings("ignore", message=".*unclosed.*SSLSocket.*")
# HuggingFace/sentence-transformers warnings
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")
warnings.filterwarnings("ignore", message=".*hf_xet.*")
warnings.filterwarnings("ignore", message=".*Xet Storage.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress verbose logging from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("edgartools").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from bullsh import __version__
from bullsh.config import ConfigError, create_initial_env, load_config

app = typer.Typer(
    name="bullsh",
    help="Agentic CLI for investment research.",
    no_args_is_help=False,
)
console = Console()

# Subcommand groups
frameworks_app = typer.Typer(help="Manage analysis frameworks")
app.add_typer(frameworks_app, name="frameworks")


def version_callback(value: bool) -> None:
    if value:
        console.print(f"bullsh version {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            "-d",
            help="Enable debug logging to file",
        ),
    ] = False,
    debug_filter: Annotated[
        str | None,
        typer.Option(
            "--debug-filter",
            help="Filter debug logs: 'tools,api' or '!cache'",
        ),
    ] = None,
    no_intro: Annotated[
        bool,
        typer.Option(
            "--no-intro",
            help="Skip animated intro sequence",
        ),
    ] = False,
) -> None:
    """
    bullsh - Agentic CLI for investment research.

    Run without arguments to start interactive REPL.
    """
    # Store settings in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["debug_filter"] = debug_filter
    ctx.obj["no_intro"] = no_intro

    if ctx.invoked_subcommand is None:
        # No subcommand - start REPL
        _start_repl(debug=debug, debug_filter=debug_filter, skip_intro=no_intro)


def _start_repl(
    framework: str | None = None,
    debug: bool = False,
    debug_filter: str | None = None,
    skip_intro: bool = False,
) -> None:
    """Start the interactive REPL."""
    try:
        config = load_config()
    except ConfigError as e:
        # Check if it's a missing key error (first run) vs other config error
        error_msg = str(e)
        if "ANTHROPIC_API_KEY not found" in error_msg or "EDGAR_IDENTITY not found" in error_msg:
            _first_run_setup()
            config = load_config()
        else:
            # Other config error - show it and exit
            console.print(f"[red]Configuration error:[/red] {e}")
            raise typer.Exit(1)

    # Set up debug logging if enabled
    if debug:
        from bullsh.logging import setup_logging

        log_file = setup_logging(config.logs_dir, debug=True, debug_filter=debug_filter)
        console.print(f"[dim]Debug logging to: {log_file}[/dim]")

    # REPL shows its own welcome banner
    from bullsh.ui.repl import run_repl

    run_repl(config, framework=framework, skip_intro=skip_intro)


def _first_run_setup() -> None:
    """Interactive first-run setup to collect required config."""
    console.print(
        Panel.fit(
            "[bold]Welcome to bullsh![/bold]\n\nFirst-time setup required.",
            border_style="yellow",
        )
    )

    api_key = Prompt.ask(
        "\n[bold]ANTHROPIC_API_KEY[/bold]\n[dim]Get yours at https://console.anthropic.com/[/dim]"
    )

    edgar_identity = Prompt.ask(
        "\n[bold]EDGAR_IDENTITY[/bold] (name + email for SEC)\n"
        "[dim]Example: John Doe john@example.com[/dim]"
    )

    data_dir = Path.home() / ".bullsh"
    env_path = create_initial_env(data_dir, api_key, edgar_identity)

    console.print(f"\n[green]Configuration saved to {env_path}[/green]\n")


@app.command()
def research(
    ctx: typer.Context,
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g., NVDA)")],
    framework: Annotated[
        str | None,
        typer.Option(
            "--framework", "-f", help="Analysis framework (piotroski, porter, or custom:name)"
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save output to file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show detailed tool calls"),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Enter interactive mode after initial research"),
    ] = True,
) -> None:
    """Research a single company."""
    import asyncio

    from bullsh.agent.orchestrator import Orchestrator
    from bullsh.storage import get_session_manager

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    # Set up debug logging if enabled via --debug flag
    if ctx.obj and ctx.obj.get("debug"):
        from bullsh.logging import setup_logging

        log_file = setup_logging(
            config.logs_dir, debug=True, debug_filter=ctx.obj.get("debug_filter")
        )
        console.print(f"[dim]Debug logging to: {log_file}[/dim]")

    ticker = ticker.upper()
    console.print(f"[bold]Researching {ticker}[/bold]")
    if framework:
        console.print(f"[dim]Framework: {framework}[/dim]")
    console.print()

    # Create orchestrator
    orchestrator = Orchestrator(config, verbose=verbose)

    # Create session
    session_manager = get_session_manager()
    session = session_manager.create(tickers=[ticker], framework=framework)

    # Wire session for artifact tracking
    orchestrator.session = session

    # Run initial research query
    initial_prompt = f"Research {ticker} for me."
    if framework == "piotroski":
        initial_prompt += (
            " Apply the Piotroski F-Score framework and calculate each of the 9 signals."
        )
    elif framework == "porter":
        initial_prompt += (
            " Apply Porter's Five Forces framework and analyze each competitive force."
        )

    async def run_research() -> str:
        """Run the research and collect output."""
        output_text = ""
        console.print("[bold green]Agent[/bold green]")

        async for chunk in orchestrator.chat(initial_prompt, framework=framework):
            console.print(chunk, end="")
            output_text += chunk

        console.print()  # Final newline
        return output_text

    # Execute research
    try:
        result = asyncio.run(run_research())
    except Exception as e:
        console.print(f"\n[red]Error during research:[/red] {e}")
        raise typer.Exit(1)

    # Save session
    session.add_message("user", initial_prompt)
    session.add_message("assistant", result)
    session_manager.save(session)

    # Export to file if requested
    if output:
        output.write_text(result)
        console.print(f"\n[dim]Saved to {output}[/dim]")

    # Enter interactive mode if requested
    if interactive:
        console.print("\n[dim]Entering interactive mode. Type /exit to quit.[/dim]\n")
        from bullsh.ui.repl import run_repl_with_session

        run_repl_with_session(config, orchestrator, session, framework)


@app.command()
def compare(
    tickers: Annotated[
        list[str],
        typer.Argument(help="Stock tickers to compare (max 3)"),
    ],
    framework: Annotated[
        str | None,
        typer.Option("--framework", "-f", help="Analysis framework"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save output to file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show detailed tool calls"),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Enter interactive mode after comparison"),
    ] = True,
) -> None:
    """Compare multiple companies (max 3)."""
    import asyncio

    from bullsh.agent.orchestrator import Orchestrator
    from bullsh.storage import get_session_manager

    if len(tickers) > 3:
        console.print("[red]Error:[/red] Maximum 3 companies for comparison")
        raise typer.Exit(1)

    if len(tickers) < 2:
        console.print("[red]Error:[/red] Need at least 2 companies to compare")
        raise typer.Exit(1)

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    tickers_upper = [t.upper() for t in tickers]
    console.print(f"[bold]Comparing {', '.join(tickers_upper)}[/bold]")
    if framework:
        console.print(f"[dim]Framework: {framework}[/dim]")
    console.print()

    # Create orchestrator and session
    orchestrator = Orchestrator(config, verbose=verbose)
    session_manager = get_session_manager()
    session = session_manager.create(tickers=tickers_upper, framework=framework)

    # Wire session for artifact tracking
    orchestrator.session = session

    # Build comparison prompt
    ticker_list = ", ".join(tickers_upper)
    initial_prompt = f"Compare {ticker_list} for me. For each company, gather key financial data and analyst sentiment, then provide a side-by-side comparison highlighting:\n1. Valuation metrics (P/E, EV/EBITDA)\n2. Growth rates\n3. Financial health\n4. Analyst sentiment\n5. Key risks\n\nConclude with which company appears most attractive and why."

    if framework == "piotroski":
        initial_prompt += "\n\nCalculate the Piotroski F-Score for each company and compare their financial health."
    elif framework == "porter":
        initial_prompt += (
            "\n\nAnalyze Porter's Five Forces for each company's industry positioning."
        )

    async def run_compare() -> str:
        """Run the comparison and collect output."""
        output_text = ""
        console.print("[bold green]Agent[/bold green]")

        async for chunk in orchestrator.chat(initial_prompt, framework=framework):
            console.print(chunk, end="")
            output_text += chunk

        console.print()
        return output_text

    # Execute comparison
    try:
        result = asyncio.run(run_compare())
    except Exception as e:
        console.print(f"\n[red]Error during comparison:[/red] {e}")
        raise typer.Exit(1)

    # Save session
    session.add_message("user", initial_prompt)
    session.add_message("assistant", result)
    session_manager.save(session)

    # Export if requested
    if output:
        output.write_text(result)
        console.print(f"\n[dim]Saved to {output}[/dim]")

    # Enter interactive mode if requested
    if interactive:
        console.print("\n[dim]Entering interactive mode. Type /exit to quit.[/dim]\n")
        from bullsh.ui.repl import run_repl_with_session

        run_repl_with_session(config, orchestrator, session, framework)


@app.command()
def thesis(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save thesis to file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show detailed tool calls"),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Enter interactive mode after thesis generation"),
    ] = False,
) -> None:
    """Generate a full investment thesis (Hedge Fund Pitch format)."""
    import asyncio

    from bullsh.agent.orchestrator import Orchestrator
    from bullsh.storage import get_session_manager
    from bullsh.tools import thesis as thesis_tool

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    ticker = ticker.upper()
    console.print(f"[bold]Generating investment thesis for {ticker}[/bold]")
    console.print("[dim]Using Hedge Fund Pitch format[/dim]\n")

    # Create orchestrator and session
    orchestrator = Orchestrator(config, verbose=verbose)
    session_manager = get_session_manager()
    session = session_manager.create(tickers=[ticker], framework="pitch")

    # Wire session for artifact tracking
    orchestrator.session = session

    # Build thesis prompt
    initial_prompt = f"""Generate a complete investment thesis for {ticker} in the Hedge Fund Stock Pitch format.

Research the company thoroughly using SEC filings, analyst data, and sentiment. Then structure your thesis as follows:

## Investment Thesis
A compelling 2-3 sentence summary of why this is an attractive (or unattractive) investment opportunity.

## Company Overview
Brief description of what the company does, its market position, and key business segments.

## Key Catalysts
3-5 specific events or trends that could unlock value:
- Near-term catalysts (next 6-12 months)
- Medium-term catalysts (1-2 years)

## Valuation Analysis
- Current valuation metrics (P/E, EV/EBITDA, P/S)
- Comparison to peers and historical averages
- Assessment of whether the stock is undervalued/overvalued

## Financial Health
- Revenue and earnings trends
- Balance sheet strength
- Cash flow analysis
- Key financial ratios

## Risk Factors
The top 3-5 risks that could invalidate the thesis:
- Company-specific risks
- Industry risks
- Macro risks

## Conclusion
Final investment recommendation with price target range or expected return.

Be specific with numbers and cite your sources. This should read like a professional hedge fund pitch."""

    async def run_thesis() -> str:
        """Generate the thesis and collect output."""
        output_text = ""
        console.print("[bold green]Agent[/bold green]")

        async for chunk in orchestrator.chat(initial_prompt, framework="pitch"):
            console.print(chunk, end="")
            output_text += chunk

        console.print()
        return output_text

    # Execute thesis generation
    try:
        result = asyncio.run(run_thesis())
    except Exception as e:
        console.print(f"\n[red]Error generating thesis:[/red] {e}")
        raise typer.Exit(1)

    # Save session
    session.add_message("user", initial_prompt)
    session.add_message("assistant", result)
    session_manager.save(session)

    # Export to file
    output_path = output
    if not output_path:
        # Auto-generate filename
        from datetime import datetime

        date_str = datetime.now().strftime("%Y%m%d")
        output_path = Path(f"{ticker}_thesis_{date_str}.md")

    # Save thesis with provenance
    async def save_thesis():
        return await thesis_tool.save_thesis(ticker, result, str(output_path))

    save_result = asyncio.run(save_thesis())
    if save_result.data.get("path"):
        console.print(f"\n[green]Thesis saved to:[/green] {save_result.data['path']}")
    else:
        # Fallback: write directly
        output_path.write_text(result)
        console.print(f"\n[green]Thesis saved to:[/green] {output_path}")

    # Enter interactive mode if requested
    if interactive:
        console.print("\n[dim]Entering interactive mode. Type /exit to quit.[/dim]\n")
        from bullsh.ui.repl import run_repl_with_session

        run_repl_with_session(config, orchestrator, session, "pitch")


@app.command()
def debate(
    ctx: typer.Context,
    ticker: Annotated[str, typer.Argument(help="Stock ticker to debate")],
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Two rebuttal rounds (more thorough, ~40K tokens)"),
    ] = False,
    framework: Annotated[
        str | None,
        typer.Option("--framework", "-f", help="Include framework context (piotroski, porter)"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Export debate to file"),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Enter interactive mode after debate"),
    ] = False,
) -> None:
    """Run adversarial bull vs. bear debate on a stock."""
    import asyncio

    from bullsh.agent.debate import DebateCoordinator, DebateRefused
    from bullsh.storage import get_session_manager

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    # Set up debug logging if enabled
    if ctx.obj and ctx.obj.get("debug"):
        from bullsh.logging import setup_logging

        log_file = setup_logging(
            config.logs_dir, debug=True, debug_filter=ctx.obj.get("debug_filter")
        )
        console.print(f"[dim]Debug logging to: {log_file}[/dim]")

    ticker = ticker.upper()
    mode_str = "Deep" if deep else "Quick"
    console.print(f"[bold]Bull vs. Bear Debate: {ticker}[/bold]")
    console.print(f"[dim]Mode: {mode_str} | Framework: {framework or 'None'}[/dim]\n")

    # Get framework context if specified
    framework_context = None
    if framework:
        try:
            from bullsh.frameworks.base import load_framework

            fw = load_framework(framework)
            framework_context = (
                f"Framework: {fw.display_name}\nCriteria: {', '.join(c.name for c in fw.criteria)}"
            )
        except ValueError:
            console.print(
                f"[yellow]Warning: Framework '{framework}' not found, proceeding without it[/yellow]"
            )

    # Create debate coordinator
    coordinator = DebateCoordinator(
        config=config,
        ticker=ticker,
        deep_mode=deep,
        framework=framework,
        framework_context=framework_context,
    )

    # Create session for saving
    session_manager = get_session_manager()
    session = session_manager.create(tickers=[ticker], framework=framework)

    async def run_debate() -> str:
        """Run the debate and collect output."""
        output_text = ""

        try:
            async for chunk in coordinator.run():
                console.print(chunk, end="")
                output_text += chunk
        except DebateRefused as e:
            console.print(f"\n[red]{e}[/red]")
            raise typer.Exit(1)

        console.print()
        return output_text

    # Execute debate
    try:
        result = asyncio.run(run_debate())
    except Exception as e:
        console.print(f"\n[red]Debate failed:[/red] {e}")
        raise typer.Exit(1)

    # Save session
    session.add_message("user", f"Debate {ticker} ({mode_str} mode)")
    session.add_message("assistant", result)
    session_manager.save(session)

    # Show token usage
    console.print(f"\n[dim]Tokens used: {coordinator.state.tokens_used:,}[/dim]")

    # Export if requested
    if output:
        # Determine format from extension
        suffix = output.suffix.lower()
        if suffix == ".md" or suffix == "":
            output.write_text(result)
        else:
            # For other formats, just save as markdown for now
            output.write_text(result)
        console.print(f"[dim]Saved to {output}[/dim]")

    # Enter interactive mode if requested
    if interactive:
        console.print("\n[dim]Entering interactive mode. Type /exit to quit.[/dim]\n")
        from bullsh.agent.orchestrator import Orchestrator
        from bullsh.ui.repl import run_repl_with_session

        orchestrator = Orchestrator(config)
        orchestrator.session = session

        # Add debate context to orchestrator history
        from bullsh.agent.orchestrator import AgentMessage

        orchestrator.history.append(AgentMessage(role="user", content=f"Debate {ticker}"))
        orchestrator.history.append(AgentMessage(role="assistant", content=result))

        run_repl_with_session(config, orchestrator, session, framework)


@app.command()
def summary(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Save summary to file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show detailed tool calls"),
    ] = False,
) -> None:
    """Quick summary of a company."""
    import asyncio

    from bullsh.agent.orchestrator import Orchestrator

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    ticker = ticker.upper()
    console.print(f"[bold]Quick Summary: {ticker}[/bold]\n")

    orchestrator = Orchestrator(config, verbose=verbose)

    # Build a focused summary prompt
    prompt = f"""Give me a quick summary of {ticker}. Keep it brief and focused:

1. **What they do** (1-2 sentences)
2. **Current price and valuation** (P/E, market cap)
3. **Recent performance** (stock price trend, revenue growth)
4. **Analyst sentiment** (ratings, price targets)
5. **Key risk** (one main concern)

Use the available tools to gather current data. Keep the total response under 300 words."""

    async def run_summary() -> str:
        output_text = ""
        console.print("[bold green]Agent[/bold green]")

        async for chunk in orchestrator.chat(prompt):
            console.print(chunk, end="")
            output_text += chunk

        console.print()
        return output_text

    try:
        result = asyncio.run(run_summary())
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if output:
        output.write_text(result)
        console.print(f"\n[dim]Saved to {output}[/dim]")


@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(help="Session ID or name to resume")],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show detailed tool calls"),
    ] = False,
) -> None:
    """Resume a previous research session."""
    from bullsh.agent.orchestrator import Orchestrator
    from bullsh.storage import get_session_manager

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    session_manager = get_session_manager()

    # Find matching session
    sessions = session_manager.list_sessions(limit=50)
    match = None
    for s in sessions:
        if session_id in s["id"] or session_id.lower() in s["name"].lower():
            match = s["id"]
            break

    if not match:
        console.print(f"[red]Session not found: {session_id}[/red]")
        console.print("\n[dim]Available sessions:[/dim]")
        for s in sessions[:5]:
            console.print(f"  • {s['id'][-12:]} - {s['name']}")
        raise typer.Exit(1)

    try:
        session = session_manager.load(match)
    except ValueError as e:
        console.print(f"[red]Error loading session: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Resuming: {session.name}[/bold]")
    console.print(f"[dim]Tickers: {', '.join(session.tickers) or 'None'}[/dim]")
    console.print(f"[dim]Messages: {len(session.messages)}[/dim]")

    if session.framework:
        console.print(f"[dim]Framework: {session.framework}[/dim]")

    # Show last few messages for context
    if session.messages:
        console.print("\n[dim]Recent context:[/dim]")
        for msg in session.messages[-4:]:
            role_color = "cyan" if msg.role == "user" else "green"
            preview = msg.content[:100].replace("\n", " ")
            console.print(f"  [{role_color}]{msg.role}[/{role_color}]: {preview}...")

    console.print("\n[dim]Entering interactive mode. Type /exit to quit.[/dim]\n")

    # Create orchestrator and restore history
    orchestrator = Orchestrator(config, verbose=verbose)
    for msg in session.messages:
        from bullsh.agent.orchestrator import AgentMessage

        orchestrator.history.append(AgentMessage(role=msg.role, content=msg.content))

    # Wire session for artifact tracking
    orchestrator.session = session

    # Enter REPL
    from bullsh.ui.repl import run_repl_with_session

    run_repl_with_session(config, orchestrator, session, session.framework)


# Framework subcommands
@frameworks_app.command("list")
def frameworks_list() -> None:
    """List available analysis frameworks."""
    from rich.table import Table

    from bullsh.frameworks.base import list_frameworks

    frameworks = list_frameworks()

    table = Table(title="Available Frameworks")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Type")
    table.add_column("Source")

    for fw in frameworks:
        source = "builtin" if fw.get("builtin") else "custom"
        table.add_row(
            fw["name"],
            fw["description"],
            fw["type"],
            source,
        )

    console.print(table)

    if not any(not fw.get("builtin") for fw in frameworks):
        console.print(
            "\n[dim]No custom frameworks. Create one with: bullsh frameworks create[/dim]"
        )


@frameworks_app.command("show")
def frameworks_show(
    name: Annotated[str, typer.Argument(help="Framework name")],
) -> None:
    """Show details of a framework."""
    from rich.table import Table

    from bullsh.frameworks.base import load_framework

    try:
        fw = load_framework(name)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]{fw.display_name}[/bold]")
    console.print(f"[dim]{fw.description}[/dim]\n")
    console.print(f"Type: {fw.framework_type.value}")
    console.print(f"Scoring: {'Enabled' if fw.scoring_enabled else 'Disabled'}")
    if fw.pass_threshold:
        console.print(f"Pass threshold: {fw.pass_threshold}/{len(fw.criteria)}")

    console.print(f"\n[bold]Criteria ({len(fw.criteria)}):[/bold]")

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Question")
    table.add_column("Source")

    for c in fw.criteria:
        table.add_row(
            c.id, c.name, c.question[:50] + "..." if len(c.question) > 50 else c.question, c.source
        )

    console.print(table)


@frameworks_app.command("create")
def frameworks_create() -> None:
    """Create a new custom framework interactively."""
    import tomli_w
    from rich.table import Table

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    console.print("[bold]Create Custom Framework[/bold]\n")
    console.print("[dim]This wizard will help you create a custom analysis framework.[/dim]\n")

    # Get framework name
    name = Prompt.ask("Framework name (lowercase, no spaces)", default="my_framework")
    name = name.lower().replace(" ", "_")

    display_name = Prompt.ask("Display name", default=name.replace("_", " ").title())
    description = Prompt.ask("Description", default="Custom analysis framework")

    # Scoring
    enable_scoring = Prompt.ask("Enable scoring?", choices=["y", "n"], default="n") == "y"
    pass_threshold = None
    if enable_scoring:
        pass_threshold = int(Prompt.ask("Pass threshold (score needed to pass)", default="7"))

    # Criteria
    console.print("\n[bold]Add Criteria[/bold]")
    console.print("[dim]Enter criteria one at a time. Type 'done' when finished.[/dim]\n")

    criteria = []
    while True:
        criterion_name = Prompt.ask("Criterion name (or 'done' to finish)")
        if criterion_name.lower() == "done":
            if not criteria:
                console.print("[yellow]You need at least one criterion![/yellow]")
                continue
            break

        criterion_question = Prompt.ask("Question to answer")
        criterion_source = Prompt.ask(
            "Data source", choices=["sec", "yahoo", "social", "synthesis"], default="sec"
        )

        criteria.append(
            {
                "id": criterion_name.lower().replace(" ", "_"),
                "name": criterion_name,
                "question": criterion_question,
                "source": criterion_source,
            }
        )

        console.print(f"[green]Added:[/green] {criterion_name}\n")

    # Show preview
    console.print("\n[bold]Framework Preview[/bold]")
    table = Table()
    table.add_column("Criterion")
    table.add_column("Question")
    table.add_column("Source")

    for c in criteria:
        table.add_row(c["name"], c["question"][:40] + "...", c["source"])

    console.print(table)

    # Confirm
    if Prompt.ask("\nSave this framework?", choices=["y", "n"], default="y") != "y":
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Build TOML structure
    framework_data = {
        "meta": {
            "name": display_name,
            "description": description,
            "author": "user",
        },
        "scoring": {
            "enabled": enable_scoring,
        },
        "criteria": {
            "items": criteria,
        },
    }

    if pass_threshold:
        framework_data["scoring"]["pass_threshold"] = pass_threshold

    # Save to file
    framework_path = config.custom_frameworks_dir / f"{name}.toml"
    config.custom_frameworks_dir.mkdir(parents=True, exist_ok=True)

    with open(framework_path, "wb") as f:
        tomli_w.dump(framework_data, f)

    console.print(f"\n[green]Framework saved to:[/green] {framework_path}")
    console.print(f"\n[dim]Use it with: bullsh research TICKER --framework custom:{name}[/dim]")


@frameworks_app.command("edit")
def frameworks_edit(
    name: Annotated[str, typer.Argument(help="Framework name to edit")],
) -> None:
    """Edit an existing custom framework."""
    import os
    import subprocess
    import sys

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    # Only allow editing custom frameworks
    if not name.startswith("custom:"):
        console.print("[red]Can only edit custom frameworks.[/red]")
        console.print(f"[dim]Use: bullsh frameworks edit custom:{name}[/dim]")
        raise typer.Exit(1)

    framework_name = name[7:]  # Remove "custom:" prefix
    framework_path = config.custom_frameworks_dir / f"{framework_name}.toml"

    if not framework_path.exists():
        console.print(f"[red]Framework not found:[/red] {framework_path}")
        console.print("[dim]Create it first with: bullsh frameworks create[/dim]")
        raise typer.Exit(1)

    # Determine editor
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL"))

    if not editor:
        # Try common editors
        for e in ["code", "vim", "nano", "notepad"]:
            if (
                subprocess.run(
                    ["which", e] if sys.platform != "win32" else ["where", e], capture_output=True
                ).returncode
                == 0
            ):
                editor = e
                break

    if not editor:
        console.print("[yellow]No editor found. Edit manually:[/yellow]")
        console.print(f"  {framework_path}")
        return

    console.print(f"[dim]Opening {framework_path} in {editor}...[/dim]")

    try:
        subprocess.run([editor, str(framework_path)])
        console.print("[green]Framework updated.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to open editor:[/red] {e}")
        console.print(f"[dim]Edit manually: {framework_path}[/dim]")


if __name__ == "__main__":
    app()
