"""Interactive REPL for bullsh - the primary interface."""

import asyncio
import atexit
import gc
import re
import shlex
import sys
import warnings

# Suppress SSL socket warnings on exit
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*unclosed.*SSLSocket.*")
warnings.filterwarnings("ignore", message=".*unclosed.*socket.*")

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bullsh.config import Config
from bullsh.agent.orchestrator import Orchestrator, TokenLimitExceeded
from bullsh.storage import (
    Session,
    get_cache,
    get_session_manager,
)
from bullsh.ui.theme import RICH_THEME, COLORS


def _cleanup_on_exit():
    """Clean up resources on exit to avoid warnings."""
    gc.collect()


atexit.register(_cleanup_on_exit)

# Use themed console
console = Console(theme=RICH_THEME)


# Keybinding action flags (set by keybindings, checked in loop)
class KeybindingAction:
    """Store keybinding action to execute."""
    action: str | None = None

_kb_action = KeybindingAction()


def _create_keybindings() -> KeyBindings:
    """Create prompt_toolkit keybindings for the REPL."""
    kb = KeyBindings()

    @kb.add('c-s')
    def _(event):
        """Ctrl+S: Save session."""
        _kb_action.action = "save"
        event.app.exit(result="")

    @kb.add('c-l')
    def _(event):
        """Ctrl+L: Clear screen."""
        _kb_action.action = "clear"
        event.app.exit(result="")

    @kb.add('c-e')
    def _(event):
        """Ctrl+E: Export."""
        _kb_action.action = "export"
        event.app.exit(result="")

    return kb


class BullshCompleter(Completer):
    """Custom completer for bullsh commands."""

    # Slash commands with descriptions
    SLASH_COMMANDS = [
        ("/help", "Show help and available commands"),
        ("/save", "Save current session"),
        ("/sessions", "List saved sessions"),
        ("/resume", "Resume a previous session"),
        ("/export", "Export research (md/pdf/docx)"),
        ("/excel", "Generate Excel financial model"),
        ("/format", "Re-display last response beautifully formatted"),
        ("/framework", "Switch analysis framework"),
        ("/framework piotroski", "Use Piotroski F-Score (9-point financial health)"),
        ("/framework porter", "Use Porter's Five Forces (competitive analysis)"),
        ("/framework pitch", "Use Hedge Fund Stock Pitch format"),
        ("/framework valuation", "Use Valuation Analysis (price targets)"),
        ("/framework off", "Return to freestyle mode"),
        ("/checklist", "Show framework progress checklist"),
        ("/cache", "Show cache statistics"),
        ("/cache clear", "Clear all cached data"),
        ("/rag", "Show RAG vector database stats"),
        ("/rag list", "List indexed SEC filings"),
        ("/sources", "Show data sources used this session"),
        ("/usage", "Show token usage and cost"),
        ("/config", "Show current configuration"),
        ("/exit", "Exit bullsh"),
    ]

    # Top-level commands with descriptions
    COMMANDS = [
        ("research", "Research a single company (e.g., research NVDA)"),
        ("compare", "Compare 2-3 companies (e.g., compare AAPL MSFT)"),
        ("thesis", "Generate investment thesis (e.g., thesis TSLA)"),
        ("summary", "Quick company overview (e.g., summary GOOGL)"),
        ("frameworks", "List available analysis frameworks"),
    ]

    # Framework suggestions
    FRAMEWORKS = [
        ("piotroski", "9-point quantitative financial health score"),
        ("porter", "Porter's Five Forces competitive analysis"),
        ("pitch", "Hedge Fund Stock Pitch thesis format"),
        ("valuation", "Multi-method price target generation"),
    ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        word = document.get_word_before_cursor()

        # Slash commands - show when typing /
        if text.startswith("/"):
            typed = text.lower()
            for cmd, desc in self.SLASH_COMMANDS:
                if cmd.lower().startswith(typed):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )

        # Top-level commands - show at start of input
        elif not text or text == word:
            typed = text.lower()
            for cmd, desc in self.COMMANDS:
                if cmd.startswith(typed):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )
            # Also show slash commands at empty prompt
            if not text:
                for cmd, desc in self.SLASH_COMMANDS[:5]:  # Top 5 slash commands
                    yield Completion(
                        cmd,
                        start_position=0,
                        display_meta=desc,
                    )

        # Framework suggestions after -f or --framework
        elif text.endswith("-f ") or text.endswith("--framework "):
            for fw, desc in self.FRAMEWORKS:
                yield Completion(
                    fw,
                    start_position=0,
                    display_meta=desc,
                )

        # After "research", "thesis", "summary" - could suggest recent tickers
        # For now, just show placeholder
        elif any(text.lower().startswith(cmd + " ") for cmd in ["research", "thesis", "summary"]):
            # Show framework flag as option
            if "-f" not in text and "--framework" not in text:
                yield Completion(
                    "-f piotroski",
                    start_position=0,
                    display_meta="Add Piotroski F-Score analysis",
                )
                yield Completion(
                    "-f porter",
                    start_position=0,
                    display_meta="Add Porter's Five Forces analysis",
                )


def _get_prompt_session(config: Config) -> PromptSession:
    """Create a prompt_toolkit session with history, keybindings, and completion."""
    history_file = config.data_dir / ".history"

    # Use theme colors for prompt styling
    style = Style.from_dict({
        'prompt': f'bold {COLORS["primary"]}',
        'completion-menu': f'bg:{COLORS["bg_panel"]} {COLORS["text"]}',
        'completion-menu.completion': f'bg:{COLORS["bg_panel"]} {COLORS["text"]}',
        'completion-menu.completion.current': f'bg:{COLORS["bg_highlight"]} {COLORS["text_bright"]} bold',
        'completion-menu.meta': f'bg:{COLORS["bg_panel"]} {COLORS["text_muted"]} italic',
        'completion-menu.meta.current': f'bg:{COLORS["bg_highlight"]} {COLORS["secondary"]} italic',
    })

    return PromptSession(
        history=FileHistory(str(history_file)),
        key_bindings=_create_keybindings(),
        completer=BullshCompleter(),
        complete_while_typing=False,  # Only complete on Tab
        style=style,
    )


def run_repl(config: Config, framework: str | None = None, skip_intro: bool = False) -> None:
    """Run the interactive REPL loop."""
    asyncio.run(_async_repl(config, framework, skip_intro))


def run_repl_with_session(
    config: Config,
    orchestrator: Orchestrator,
    session: Session,
    framework: str | None = None,
) -> None:
    """Run the REPL with an existing session and orchestrator."""
    asyncio.run(_async_repl_with_session(config, orchestrator, session, framework))


async def _async_repl(config: Config, framework: str | None = None, skip_intro: bool = False) -> None:
    """Async REPL implementation."""
    orchestrator = Orchestrator(config)
    session_manager = get_session_manager()
    session = session_manager.create(framework=framework)

    # Play animated intro (unless skipped)
    if not skip_intro:
        await _play_intro()

    # Show welcome banner
    _show_welcome(framework)

    await _repl_loop(config, orchestrator, session, framework)


async def _play_intro(skip: bool = False) -> None:
    """Play the animated intro sequence."""
    if skip:
        return

    try:
        from bullsh.ui.intro import play_intro_animation
        await play_intro_animation(duration=4.0)
    except Exception:
        # Fallback if animation fails
        pass


def _show_welcome(framework: str | None = None, compact: bool = False) -> None:
    """Display welcome banner."""
    from bullsh.ui.intro import render_logo, TAGLINE, THEME

    if not compact:
        # Show logo
        console.print()
        logo = render_logo("large" if console.width and console.width > 80 else "small")
        console.print(logo, justify="center")
        console.print()
        console.print(f"[italic {THEME['secondary']}]{TAGLINE}[/]", justify="center")
        console.print(f"[dim]SEC filings • Market data • AI analysis[/dim]", justify="center")
        console.print(f"[{THEME['muted']}]{'─' * 40}[/]", justify="center")
        console.print(f"[dim]Made with [red]❤[/red] by Alexander Duria[/dim]", justify="center")
        console.print()

    # Compact command hints
    hints = f"""[bold]Quick Start:[/bold]
  [cyan]research TICKER[/cyan]    Research a company      [dim]e.g. research NVDA[/dim]
  [cyan]compare T1 T2[/cyan]      Compare companies       [dim]e.g. compare AAPL MSFT[/dim]
  [cyan]/framework NAME[/cyan]   Use analysis framework  [dim]piotroski, porter, valuation[/dim]

[dim]Tab for suggestions • /help for all commands • Ctrl+C to exit[/dim]
"""
    console.print(hints)

    if framework:
        console.print(f"[dim]Framework: {framework}[/dim]")

    # Ready indicator
    console.print(f"[bold {THEME['primary']}]✓[/] Ready\n")


async def _async_repl_with_session(
    config: Config,
    orchestrator: Orchestrator,
    session: Session,
    framework: str | None = None,
) -> None:
    """Async REPL with existing session."""
    await _repl_loop(config, orchestrator, session, framework)


async def _repl_loop(
    config: Config,
    orchestrator: Orchestrator,
    session: Session,
    framework: str | None = None,
) -> None:
    """Core REPL loop."""
    current_framework = framework
    session_manager = get_session_manager()
    prompt_session = _get_prompt_session(config)

    while True:
        try:
            # Reset keybinding action
            _kb_action.action = None

            # Get user input with prompt_toolkit
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: prompt_session.prompt("\n> "),
            )

            # Handle keybinding actions
            if _kb_action.action:
                action = _kb_action.action
                _kb_action.action = None

                if action == "save":
                    session_manager.save(session)
                    console.print(f"[green]✓ Session saved:[/green] {session.name}")
                    continue
                elif action == "clear":
                    console.clear()
                    _show_welcome(current_framework)
                    continue
                elif action == "export":
                    # Trigger export command
                    user_input = "/export"

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            # Save session before exit
            session_manager.save(session)
            break

        # Handle empty input
        if not user_input.strip():
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            result = await _handle_slash_command(
                user_input,
                orchestrator,
                session,
                current_framework,
                config,
            )
            if result == "exit":
                session_manager.save(session)
                break
            if result and result.startswith("framework:"):
                current_framework = result.split(":", 1)[1] or None
                session.framework = current_framework
            continue

        # Check if input is a built-in command
        command_result = await _handle_command(
            user_input,
            orchestrator,
            session,
            current_framework,
            config,
        )
        if command_result is not None:
            if command_result.startswith("framework:"):
                current_framework = command_result.split(":", 1)[1] or None
                session.framework = current_framework
            continue

        # Not a command - process as natural language through agent
        console.print()  # Spacing

        try:
            response_text = ""

            # Stream response - show tool calls and progress live
            async for chunk in orchestrator.chat(user_input, framework=current_framework):
                console.print(chunk, end="")
                response_text += chunk

            console.print()  # Final newline after streaming

            # Add to session
            session.add_message("user", user_input)
            session.add_message("assistant", response_text)

            # Auto-save periodically
            if len(session.messages) % 4 == 0:
                session_manager.save(session)

        except TokenLimitExceeded as e:
            console.print(f"\n[bold red]Token limit exceeded:[/bold red] {e}")
            console.print("[dim]Use /usage to see current usage. Start a new session to continue.[/dim]")
            session_manager.save(session)
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")


async def _handle_command(
    user_input: str,
    orchestrator: Orchestrator,
    session: Session,
    current_framework: str | None,
    config: Config,
) -> str | None:
    """
    Parse and handle built-in commands like 'research NVDA'.

    Returns:
        - None if not a recognized command (pass to agent)
        - "handled" if command was executed
        - "framework:<name>" if framework changed
    """
    # Parse input into tokens
    try:
        tokens = shlex.split(user_input)
    except ValueError:
        tokens = user_input.split()

    if not tokens:
        return None

    cmd = tokens[0].lower()
    args = tokens[1:]

    # Parse common flags
    framework_override = current_framework
    verbose = False
    remaining_args = []

    i = 0
    while i < len(args):
        if args[i] in ("--framework", "-f") and i + 1 < len(args):
            framework_override = args[i + 1]
            i += 2
        elif args[i] == "--verbose":
            verbose = True
            i += 1
        else:
            remaining_args.append(args[i])
            i += 1

    # Handle commands
    match cmd:
        case "research":
            if not remaining_args:
                console.print("[red]Usage: research <TICKER>[/red]")
                return "handled"
            ticker = remaining_args[0].upper()
            await _run_research(orchestrator, session, ticker, framework_override, config)
            return "handled"

        case "compare":
            if len(remaining_args) < 2:
                console.print("[red]Usage: compare <TICKER1> <TICKER2> [TICKER3][/red]")
                return "handled"
            if len(remaining_args) > 3:
                console.print("[red]Maximum 3 tickers for comparison[/red]")
                return "handled"
            tickers = [t.upper() for t in remaining_args[:3]]
            await _run_compare(orchestrator, session, tickers, framework_override, config)
            return "handled"

        case "thesis":
            if not remaining_args:
                console.print("[red]Usage: thesis <TICKER>[/red]")
                return "handled"
            ticker = remaining_args[0].upper()
            await _run_thesis(orchestrator, session, ticker, config)
            return "handled"

        case "summary":
            if not remaining_args:
                console.print("[red]Usage: summary <TICKER>[/red]")
                return "handled"
            ticker = remaining_args[0].upper()
            await _run_summary(orchestrator, session, ticker, config)
            return "handled"

        case "frameworks":
            await _handle_frameworks_command(remaining_args, config)
            return "handled"

        case "help":
            _show_help()
            return "handled"

        case "exit" | "quit" | "q":
            console.print("[dim]Goodbye![/dim]")
            get_session_manager().save(session)
            import sys
            sys.exit(0)

        case _:
            # Not a recognized command - let agent handle it
            return None


async def _run_research(
    orchestrator: Orchestrator,
    session: Session,
    ticker: str,
    framework: str | None,
    config: Config,
) -> None:
    """Execute research command."""
    console.print(f"[bold]Researching {ticker}[/bold]")
    if framework:
        console.print(f"[dim]Framework: {framework}[/dim]")
    console.print()

    # Add ticker to session
    if ticker not in session.tickers:
        session.tickers.append(ticker)

    # Build prompt
    prompt = f"Research {ticker} for me."
    if framework == "piotroski":
        prompt += " Apply the Piotroski F-Score framework and calculate each of the 9 signals."
    elif framework == "porter":
        prompt += " Apply Porter's Five Forces framework and analyze each competitive force."

    await _execute_agent_query(orchestrator, session, prompt, framework)


async def _run_compare(
    orchestrator: Orchestrator,
    session: Session,
    tickers: list[str],
    framework: str | None,
    config: Config,
) -> None:
    """Execute compare command."""
    ticker_list = ", ".join(tickers)
    console.print(f"[bold]Comparing {ticker_list}[/bold]")
    if framework:
        console.print(f"[dim]Framework: {framework}[/dim]")
    console.print()

    # Add tickers to session
    for t in tickers:
        if t not in session.tickers:
            session.tickers.append(t)

    # Build prompt
    prompt = f"""Compare {ticker_list} for me. For each company, gather key financial data and analyst sentiment, then provide a side-by-side comparison highlighting:
1. Valuation metrics (P/E, EV/EBITDA)
2. Growth rates
3. Financial health
4. Analyst sentiment
5. Key risks

Conclude with which company appears most attractive and why."""

    if framework == "piotroski":
        prompt += "\n\nCalculate the Piotroski F-Score for each company."
    elif framework == "porter":
        prompt += "\n\nAnalyze Porter's Five Forces for each company."

    await _execute_agent_query(orchestrator, session, prompt, framework)


async def _run_thesis(
    orchestrator: Orchestrator,
    session: Session,
    ticker: str,
    config: Config,
) -> None:
    """Execute thesis command."""
    console.print(f"[bold]Generating thesis for {ticker}[/bold]")
    console.print("[dim]Using Hedge Fund Pitch format[/dim]\n")

    if ticker not in session.tickers:
        session.tickers.append(ticker)

    prompt = f"""Generate a complete investment thesis for {ticker} in the Hedge Fund Stock Pitch format.

Research the company thoroughly using SEC filings, analyst data, and sentiment. Then structure your thesis as follows:

## Investment Thesis
A compelling 2-3 sentence summary of why this is an attractive (or unattractive) investment opportunity.

## Company Overview
Brief description of what the company does, its market position, and key business segments.

## Key Catalysts
3-5 specific events or trends that could unlock value.

## Valuation Analysis
Current valuation metrics and assessment of whether the stock is mispriced.

## Financial Health
Revenue trends, balance sheet strength, cash flow analysis.

## Risk Factors
Top 3-5 risks that could invalidate the thesis.

## Conclusion
Final investment assessment with expected return potential.

Be specific with numbers and cite your sources."""

    await _execute_agent_query(orchestrator, session, prompt, "pitch")


async def _run_summary(
    orchestrator: Orchestrator,
    session: Session,
    ticker: str,
    config: Config,
) -> None:
    """Execute summary command."""
    console.print(f"[bold]Quick Summary: {ticker}[/bold]\n")

    if ticker not in session.tickers:
        session.tickers.append(ticker)

    prompt = f"""Give me a quick summary of {ticker}. Keep it brief:

1. **What they do** (1-2 sentences)
2. **Current price and valuation** (P/E, market cap)
3. **Recent performance** (stock trend, revenue growth)
4. **Analyst sentiment** (ratings, price targets)
5. **Key risk** (one main concern)

Keep the response under 300 words."""

    await _execute_agent_query(orchestrator, session, prompt, None)


async def _execute_agent_query(
    orchestrator: Orchestrator,
    session: Session,
    prompt: str,
    framework: str | None,
) -> None:
    """Execute a query through the agent and save to session."""
    session_manager = get_session_manager()

    try:
        response_text = ""

        # Stream response - show tool calls and progress live
        async for chunk in orchestrator.chat(prompt, framework=framework):
            console.print(chunk, end="")
            response_text += chunk

        console.print()  # Final newline

        # Save to session
        session.add_message("user", prompt)
        session.add_message("assistant", response_text)
        session_manager.save(session)

    except TokenLimitExceeded as e:
        console.print(f"\n[bold red]Token limit exceeded:[/bold red] {e}")
        console.print("[dim]Use /usage to see current usage. Start a new session to continue.[/dim]")
        session_manager.save(session)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")


async def _handle_frameworks_command(args: list[str], config: Config) -> None:
    """Handle 'frameworks' subcommands."""
    from bullsh.frameworks.base import list_frameworks, load_framework

    if not args:
        args = ["list"]

    subcmd = args[0].lower()

    match subcmd:
        case "list":
            frameworks = list_frameworks()
            table = Table(title="Available Frameworks")
            table.add_column("Name", style="cyan")
            table.add_column("Description")
            table.add_column("Type")

            for fw in frameworks:
                table.add_row(fw["name"], fw["description"], fw["type"])

            console.print(table)

        case "show":
            if len(args) < 2:
                console.print("[red]Usage: frameworks show <name>[/red]")
                return
            try:
                fw = load_framework(args[1])
                console.print(f"[bold]{fw.display_name}[/bold]")
                console.print(f"[dim]{fw.description}[/dim]\n")
                console.print(f"Criteria: {len(fw.criteria)}")
                console.print(f"Scoring: {'Enabled' if fw.scoring_enabled else 'Disabled'}")
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")

        case _:
            console.print(f"[red]Unknown subcommand: {subcmd}[/red]")
            console.print("[dim]Available: list, show[/dim]")


async def _handle_slash_command(
    command: str,
    orchestrator: Orchestrator,
    session: Session,
    current_framework: str | None,
    config: Config,
) -> str | None:
    """
    Handle slash commands.

    Returns:
        - "exit" to exit the REPL
        - "framework:<name>" to change framework
        - None otherwise
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    session_manager = get_session_manager()
    cache = get_cache()

    match cmd:
        case "/help":
            _show_help()
            return None

        case "/exit" | "/quit" | "/q":
            console.print("[dim]Goodbye![/dim]")
            return "exit"

        case "/save":
            session_manager.save(session)
            console.print(f"[green]Session saved:[/green] {session.name} ({session.id})")
            return None

        case "/sessions":
            sessions = session_manager.list_sessions(limit=10)
            if not sessions:
                console.print("[dim]No saved sessions[/dim]")
            else:
                table = Table(title="Recent Sessions")
                table.add_column("ID", style="cyan")
                table.add_column("Name")
                table.add_column("Tickers")
                table.add_column("Msgs")
                table.add_column("Summary", style="dim")
                table.add_column("Updated")
                for s in sessions:
                    summary = s.get("summary", "")[:40] + "..." if s.get("summary") else ""
                    table.add_row(
                        s["id"][-12:],
                        s["name"][:25],
                        ", ".join(s["tickers"][:2]),
                        str(s["message_count"]),
                        summary,
                        s["updated_at"][:10],
                    )
                console.print(table)
            return None

        case "/resume":
            if not args:
                console.print("[red]Usage: /resume <session_id>[/red]")
                return None
            try:
                # Find matching session
                sessions = session_manager.list_sessions(limit=50)
                match = None
                for s in sessions:
                    if args in s["id"] or args.lower() in s["name"].lower():
                        match = s["id"]
                        break

                if not match:
                    console.print(f"[red]Session not found: {args}[/red]")
                    return None

                loaded = session_manager.load(match)
                console.print(f"[green]Loaded session:[/green] {loaded.name}")
                console.print(f"[dim]{len(loaded.messages)} messages, tickers: {', '.join(loaded.tickers)}[/dim]")
                # Note: Full resume would require reinitializing orchestrator with history
                return None
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                return None

        case "/cache":
            subparts = args.split(maxsplit=1)
            subcmd = subparts[0] if subparts else "stats"

            match subcmd:
                case "stats" | "":
                    stats = cache.get_stats()
                    console.print("[bold]Cache Statistics[/bold]")
                    console.print(f"  Total entries: {stats['total_entries']}")
                    console.print(f"  Expired: {stats['expired_entries']}")
                    console.print(f"  Size: {stats['total_size_mb']} MB")
                    if stats['by_source']:
                        console.print("  By source:")
                        for src, count in stats['by_source'].items():
                            console.print(f"    {src}: {count}")
                case "list":
                    entries = cache.list_entries()[:10]
                    if not entries:
                        console.print("[dim]Cache is empty[/dim]")
                    else:
                        table = Table(title="Cached Data")
                        table.add_column("Source")
                        table.add_column("Ticker")
                        table.add_column("Cached")
                        table.add_column("Hits")
                        table.add_column("Status")
                        for e in entries:
                            status = "[red]Expired[/red]" if e["expired"] else "[green]Valid[/green]"
                            table.add_row(
                                e["source"],
                                e["ticker"] or "-",
                                e["created_at"][:16],
                                str(e["hit_count"]),
                                status,
                            )
                        console.print(table)
                case "clear":
                    count = cache.clear()
                    console.print(f"[green]Cleared {count} cache entries[/green]")
                case "refresh":
                    ticker = subparts[1] if len(subparts) > 1 else ""
                    if not ticker:
                        console.print("[red]Usage: /cache refresh <ticker>[/red]")
                    else:
                        count = cache.invalidate_ticker(ticker.upper())
                        console.print(f"[green]Invalidated {count} entries for {ticker.upper()}[/green]")
                case _:
                    console.print(f"[red]Unknown cache command: {subcmd}[/red]")
                    console.print("[dim]Available: stats, list, clear, refresh <ticker>[/dim]")
            return None

        case "/rag":
            subparts = args.split(maxsplit=1)
            subcmd = subparts[0] if subparts else "stats"

            try:
                from bullsh.storage.vectordb import get_vectordb
                vectordb = get_vectordb()
            except ImportError:
                console.print("[red]RAG not available. Install with: pip install bullsh[rag][/red]")
                return None

            match subcmd:
                case "stats" | "":
                    stats = vectordb.get_stats()
                    console.print("[bold]RAG Vector Database[/bold]")
                    console.print(f"  Total chunks: {stats['total_chunks']}")
                    console.print(f"  Indexed filings: {stats['unique_filings']}")
                    console.print(f"  Unique tickers: {stats['unique_tickers']}")
                    console.print(f"  Database: {stats['db_path']}")
                case "list":
                    ticker_filter = subparts[1] if len(subparts) > 1 else None
                    indexed = vectordb.list_indexed(ticker=ticker_filter)
                    if not indexed:
                        console.print("[dim]No filings indexed[/dim]")
                    else:
                        table = Table(title="Indexed Filings")
                        table.add_column("Ticker")
                        table.add_column("Form")
                        table.add_column("Year")
                        table.add_column("Chunks")
                        for f in indexed[:20]:
                            table.add_row(
                                f["ticker"],
                                f["form"],
                                str(f["year"]),
                                str(f["chunks"]),
                            )
                        console.print(table)
                case "clear":
                    deleted = vectordb.clear()
                    console.print(f"[green]Cleared vector database: {deleted} chunks deleted[/green]")
                    console.print("[dim]Filings will be re-indexed on next sec_fetch[/dim]")
                case _:
                    console.print(f"[red]Unknown rag command: {subcmd}[/red]")
                    console.print("[dim]Available: stats, list [ticker], clear[/dim]")
            return None

        case "/sources":
            if not session.tickers:
                console.print("[dim]No tickers researched in this session[/dim]")
            else:
                console.print("[bold]Data sources used this session:[/bold]")
                for ticker in session.tickers:
                    entries = cache.list_entries(ticker=ticker)
                    console.print(f"\n  [cyan]{ticker}[/cyan]")
                    for e in entries:
                        console.print(f"    • {e['source']} (cached {e['created_at'][:10]})")
            return None

        case "/export":
            from bullsh.tools import export as export_tool

            if not session.messages:
                console.print("[red]No conversation to export[/red]")
                return None

            # Aggregate ALL assistant messages into a structured document
            export_content = _build_session_export(session, current_framework)

            if not export_content.strip():
                console.print("[red]No research to export[/red]")
                return None

            # Parse args: /export [filename] [--format pdf|docx|md]
            parts = args.split() if args else []
            filename = None
            format_type = "md"

            for i, part in enumerate(parts):
                if part == "--format" and i + 1 < len(parts):
                    format_type = parts[i + 1]
                elif not part.startswith("--"):
                    filename = part

            # Auto-detect format from filename extension
            if filename:
                if filename.endswith('.pdf'):
                    format_type = "pdf"
                elif filename.endswith('.docx') or filename.endswith('.doc'):
                    format_type = "docx"
                elif filename.endswith('.md'):
                    format_type = "md"

            ticker_str = "_".join(session.tickers[:3]) if session.tickers else "Research"
            title = f"{ticker_str} Investment Research"

            console.print(f"[dim]Exporting session ({len(session.messages)} messages) to {format_type.upper()}...[/dim]")
            result = await export_tool.export_content(
                export_content,
                filename=filename,
                format=format_type,
                title=title,
            )

            if result.status.value == "success":
                console.print(f"[green]✓ Exported to:[/green] {result.data.get('path')}")
            else:
                console.print(f"[red]✗ Export failed:[/red] {result.error_message}")
            return None

        case "/config":
            _show_config(config)
            return None

        case "/format":
            # Re-display the last assistant response with beautiful formatting
            last_response = None
            for msg in reversed(session.messages):
                if msg.role == "assistant" and msg.content.strip():
                    last_response = msg.content
                    break

            if not last_response:
                console.print("[dim]No response to format[/dim]")
                return None

            console.print()
            console.print(f"[dim]{'─' * 60}[/dim]")
            console.print(f"[bold {COLORS['secondary']}]Formatted Analysis[/bold {COLORS['secondary']}]")
            console.print(f"[dim]{'─' * 60}[/dim]")
            console.print()
            _display_formatted_response(last_response)
            return None

        case "/usage" | "/cost":
            _show_usage(orchestrator, config)
            return None

        case "/framework":
            if not args:
                if current_framework:
                    console.print(f"[dim]Current framework: {current_framework}[/dim]")
                else:
                    console.print("[dim]No framework selected (freestyle mode)[/dim]")
                return None

            if args.lower() == "off":
                console.print("[dim]Switched to freestyle mode[/dim]")
                return "framework:"

            # Validate framework
            valid_frameworks = ["piotroski", "porter", "pitch", "valuation"]
            if args.lower() in valid_frameworks or args.startswith("custom:"):
                fw_name = args.lower()
                console.print(f"\n[bold green]✓ Framework activated:[/bold green] {fw_name}\n")

                # Show framework-specific prompts
                if fw_name == "piotroski":
                    console.print(Panel(
                        "[bold]Piotroski F-Score[/bold] - Financial Health Analysis\n\n"
                        "This framework scores companies on 9 financial signals:\n"
                        "• Profitability (4 pts): ROA, Cash Flow, Earnings Quality\n"
                        "• Leverage (3 pts): Debt, Liquidity, Dilution\n"
                        "• Efficiency (2 pts): Margins, Asset Turnover\n\n"
                        "[cyan]To start, enter a ticker:[/cyan]\n"
                        "  [dim]research AAPL[/dim]  or just  [dim]AAPL[/dim]",
                        border_style="blue",
                    ))
                elif fw_name == "porter":
                    console.print(Panel(
                        "[bold]Porter's Five Forces[/bold] - Competitive Analysis\n\n"
                        "This framework analyzes competitive dynamics:\n"
                        "• Threat of New Entrants\n"
                        "• Supplier Power\n"
                        "• Buyer Power\n"
                        "• Threat of Substitutes\n"
                        "• Competitive Rivalry\n\n"
                        "[cyan]To start, enter a ticker:[/cyan]\n"
                        "  [dim]research MSFT[/dim]  or just  [dim]MSFT[/dim]",
                        border_style="blue",
                    ))
                elif fw_name == "pitch":
                    console.print(Panel(
                        "[bold]Hedge Fund Stock Pitch[/bold] - Thesis Format\n\n"
                        "This framework structures a professional thesis:\n"
                        "• Investment Thesis (1-2 sentences)\n"
                        "• Key Catalysts\n"
                        "• Valuation Analysis\n"
                        "• Risk Factors\n\n"
                        "[cyan]To start, enter a ticker:[/cyan]\n"
                        "  [dim]thesis NVDA[/dim]  or  [dim]research NVDA[/dim]",
                        border_style="blue",
                    ))
                elif fw_name == "valuation":
                    console.print(Panel(
                        "[bold]Valuation Analysis[/bold] - Price Target Generation\n\n"
                        "This framework calculates fair value using:\n"
                        "• P/E Multiple (vs sector)\n"
                        "• Forward P/E\n"
                        "• EV/EBITDA Multiple\n"
                        "• Analyst Consensus\n"
                        "• Growth-Adjusted (PEG)\n\n"
                        "Output: Bear / Base / Bull price targets\n\n"
                        "[cyan]To start, enter a ticker:[/cyan]\n"
                        "  [dim]research TSLA[/dim]  or just  [dim]TSLA[/dim]",
                        border_style="blue",
                    ))
                else:
                    console.print("[dim]Enter a ticker to begin analysis[/dim]")

                return f"framework:{fw_name}"
            else:
                console.print(f"[red]Unknown framework: {args}[/red]")
                console.print(f"[dim]Available: {', '.join(valid_frameworks)}[/dim]")
                return None

        case "/checklist":
            if not current_framework:
                console.print("[dim]No framework selected - use /framework to set one[/dim]")
            else:
                from bullsh.frameworks.base import load_framework
                try:
                    fw = load_framework(current_framework)
                    console.print(fw.to_checklist_display())
                except ValueError as e:
                    console.print(f"[red]Error: {e}[/red]")
            return None

        case "/excel":
            from bullsh.tools import excel as excel_tool

            # Parse args: /excel or /excel TSLA or /excel compare T1 T2
            parts = args.split() if args else []

            if not parts:
                # No args: use ALL session tickers
                if not session.tickers:
                    console.print("[red]No tickers in this session. Run research first or specify a ticker:[/red]")
                    console.print("[dim]  /excel TSLA[/dim]")
                    console.print("[dim]  /excel compare AAPL MSFT[/dim]")
                    return None

                tickers = session.tickers[:3]  # Max 3 for comparison
                if len(tickers) == 1:
                    console.print(f"[dim]Generating Excel for {tickers[0]} (from session)...[/dim]")
                    result = await excel_tool.generate_excel(tickers[0])
                else:
                    console.print(f"[dim]Generating comparison Excel for {', '.join(tickers)} (from session)...[/dim]")
                    result = await excel_tool.generate_excel(
                        tickers[0],
                        include_ratios=True,
                        compare_tickers=tickers[1:],
                    )
            elif parts[0].lower() == "compare" and len(parts) >= 3:
                # Explicit comparison: /excel compare AAPL MSFT GOOGL
                tickers = [t.upper() for t in parts[1:4]]
                # Warn if tickers not in session
                missing = [t for t in tickers if t not in session.tickers]
                if missing:
                    console.print(f"[yellow]⚠ Tickers not researched in this session: {', '.join(missing)}[/yellow]")
                    console.print("[dim]Data will be fetched fresh (not from session cache)[/dim]")
                console.print(f"[dim]Generating comparison Excel for {', '.join(tickers)}...[/dim]")
                result = await excel_tool.generate_excel(
                    tickers[0],
                    include_ratios=True,
                    compare_tickers=tickers[1:],
                )
            else:
                # Single ticker: /excel TSLA
                ticker = parts[0].upper()
                if ticker not in session.tickers:
                    console.print(f"[yellow]⚠ {ticker} was not researched in this session[/yellow]")
                    console.print("[dim]Data will be fetched fresh (not from session cache)[/dim]")
                console.print(f"[dim]Generating Excel for {ticker}...[/dim]")
                result = await excel_tool.generate_excel(ticker)

            if result.status.value == "success":
                console.print(f"[green]✓ Excel saved to:[/green] {result.data.get('path')}")
                console.print(f"[dim]Sheets: {', '.join(result.data.get('sheets', []))}[/dim]")
            else:
                console.print(f"[red]✗ Failed:[/red] {result.error_message}")
            return None

        case _:
            console.print(f"[red]Unknown command: {cmd}[/red]")
            console.print("[dim]Type /help for available commands[/dim]")
            return None


def _display_formatted_response(content: str) -> None:
    """Display agent response with beautiful formatting."""
    from bullsh.ui.formatter import format_agent_response

    # Check if response looks like structured analysis (has headers, lists)
    has_structure = (
        "##" in content or
        "**" in content or
        re.search(r'^\d+[.\)]\s', content, re.MULTILINE) or
        re.search(r'^[-*]\s', content, re.MULTILINE)
    )

    if has_structure:
        # Use our beautiful formatter
        formatted = format_agent_response(content)
        console.print(formatted)
    else:
        # Simple response - just clean it up a bit
        from bullsh.ui.formatter import _process_inline_formatting
        lines = content.strip().split("\n")
        for line in lines:
            if line.strip():
                formatted_line = _process_inline_formatting(line.strip())
                console.print("  ", end="")
                console.print(formatted_line)
            else:
                console.print()

    console.print()  # Final spacing


def _build_session_export(session: Session, framework: str | None) -> str:
    """
    Build a structured export document from the entire session.

    Aggregates all assistant messages and structures them with:
    - Header with metadata
    - All research findings
    - Clean formatting (removes tool status noise)
    """
    from datetime import datetime

    lines = []

    # Header
    ticker_str = ", ".join(session.tickers) if session.tickers else "General Research"
    lines.append(f"# {ticker_str} Investment Research\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if framework:
        framework_names = {
            "piotroski": "Piotroski F-Score",
            "porter": "Porter's Five Forces",
            "pitch": "Hedge Fund Stock Pitch",
            "valuation": "Valuation Analysis",
        }
        lines.append(f"**Framework:** {framework_names.get(framework, framework)}")
    lines.append(f"**Session:** {session.name}")
    lines.append("\n---\n")

    # Collect all assistant responses, cleaning up tool noise
    section_count = 0
    for msg in session.messages:
        if msg.role != "assistant":
            continue

        content = msg.content
        if not content or not content.strip():
            continue

        # Skip if it's just tool status output (starts with spinner characters)
        if content.strip().startswith(("◐", "◑", "◒", "◓", "✓", "✗", "[Data gathered")):
            continue

        # Clean up any remaining tool status lines (◐ ◑ ◒ ◓ ✓ ✗)
        cleaned_lines = []
        for line in content.split("\n"):
            # Skip tool status lines
            if line.strip().startswith(("◐", "◑", "◒", "◓", "⚠️ Maximum tool", "[Data gathered")):
                continue
            # Skip empty lines at the start
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)

        cleaned_content = "\n".join(cleaned_lines).strip()
        if not cleaned_content:
            continue

        section_count += 1

        # Add section separator for multiple responses
        if section_count > 1:
            lines.append("\n---\n")

        lines.append(cleaned_content)
        lines.append("")

    # Footer
    lines.append("\n---\n")
    lines.append("*This report was generated by bullsh - Investment Research Agent*")
    lines.append("*Made with ❤ by Alexander Duria*")

    return "\n".join(lines)


def _show_help() -> None:
    """Display help information."""
    help_text = """
[bold]Research Commands[/bold]

  [cyan]research <TICKER>[/cyan]           Research a single company
  [cyan]research <TICKER> -f piotroski[/cyan]   With Piotroski F-Score
  [cyan]research <TICKER> -f porter[/cyan]      With Porter's Five Forces
  [cyan]compare <T1> <T2> [T3][/cyan]      Compare up to 3 companies (parallel)
  [cyan]thesis <TICKER>[/cyan]             Generate investment thesis
  [cyan]summary <TICKER>[/cyan]            Quick company overview
  [cyan]frameworks list[/cyan]             Show available frameworks

[bold]Keybindings[/bold]

  [cyan]Tab[/cyan]                Show command suggestions
  [cyan]Ctrl+S[/cyan]             Save session
  [cyan]Ctrl+L[/cyan]             Clear screen
  [cyan]Ctrl+E[/cyan]             Export current research

[bold]Session Commands[/bold]

  [cyan]/save[/cyan]              Save current session
  [cyan]/sessions[/cyan]          List saved sessions
  [cyan]/resume <id>[/cyan]       Resume a previous session
  [cyan]/export [file][/cyan]     Export to markdown

[bold]Framework Commands[/bold]

  [cyan]/framework[/cyan]         Show current framework
  [cyan]/framework <name>[/cyan]  Switch framework (piotroski, porter)
  [cyan]/framework off[/cyan]     Return to freestyle mode
  [cyan]/checklist[/cyan]         Show framework progress

[bold]Export Commands[/bold]

  [cyan]/excel <TICKER>[/cyan]           Generate Excel financial model
  [cyan]/excel compare T1 T2[/cyan]      Excel with side-by-side comparison
  [cyan]/export [file][/cyan]            Export to markdown (default)
  [cyan]/export thesis.pdf[/cyan]        Export to PDF (auto-detect)
  [cyan]/export report.docx[/cyan]       Export to Word document
  [cyan]/export --format pdf[/cyan]      Specify format explicitly
  [cyan]/format[/cyan]                   Re-display last response beautifully

[bold]Other Commands[/bold]

  [cyan]/cache[/cyan]             Cache statistics
  [cyan]/cache clear[/cyan]       Clear cache
  [cyan]/rag[/cyan]               RAG vector database stats
  [cyan]/rag list[/cyan]          Show indexed filings
  [cyan]/rag clear[/cyan]         Clear index (re-index on next fetch)
  [cyan]/sources[/cyan]           Show data sources used
  [cyan]/usage[/cyan]             Token usage and cost
  [cyan]/config[/cyan]            Show configuration
  [cyan]/help[/cyan]              This help message
  [cyan]exit[/cyan]               Exit bullsh

[bold]Natural Language[/bold]

  You can also just ask questions naturally:
  • "What does the 10-K say about competition?"
  • "Show me the revenue breakdown"
  • "What are the main risks?"
  • "Compare AAPL vs MSFT" (triggers parallel comparison)

[bold]Available Frameworks[/bold]

  [yellow]piotroski[/yellow]  - 9-point quantitative financial health score
  [yellow]porter[/yellow]     - Competitive moat analysis (Five Forces)
  [yellow]pitch[/yellow]      - Hedge Fund Stock Pitch thesis format
  [yellow]valuation[/yellow]  - Multi-method price target generation
"""
    console.print(Panel(help_text, title="bullsh Help", border_style="blue"))


def _show_config(config: Config) -> None:
    """Display current configuration."""
    config_text = f"""
[bold]Configuration[/bold]

  Model:         {config.model}
  Verbosity:     {config.verbosity}
  Log level:     {config.log_level}
  Data dir:      {config.data_dir}
  Verbose tools: {config.verbose_tools}

[bold]Paths[/bold]

  Cache:         {config.cache_dir}
  Sessions:      {config.sessions_dir}
  Theses:        {config.theses_dir}
  Frameworks:    {config.frameworks_dir}
"""
    console.print(Panel(config_text, title="Current Configuration", border_style="dim"))


def _show_usage(orchestrator: Orchestrator, config: Config) -> None:
    """Display current token usage and cost estimates."""
    session_usage = orchestrator.session_usage
    session_cost = orchestrator.get_session_cost()

    # Calculate percentages
    session_pct = (session_usage.total_tokens / config.max_tokens_per_session) * 100

    # Determine color based on usage
    if session_pct >= 80:
        color = "red"
    elif session_pct >= 50:
        color = "yellow"
    else:
        color = "green"

    # Cache info
    cache_info = ""
    if session_usage.cache_read_tokens > 0 or session_usage.cache_creation_tokens > 0:
        cache_info = f"""
[bold]Prompt Caching[/bold]

  Cache reads:    {session_usage.cache_read_tokens:,} tokens (90% discount)
  Cache writes:   {session_usage.cache_creation_tokens:,} tokens
  Cache hit rate: [green]{session_usage.cache_hit_rate:.1f}%[/green]
"""

    # History info
    history_len = len(orchestrator.history)
    history_status = "OK"
    if history_len >= config.max_history_messages:
        history_status = f"[yellow]sliding window active[/yellow]"

    usage_text = f"""
[bold]Session Token Usage[/bold]

  Input tokens:   {session_usage.input_tokens:,}
  Output tokens:  {session_usage.output_tokens:,}
  Total tokens:   [{color}]{session_usage.total_tokens:,}[/{color}]
  API calls:      {session_usage.api_calls}
  History:        {history_len} messages ({history_status})
{cache_info}
[bold]Estimated Cost[/bold]

  This session:   ${session_cost:.4f}

[bold]Limits[/bold]

  Session limit:  {config.max_tokens_per_session:,} tokens
  Turn limit:     {config.max_tokens_per_turn:,} tokens
  History limit:  {config.max_history_messages} messages
  Usage:          [{color}]{session_pct:.1f}%[/{color}] of session limit
"""
    console.print(Panel(usage_text, title="Token Usage", border_style=color))
