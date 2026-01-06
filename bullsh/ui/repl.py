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
        ("/framework factors", "Interactive multi-factor stock analysis"),
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
        ("factors", "Interactive multi-factor stock analysis"),
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

    # Wire session to orchestrator for artifact tracking
    orchestrator.session = session

    # Play animated intro (unless skipped)
    # If animation completes, it already shows the welcome screen
    intro_showed_welcome = False
    if not skip_intro:
        intro_showed_welcome = await _play_intro()

    # Only show welcome banner if intro didn't already show it
    if not intro_showed_welcome:
        _show_welcome(framework)

    await _repl_loop(config, orchestrator, session, framework)


async def _play_intro(skip: bool = False) -> bool:
    """
    Play the animated intro sequence.

    Returns:
        True if animation completed and showed welcome screen
    """
    if skip:
        return False

    try:
        from bullsh.ui.intro import play_intro_animation
        return await play_intro_animation(duration=4.0)
    except Exception:
        # Fallback if animation fails
        return False


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
                fw_value = result.split(":", 1)[1] or None
                # Check for factors:start signal to enter interactive session
                if fw_value == "factors:start":
                    current_framework = "factors"
                    session.framework = current_framework
                    await _run_factor_session(orchestrator, session, config)
                else:
                    current_framework = fw_value
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


async def _run_factor_session(
    orchestrator: Orchestrator,
    session: Session,
    config: Config,
) -> None:
    """
    Run interactive 8-stage factor analysis session.

    Uses pure Python for calculations (zero tokens) and Claude
    only for professor explanations of pre-computed results.
    """
    from datetime import datetime
    from bullsh.factors.session import FactorSession, FactorStage, validate_us_ticker, validate_peer_set
    from bullsh.factors.fetcher import fetch_all_factor_data
    from bullsh.factors.calculator import calculate_factor_scores, calculate_composite_score
    from bullsh.factors.regression import (
        run_factor_regression, run_rolling_regression, calculate_variance_decomposition,
        calculate_correlations, prepare_fama_french_data, prepare_stock_returns
    )
    from bullsh.factors.scenarios import calculate_all_scenario_returns, SCENARIOS
    from bullsh.factors.prompts import (
        build_stage_prompt, format_factor_menu, parse_factor_selection, parse_weight_input
    )
    from bullsh.factors.excel_factors import generate_factor_excel

    # Initialize or resume factor session
    factor_session = FactorSession(session)
    session_manager = get_session_manager()

    # Data cache for the session (not persisted, just for this run)
    cached_data: dict = {}

    console.print("\n[bold cyan]Multi-Factor Analysis Session[/bold cyan]")
    console.print(f"[dim]Stage {factor_session.stage.value}/8: {factor_session.stage.display_name}[/dim]\n")

    while not factor_session.is_complete:
        stage = factor_session.stage

        # =====================================================
        # STAGE 1: Ticker Selection
        # =====================================================
        if stage == FactorStage.TICKER_SELECTION:
            if factor_session.state.primary_ticker:
                console.print(f"[dim]Primary ticker: {factor_session.state.primary_ticker}[/dim]")
                factor_session.advance_stage()
                continue

            # Get professor introduction via Claude
            prompt = build_stage_prompt(stage, factor_session.state)
            console.print("[bold]Professor:[/bold]")
            async for chunk in orchestrator.chat(
                "Start the factor analysis session. Welcome the student and ask for a ticker.",
                system_override=prompt
            ):
                console.print(chunk, end="")
            console.print("\n")

            # Get ticker input
            try:
                ticker_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("Enter ticker: ").strip().upper()
                )
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Factor session cancelled.[/dim]")
                return

            if not ticker_input:
                continue

            # Handle exit commands
            if ticker_input.lower() in ("exit", "quit", "q", "/exit"):
                console.print("[dim]Factor session ended.[/dim]")
                return

            # Validate ticker
            is_valid, msg = validate_us_ticker(ticker_input)
            if not is_valid:
                console.print(f"[red]{msg}[/red]")
                continue

            # Set ticker and advance
            factor_session.set_ticker(ticker_input)
            session.tickers.append(ticker_input)
            console.print(f"[green]✓ {ticker_input} selected[/green]\n")
            factor_session.advance_stage()

        # =====================================================
        # STAGE 2: Peer Selection
        # =====================================================
        elif stage == FactorStage.PEER_SELECTION:
            if len(factor_session.state.peers) >= 2:
                console.print(f"[dim]Peers: {', '.join(factor_session.state.peers)}[/dim]")
                factor_session.advance_stage()
                continue

            console.print(f"[dim]Stage 2/8: Peer Selection[/dim]")
            console.print(f"[bold]Analyzing:[/bold] {factor_session.state.primary_ticker}\n")

            # Professor guidance
            prompt = build_stage_prompt(stage, factor_session.state)
            console.print("[bold]Professor:[/bold]")
            async for chunk in orchestrator.chat(
                f"Guide peer selection for {factor_session.state.primary_ticker}. Ask for 2-6 peer tickers.",
                system_override=prompt
            ):
                console.print(chunk, end="")
            console.print("\n")

            # Get peer input
            try:
                peer_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("Enter peers (comma-separated): ").strip()
                )
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Factor session cancelled.[/dim]")
                return

            if not peer_input:
                continue

            # Handle commands
            if peer_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Factor session ended.[/dim]")
                return
            if peer_input.lower() == "back":
                factor_session.go_back()
                continue

            # Parse peers
            peers = [p.strip().upper() for p in peer_input.replace(",", " ").split() if p.strip()]

            # Validate peer set
            is_valid, issues = validate_peer_set(factor_session.state.primary_ticker, peers)
            if not is_valid:
                for issue in issues:
                    console.print(f"[red]• {issue}[/red]")
                continue

            # Set peers and advance
            factor_session.set_peers(peers)
            console.print(f"[green]✓ Peers set: {', '.join(peers)}[/green]\n")
            factor_session.advance_stage()

        # =====================================================
        # STAGE 3: Factor Selection & Weighting
        # =====================================================
        elif stage == FactorStage.FACTOR_WEIGHTING:
            if factor_session.state.selected_factors:
                console.print(f"[dim]Factors: {', '.join(factor_session.state.selected_factors)}[/dim]")
                factor_session.advance_stage()
                continue

            console.print(f"[dim]Stage 3/8: Factor Selection & Weighting[/dim]\n")

            # Show factor menu
            console.print(format_factor_menu())

            # Get factor selection
            try:
                factor_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\nYour selection: ").strip()
                )
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Factor session cancelled.[/dim]")
                return

            if not factor_input:
                continue

            if factor_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Factor session ended.[/dim]")
                return
            if factor_input.lower() == "back":
                factor_session.go_back()
                continue

            # Parse selection
            factors = parse_factor_selection(factor_input)
            if not factors:
                console.print("[red]No valid factors selected. Try again.[/red]")
                continue

            console.print(f"[green]✓ Selected factors: {', '.join(factors)}[/green]")

            # Ask about weights
            console.print("\n[1] Equal weights (recommended)")
            console.print("[2] Custom weights\n")

            try:
                weight_choice = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("Weight choice (1/2): ").strip()
                )
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Factor session cancelled.[/dim]")
                return

            if weight_choice == "2":
                console.print("\nEnter weights as percentages (must sum to 100):")
                console.print(f"[dim]Factors: {', '.join(factors)}[/dim]")
                try:
                    weight_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("Weights: ").strip()
                    )
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[dim]Factor session cancelled.[/dim]")
                    return

                weights = parse_weight_input(weight_input, factors)
                if not weights:
                    console.print("[yellow]Invalid weights, using equal weights.[/yellow]")
                    weights = None
            else:
                weights = None

            factor_session.set_factors(factors, weights)
            console.print(f"[green]✓ Weights set[/green]\n")
            factor_session.advance_stage()

        # =====================================================
        # STAGE 4: Data Fetching (Pure Python - No Claude)
        # =====================================================
        elif stage == FactorStage.DATA_FETCHING:
            console.print(f"[dim]Stage 4/8: Data Fetching[/dim]\n")
            console.print("[bold]Fetching data in parallel...[/bold]")

            all_tickers = factor_session.get_all_tickers()
            console.print(f"[dim]Tickers: {', '.join(all_tickers)}[/dim]")
            console.print(f"[dim]Benchmark: ^GSPC (S&P 500)[/dim]")
            console.print(f"[dim]Fama-French factor returns[/dim]\n")

            # Fetch all data
            try:
                from rich.progress import Progress, SpinnerColumn, TextColumn
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Fetching data...", total=None)
                    cached_data = await fetch_all_factor_data(
                        factor_session.state.primary_ticker,
                        factor_session.state.peers,
                    )
                    progress.update(task, description="[green]Data fetched![/green]")

                # Mark data as fetched
                factor_session.state.data_fetched_at = datetime.now().isoformat()
                factor_session.save()

                console.print(f"[green]✓ Data fetched for {len(all_tickers)} tickers[/green]\n")
                factor_session.advance_stage()

            except Exception as e:
                console.print(f"[red]Error fetching data: {e}[/red]")
                console.print("[dim]Retry or type 'exit' to cancel[/dim]")
                try:
                    retry = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("Press Enter to retry: ").strip()
                    )
                    if retry.lower() in ("exit", "quit", "q"):
                        return
                except (KeyboardInterrupt, EOFError):
                    return

        # =====================================================
        # STAGE 5: Factor Calculation (Python calc, Claude explains)
        # =====================================================
        elif stage == FactorStage.FACTOR_CALCULATION:
            console.print(f"[dim]Stage 5/8: Factor Calculation[/dim]\n")

            # Calculate factor scores (Pure Python)
            console.print("[bold]Computing factor scores...[/bold]")

            try:
                profiles = calculate_factor_scores(
                    factor_session.state.primary_ticker,
                    factor_session.state.peers,
                    cached_data.get("yahoo_data", {}),
                    factor_session.state.selected_factors,
                    cached_data.get("price_history"),
                )

                # Extract z-scores for storage
                scores = {}
                for ticker, profile in profiles.items():
                    scores[ticker] = {f: s.z_score for f, s in profile.scores.items()}

                    # Calculate composite
                    profile.composite_score = calculate_composite_score(
                        profile.scores,
                        factor_session.state.weights,
                    )

                factor_session.set_factor_scores(scores)
                # Store full profiles for Excel generation
                cached_data["profiles"] = profiles
                console.print("[green]✓ Scores calculated[/green]\n")

                # Display scores table
                table = Table(title=f"Factor Z-Scores for {factor_session.state.primary_ticker}")
                table.add_column("Factor", style="cyan")
                table.add_column("Z-Score", justify="right")
                table.add_column("Interpretation")

                primary_scores = scores.get(factor_session.state.primary_ticker, {})
                for factor in factor_session.state.selected_factors:
                    z = primary_scores.get(factor, 0)
                    if z > 0.5:
                        interp = "[green]Above Average[/green]"
                    elif z < -0.5:
                        interp = "[red]Below Average[/red]"
                    else:
                        interp = "[yellow]Average[/yellow]"
                    table.add_row(factor.title(), f"{z:.2f}", interp)

                console.print(table)
                console.print()

                # Professor explains (Claude call)
                prompt = build_stage_prompt(stage, factor_session.state, {"factor_scores": scores})
                console.print("[bold]Professor:[/bold]")
                async for chunk in orchestrator.chat(
                    f"Explain the factor scores I calculated for {factor_session.state.primary_ticker}. Walk through one calculation example.",
                    system_override=prompt
                ):
                    console.print(chunk, end="")
                console.print("\n")

                factor_session.advance_stage()

            except Exception as e:
                console.print(f"[red]Calculation error: {e}[/red]")
                factor_session.advance_stage()  # Continue anyway

        # =====================================================
        # STAGE 6: Risk Decomposition
        # =====================================================
        elif stage == FactorStage.RISK_DECOMPOSITION:
            console.print(f"[dim]Stage 6/8: Risk Decomposition[/dim]\n")

            try:
                # Calculate correlations
                correlations = calculate_correlations(
                    cached_data.get("price_history", {}),
                    cached_data.get("benchmark_history"),
                )
                factor_session.set_correlation_matrix(correlations)

                # Run Fama-French regression for real variance decomposition
                console.print("[bold]Running factor regression...[/bold]")

                ff_data = cached_data.get("fama_french", {})
                price_history = cached_data.get("price_history", {})
                primary_ticker = factor_session.state.primary_ticker

                variance_decomp = {}
                regression_betas = {}

                if ff_data and ff_data.get("factors") and price_history.get(primary_ticker):
                    try:
                        # Prepare data
                        stock_returns = prepare_stock_returns(price_history[primary_ticker])
                        factor_returns, rf_returns = prepare_fama_french_data(
                            ff_data, price_history[primary_ticker]
                        )

                        if not stock_returns.empty and not factor_returns.empty:
                            # Run regression
                            reg_result = run_factor_regression(stock_returns, factor_returns, rf_returns)

                            if reg_result is not None:
                                # Store betas
                                regression_betas[primary_ticker] = dict(reg_result.betas)
                                factor_session.set_regression_betas(regression_betas)

                                # Calculate factor variances for decomposition
                                factor_variances = {
                                    col: float(factor_returns[col].var())
                                    for col in factor_returns.columns
                                }

                                variance_decomp = calculate_variance_decomposition(
                                    reg_result, factor_variances
                                )
                                console.print("[green]✓ Fama-French regression complete[/green]")

                                # Run rolling regression for historical exposures
                                console.print("[bold]Calculating rolling factor exposures...[/bold]")
                                rolling_betas = run_rolling_regression(
                                    stock_returns, factor_returns, rf_returns, window=36
                                )
                                if rolling_betas:
                                    cached_data["rolling_betas"] = rolling_betas
                                    console.print("[green]✓ Rolling regression complete[/green]")
                            else:
                                console.print("[yellow]⚠ Regression failed - using approximation[/yellow]")
                    except Exception as e:
                        console.print(f"[yellow]⚠ Regression error: {e} - using approximation[/yellow]")

                # Fallback to approximation if regression failed
                if not variance_decomp:
                    primary_scores = factor_session.state.factor_scores.get(primary_ticker, {})
                    total_exposure = sum(abs(z) for z in primary_scores.values()) or 1
                    for factor, z in primary_scores.items():
                        variance_decomp[factor] = (abs(z) / total_exposure) * 70
                    variance_decomp["idiosyncratic"] = 30

                factor_session.set_variance_decomposition(variance_decomp)
                console.print("[green]✓ Risk decomposition complete[/green]\n")

                # Display pie breakdown
                console.print("[bold]Variance Attribution:[/bold]")
                for factor, pct in variance_decomp.items():
                    bar = "█" * int(pct / 5)
                    console.print(f"  {factor.title():15} {pct:5.1f}% {bar}")
                console.print()

                # Professor explains
                prompt = build_stage_prompt(stage, factor_session.state, {
                    "variance_decomposition": variance_decomp,
                    "correlations": correlations,
                })
                console.print("[bold]Professor:[/bold]")
                async for chunk in orchestrator.chat(
                    "Explain the risk decomposition results. What does this tell us about systematic vs idiosyncratic risk?",
                    system_override=prompt
                ):
                    console.print(chunk, end="")
                console.print("\n")

                factor_session.advance_stage()

            except Exception as e:
                console.print(f"[yellow]Risk decomposition warning: {e}[/yellow]")
                factor_session.advance_stage()

        # =====================================================
        # STAGE 7: Scenario Analysis
        # =====================================================
        elif stage == FactorStage.SCENARIO_ANALYSIS:
            console.print(f"[dim]Stage 7/8: Scenario Analysis[/dim]\n")

            try:
                # Get factor exposures for primary ticker
                exposures = factor_session.state.factor_scores.get(
                    factor_session.state.primary_ticker, {}
                )

                # Calculate scenario returns
                scenario_returns = calculate_all_scenario_returns(exposures)
                factor_session.set_scenario_results(scenario_returns)

                console.print("[green]✓ Scenario analysis complete[/green]\n")

                # Display results
                console.print(f"[bold]Scenario Expected Returns for {factor_session.state.primary_ticker}:[/bold]")
                for scenario_name, ret in scenario_returns.items():
                    scenario = SCENARIOS.get(scenario_name)
                    display_name = scenario.display_name if scenario else scenario_name
                    color = "green" if ret > 0 else "red"
                    console.print(f"  {display_name:30} [{color}]{ret*100:+.1f}%[/{color}]")
                console.print()

                # Professor explains
                prompt = build_stage_prompt(stage, factor_session.state, {
                    "scenario_results": scenario_returns,
                })
                console.print("[bold]Professor:[/bold]")
                async for chunk in orchestrator.chat(
                    "Explain the scenario analysis results. Which scenarios pose risks and which present opportunities?",
                    system_override=prompt
                ):
                    console.print(chunk, end="")
                console.print("\n")

                factor_session.advance_stage()

            except Exception as e:
                console.print(f"[yellow]Scenario warning: {e}[/yellow]")
                factor_session.advance_stage()

        # =====================================================
        # STAGE 8: Excel Generation (Pure Python - No Claude)
        # =====================================================
        elif stage == FactorStage.EXCEL_GENERATION:
            console.print(f"[dim]Stage 8/8: Excel Generation[/dim]\n")

            console.print("[bold]Generating 9-tab Excel workbook...[/bold]")

            try:
                excel_path = generate_factor_excel(
                    factor_session.state,
                    cached_data,
                    draft=False,
                )
                factor_session.set_excel_path(str(excel_path), final=True)

                console.print(f"[green]✓ Excel workbook saved:[/green]")
                console.print(f"  [cyan]{excel_path}[/cyan]\n")

                # Final summary
                console.print("[bold]Professor:[/bold]")
                console.print(f"Congratulations on completing the factor analysis for {factor_session.state.primary_ticker}!\n")
                console.print("[bold]Key Findings:[/bold]")

                primary_scores = factor_session.state.factor_scores.get(
                    factor_session.state.primary_ticker, {}
                )
                strongest = max(primary_scores.items(), key=lambda x: x[1]) if primary_scores else ("N/A", 0)
                weakest = min(primary_scores.items(), key=lambda x: x[1]) if primary_scores else ("N/A", 0)

                console.print(f"  • Strongest factor: {strongest[0].title()} (z={strongest[1]:.2f})")
                console.print(f"  • Weakest factor: {weakest[0].title()} (z={weakest[1]:.2f})")

                best_scenario = max(factor_session.state.scenario_results.items(), key=lambda x: x[1])
                worst_scenario = min(factor_session.state.scenario_results.items(), key=lambda x: x[1])
                console.print(f"  • Best scenario: {best_scenario[0]} ({best_scenario[1]*100:+.1f}%)")
                console.print(f"  • Worst scenario: {worst_scenario[0]} ({worst_scenario[1]*100:+.1f}%)")
                console.print()

                console.print("[bold]Excel Workbook Tabs:[/bold]")
                tabs = [
                    "Executive Summary", "Factor Exposures", "Peer Comparison",
                    "Risk Decomposition", "Historical Exposures", "Scenario Analysis",
                    "Fundamentals", "Price Data", "Methodology"
                ]
                for i, tab in enumerate(tabs, 1):
                    console.print(f"  {i}. {tab}")
                console.print()

                factor_session.mark_complete()

            except Exception as e:
                console.print(f"[red]Excel generation error: {e}[/red]")
                factor_session.mark_complete()

    # Session complete
    console.print("[bold green]✓ Factor analysis session complete![/bold green]\n")
    session_manager.save(session)


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
            valid_frameworks = ["piotroski", "porter", "pitch", "valuation", "factors"]
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
                elif fw_name == "factors":
                    console.print(Panel(
                        "[bold]Multi-Factor Stock Analysis[/bold] - Educational Session\n\n"
                        "This interactive session guides you through:\n"
                        "• Ticker & Peer Selection (2-6 comparable companies)\n"
                        "• Factor Selection (Value, Momentum, Quality, Growth, Size, Volatility)\n"
                        "• Cross-sectional Z-score calculations\n"
                        "• Risk decomposition & variance attribution\n"
                        "• Scenario analysis (Rate Shock, Risk-Off, Recession, Cyclical)\n"
                        "• Excel workbook generation (9 tabs)\n\n"
                        "A finance professor will explain each step.\n\n"
                        "[cyan]To start, enter a US stock ticker.[/cyan]",
                        border_style="blue",
                    ))
                    # Return special signal to start factor session
                    return "framework:factors:start"
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
  [cyan]/framework <name>[/cyan]  Switch framework (piotroski, porter, factors)
  [cyan]/framework factors[/cyan] Interactive multi-factor analysis session
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
