"""Interactive terminal chat loop."""

from __future__ import annotations

import asyncio
import re

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .agent import build_agent

console = Console()

BANNER = """\
[bold]Frau Agent[/bold] — German practice, local and offline.
Chat in German. Type [cyan]/help[/cyan] for commands, [cyan]/quit[/cyan] to leave."""

HELP = """\
[bold]Commands[/bold]
  /level A1|A2|B1   change difficulty (resets the conversation)
  /reset            start a fresh conversation
  /history          show how many messages are in context
  /help             this text
  /quit             exit (\u2318C also works)

[bold]Tips[/bold]
  Answer in full sentences — that is where the practice is.
  When you make a mistake worth fixing, a dimmed TIP line appears below the reply."""


def render_reply(text: str) -> Text:
    """German part in normal weight, the TIP line dimmed unless there is nothing to show."""
    out = Text()
    german, _, tip = text.partition("TIP:")
    out.append(german.rstrip())

    tip = tip.strip()
    if tip and not tip.lower().startswith("none") and not _is_noop_tip(tip):
        out.append("\n\n")
        out.append("TIP:", style="bold yellow")
        out.append(" " + tip, style="dim italic")
    return out


def _is_noop_tip(tip: str) -> bool:
    """Catch 'a' -> 'a' style non-corrections the model sometimes emits."""
    match = re.match(r"""^['"“]?(.+?)['"”]?\s*(?:->|→|=>)\s*['"“]?(.+?)['"”]?\s*(?:—|--|-\s)""", tip)
    if not match:
        return False
    left, right = match.group(1).strip(), match.group(2).strip()
    return left.casefold() == right.casefold()


async def chat_loop() -> None:
    level = "A1"
    agent = build_agent(level)
    history: list = []

    console.print(Panel(BANNER, border_style="cyan"))
    console.print()

    while True:
        try:
            user_input = console.input("[bold green]du >[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Tschüss! Bis bald.[/dim]")
            return

        if not user_input:
            continue

        # --- commands -----------------------------------------------------
        if user_input.startswith("/"):
            cmd, _, arg = user_input[1:].partition(" ")
            cmd = cmd.lower()

            if cmd in {"quit", "exit", "q"}:
                console.print("[dim]Tschüss! Bis bald.[/dim]")
                return
            if cmd == "help":
                console.print(HELP)
                continue
            if cmd == "reset":
                history.clear()
                console.print("[dim]Fresh start.[/dim]\n")
                continue
            if cmd == "history":
                console.print(f"[dim]{len(history)} messages in context.[/dim]\n")
                continue
            if cmd == "level":
                new_level = arg.strip().upper()
                if new_level not in {"A1", "A2", "B1"}:
                    console.print("[red]Use /level A1, A2 or B1.[/red]")
                    continue
                level = new_level
                agent = build_agent(level)
                history.clear()
                console.print(f"[dim]Level {level}, fresh conversation.[/dim]\n")
                continue
            console.print(f"[red]Unknown command: /{cmd}[/red]")
            continue

        # --- the actual conversation --------------------------------------
        buffer = ""
        live_text = Text("…", style="dim")
        try:
            with Live(live_text, console=console, refresh_per_second=12, transient=False) as live:
                async with agent.run_stream(user_input, message_history=history) as result:
                    async for chunk in result.stream_text(delta=True):
                        buffer += chunk
                        live.update(render_reply(buffer))
                    history = list(result.all_messages())
                live.update(render_reply(buffer))
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]\n")
            continue
        except Exception as exc:  # noqa: BLE001 - a dropped model must not kill the loop
            console.print(f"[red]Model error:[/red] {exc}")
            console.print("[dim]Is Ollama running? Try: ollama serve[/dim]\n")
            continue

        console.print()


def main() -> None:
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        pass
