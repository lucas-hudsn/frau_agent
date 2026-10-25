"""Interactive terminal chat loop."""

from __future__ import annotations

import argparse
import asyncio
import os
import re

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from . import stt
from .agent import build_agent

console = Console()

# Speak instead of typing by default (`--voice`, `/mic`, or the env var).
VOICE = os.getenv("GERMAN_BOT_VOICE", "").strip().lower() in {"1", "true", "yes", "on"}

BANNER = """\
[bold]Frau Agent[/bold] — German practice, local and offline.
Chat in German. Type [cyan]/help[/cyan] for commands, [cyan]/quit[/cyan] to leave."""

HELP = """\
[bold]Commands[/bold]
  /voice            speak one answer (microphone, local Whistle)
  /mic              toggle speaking for every answer
  /level A1|A2|B1   change difficulty (resets the conversation)
  /reset            start a fresh conversation
  /history          show how many messages are in context
  /help             this text
  /quit             exit (\u2318C also works)

[bold]Speaking[/bold]
  /voice records until you pause, then transcribes on this machine.
  Under the transcript you get pace, hesitations, and words the recogniser
  was unsure about — a hint about where to repeat yourself, not a grade.

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


# --------------------------------------------------------------------------- #
# speech input
# --------------------------------------------------------------------------- #
def render_speech(transcript: stt.Transcript) -> None:
    """Echo what was heard, then the speaking feedback Whistle's timings give us."""
    console.print()
    console.print(f'[bold cyan]🎤[/bold cyan] [italic]“{transcript.text}”[/italic]')

    line = Text()
    for index, (text, style) in enumerate(transcript.summary()):
        if index:
            line.append("   ·   ", style="dim")
        line.append(text, style=style)
    console.print("   ", line)


def listen() -> str | None:
    """Record one utterance and transcribe it. None when nothing usable came back."""
    if not stt.model_cached():
        console.print(
            "[dim]Downloading the speech model “Whistle” once "
            "(~17 MB, offline afterwards).[/dim]"
        )

    try:
        with console.status("[dim]Speak now … (stops on silence)[/dim]", spinner="dots"):
            audio = stt.record()
        with console.status("[dim]Transcribing …[/dim]", spinner="dots"):
            transcript = stt.transcribe(audio)
    except stt.NoSpeechError:
        console.print("[yellow]Heard nothing — try again?[/yellow]\n")
        return None
    except stt.STTError as exc:
        console.print(f"[red]Speech:[/red] {exc}")
        console.print(
            "[dim]If this is a microphone problem on macOS: System Settings → "
            "Privacy & Security → Microphone → enable your terminal.[/dim]\n"
        )
        return None

    render_speech(transcript)
    return transcript.text


# --------------------------------------------------------------------------- #
# the conversation
# --------------------------------------------------------------------------- #
async def respond(agent, history: list, user_input: str) -> list:
    """Stream one reply, live-rendered, and return the updated message history."""
    buffer = ""
    try:
        with Live(Text("…", style="dim"), console=console, refresh_per_second=12) as live:
            async with agent.run_stream(user_input, message_history=history) as result:
                async for chunk in result.stream_text(delta=True):
                    buffer += chunk
                    live.update(render_reply(buffer))
                history = list(result.all_messages())
            live.update(render_reply(buffer))
    except KeyboardInterrupt:
        console.print("[dim]Cancelled.[/dim]\n")
        return history
    except Exception as exc:  # noqa: BLE001 - a dropped model must not kill the loop
        console.print(f"[red]Model error:[/red] {exc}")
        console.print("[dim]Is Ollama running? Try: ollama serve[/dim]\n")
        return history

    console.print()
    return history


async def chat_loop(voice: bool = VOICE) -> None:
    level = "A1"
    agent = build_agent(level, voice=voice)
    history: list = []

    console.print(Panel(BANNER, border_style="cyan"))
    if voice:
        console.print("[dim]Voice mode on — press Enter to speak, or just type.[/dim]")
    console.print()

    while True:
        prompt = "[bold green]du 🎤 >[/bold green] " if voice else "[bold green]du >[/bold green] "
        try:
            user_input = console.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Tschüss! Bis bald.[/dim]")
            return

        # In voice mode an empty line means "my turn to speak".
        if voice and not user_input:
            user_input = listen() or ""
            if not user_input:
                continue

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
            if cmd == "voice":
                spoken = listen()
                if spoken:
                    history = await respond(agent, history, spoken)
                continue
            if cmd == "mic":
                voice = not voice
                # Same conversation, new input mode: only the prompt changes, so
                # the model is rebuilt with the speech caveat added or removed.
                agent = build_agent(level, voice=voice)
                console.print(
                    "[dim]Voice mode on — Enter to speak.[/dim]\n"
                    if voice
                    else "[dim]Voice mode off.[/dim]\n"
                )
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
                agent = build_agent(level, voice=voice)
                history.clear()
                console.print(f"[dim]Level {level}, fresh conversation.[/dim]\n")
                continue
            console.print(f"[red]Unknown command: /{cmd}[/red]")
            continue

        # --- the actual conversation --------------------------------------
        history = await respond(agent, history, user_input)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="frau", description="Local German practice partner")
    parser.add_argument(
        "--voice",
        action="store_true",
        default=VOICE,
        help="start in voice mode (Enter records, speech is transcribed locally)",
    )
    parser.add_argument("--text", dest="voice", action="store_false", help="force text input")
    args = parser.parse_args(argv)

    try:
        asyncio.run(chat_loop(voice=args.voice))
    except KeyboardInterrupt:
        pass
