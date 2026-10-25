# AGENTS.md

Instructions for any coding agent working in this repository.

## What this project is

`frau-agent` is a local German conversation partner. It runs entirely on the
machine: a chat model behind an OpenAI-compatible endpoint (Ollama by default)
via pydantic-ai, plus Whistle (Cactus Compute) for microphone input. No cloud
calls after the one-time model download, no API keys.

The learner types or speaks German; the tutor replies in German, ends every
reply with a question, and appends at most one `TIP:` correction line in
English.

## Role of the agent

You are a pair programmer on this codebase, and a German tutor is what the app
is. Keep those two voices separate:

- **You, as the coding agent**, talk to the developer in English.
- **The app, as a tutor**, talks to the learner in German.

That split is the core convention of this repo — see the language rule below.

## Language rule

Audience-facing text is **English first**, unless the whole point of the string
is German practice.

English (translate when you touch it, and do not add German equivalents):

- Terminal chrome: prompts, spinners, status lines, errors, help, banners.
- `stt.py` feedback and errors (`Wörter/Min`, `längste Pause`, `unklar` → words
  per minute, longest pause, unclear words; `Mikrofon nicht verfügbar` →
  `microphone unavailable`).
- `README.md`, `.env.example`, docstrings, comments, commit messages, this file.
- The `TIP:` line the tutor emits.

German (leave German, and keep it at the learner's level):

- The tutor's own replies: the `SYSTEM_PROMPT` body and the conversational
  content it produces.
- The goodbye line `Tschüss! Bis bald.` in `cli.py` — the one intentional German
  touch in the otherwise English chrome. Leave it alone.
- Example transcripts in the README, because they show what a learner said.
- Whistle's `language="de"` argument.

Never leave half-translated UI: if you change one string in a block, change the
neighbouring ones too, and update the README screenshot/example blocks that
quote them.

## Commands

```sh
uv sync                     # install / update dependencies
uv run main.py              # run the app (or: uv run frau)
uv run frau --voice         # start in voice mode
uv run frau --text          # force text mode
uv run ruff check .         # lint
uv run ruff format .        # format
```

Python ≥ 3.14 (`.python-version`). Dependencies are locked in `uv.lock`; use
`uv add` rather than editing `pyproject.toml` by hand.

## Layout

```
main.py                entry point, delegates to cli.main
frau_agent/agent.py    model wiring, level handling, system prompt, speech note
frau_agent/cli.py      terminal loop, streaming render, /commands, mic mode
frau_agent/stt.py      recording (VAD), Whistle, transcript timings + feedback
```

## Design invariants

- **No blocking calls in the chat loop.** Streaming uses `run_stream` and a
  Rich `Live`; anything slow stays inside `console.status`.
- **`stt.py` owns thresholds, `cli.py` owns rendering.** `Transcript.summary()`
  returns `(text, rich-style)` pairs as data; do not print from `stt.py`.
- **Errors are reported, never fatal.** Model and microphone failures print a
  hint and return, so a dropped Ollama or a denied mic does not end the session.
- **The `TIP:` line is part of the reply format.** Exactly one, English, at most
  one real correction per turn — `render_reply` splits on the last `TIP:` and
  hides it when it is `none` or a no-op (`_is_noop_tip`).
- **Reasoning is off** (`reasoning_effort: "none"`) because Qwen3 would otherwise
  deliberate over two-sentence answers.
- **Voice mode also changes the prompt**, not just the input path: `SPEECH_NOTE`
  stops the tutor "correcting" words Whistle misheard. Rebuild the agent when
  the voice flag flips (`/mic`).
- **The speech model is cached by the engine** under `~/.cache/cactus-needle/`
  (or `GERMAN_BOT_STT_WEIGHTS`); `model_cached()` decides whether to warn about
  the one-time download. Telemetry stays off: `stt.py` sets `NEEDLE_TELEMETRY=0`
  and `DO_NOT_TRACK=1` before importing `needle`.
- **MacOS specifics are documented, not automated**: mic permissions live in
  System Settings → Privacy & Security → Microphone.

## Style

- Type hints on public functions, `from __future__ import annotations`, `str | None`.
- Keep the compact, comment-the-*why* style already in `stt.py`: explain
  trade-offs and thresholds, not mechanics.
- Rich markup in strings is fine; prefer `Text` for anything assembled.
- Small modules, no new dependencies unless the existing ones cannot do the job —
  and if they cannot, say why in the module docstring or README table.
