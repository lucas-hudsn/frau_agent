# Frau Agent

A local German conversation partner. Text only, no cloud calls — the model runs
on your machine via [pydantic-ai](https://ai.pydantic.dev) + Ollama.

## Run

```sh
uv run main.py
# or, after `uv sync`, simply:
uv run frau
```

Requires the model to be pulled once: `ollama pull qwen3.8:27b` (already present
on this machine).

## Commands

| Command | Effect |
| --- | --- |
| `/level A1\|A2\|B1` | Change difficulty, resets the conversation |
| `/reset` | Fresh conversation, same level |
| `/history` | How many messages are in context |
| `/help` | Command list |
| `/quit` | Exit |

## How it behaves

- Replies in German only, 1–3 short sentences, always ending in a question so
  the conversation keeps moving.
- Corrections are rationed: at most one per turn, and only when the mistake is
  worth fixing. They appear as a dimmed `TIP:` line under the reply.
- Qwen3's thinking mode is switched off (`reasoning_effort: "none"`), which cuts
  time-to-first-token dramatically for short conversational replies.

## Configuration

Copy `.env.example` to `.env` to override. All values have defaults:

| Variable | Default | Notes |
| --- | --- | --- |
| `GERMAN_BOT_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible server |
| `GERMAN_BOT_MODEL` | `qwen3.8:27b` | Tag as your server knows it |
| `GERMAN_BOT_LEVEL` | `A1` | Starting level |
| `GERMAN_BOT_API_KEY` | `local` | Ignored by local servers |

Because it speaks the OpenAI protocol, you can point `GERMAN_BOT_BASE_URL` at LM
Studio (`:1234/v1`), llama.cpp (`:8080/v1`) or vLLM (`:8000/v1`) without code
changes.

## Layout

```
main.py              entry point
frau_agent/agent.py  model wiring, level, system prompt
frau_agent/cli.py    terminal loop, streaming, /commands
```
