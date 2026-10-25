# Frau Agent

A local German conversation partner. Type or speak — everything runs on your
machine via [pydantic-ai](https://ai.pydantic.dev) + Ollama, with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) for the microphone.
No cloud calls.

## Run

```sh
uv sync          # first time only: installs pydantic-ai, faster-whisper, sounddevice
uv run main.py
# or, after `uv sync`, simply:
uv run frau

# speak instead of typing:
uv run frau --voice

# or force text when GERMAN_BOT_VOICE=1 is set:
uv run frau --text
```

Python ≥ 3.14, macOS or Linux with a working microphone. Requires the chat model
to be pulled once: `ollama pull qwen3.8:27b` (already present on this machine).
Speaking also downloads a Whisper model once (`small`, ~460 MB on disk, ~250 MB
in RAM as int8) and then works offline.

## Commands

| Command | Effect |
| --- | --- |
| `/voice` | Speak one answer, transcribe it, send it |
| `/mic` | Toggle speaking for every answer |
| `/level A1\|A2\|B1` | Change difficulty, resets the conversation |
| `/reset` | Fresh conversation, same level |
| `/history` | How many messages are in context |
| `/help` | Command list |
| `/quit` | Exit |

## Voice input

`/voice` (or Enter in `/mic` mode) records until you stop talking, then
transcribes locally. No push-to-talk key: a 1.8 s silence ends the take,
with a 4 s window to start and a 20 s cap.

Because Whisper is asked for **word timestamps and per-word probabilities**,
each transcript comes with a small feedback line:

```
🎤 “Hallo, ich heiße Lukas. Ich wohne in Berlin und arbeite als Datenwissenschaftler.”
    148 Wörter/Min — gutes Tempo
```

```
🎤 “Ich habe gestern ein Brötchen gegessen.”
    62 Wörter/Min — etwas stockend   ·   längste Pause 1.3 s   ·   unklar: Brötchen
```

- **Pace** (`Wörter/Min`) is words divided by the time between the first and
  last word, so silence at the edges does not skew it. Under ~60 or over ~165
  gets a note.
- **Pauses** longer than 0.6 s are counted; the longest is shown once it passes
  a second. Hesitations are normal at A1 — they are information, not a verdict.
- **`unklar`** lists words the recogniser was below 0.55 on. Low confidence
  usually means the word was mumbled, unusual, or hit by noise, so treat it as
  "say that again", not as a pronunciation grade. Real pronunciation scoring
  needs a forced aligner and a reference recording, which is a much heavier
  model than this.

While in voice mode the system prompt is extended so the tutor does not correct
punctuation, capitalisation, or words the recogniser may simply have misheard.

### Why faster-whisper

| Option | Why not |
| --- | --- |
| macOS `SFSpeechRecognizer` | Free and instant, but iOS/macOS only, word-level signal is shaky, and it can silently fall back to Apple's servers |
| `vosk` | ~50 MB and streaming, but German accuracy at A1 speed is well below Whisper |
| `mlx-whisper` | Fastest on Apple Silicon, but Apple-only; the MLX runtime is a heavy dependency for a gain that is under a second on a 7 s take |
| `whisper.cpp` | Great C++ runtime, but the Python bindings are thin: you end up hand-rolling the timestamp/confidence plumbing this app depends on |

faster-whisper (CTranslate2) keeps the weights in int8, gives word timings and
probabilities as first-class output, and runs ~6× faster than realtime on this
machine's CPU: about 1 s for a 7 s answer, plus ~0.7 s the first time the model
loads.

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
| `GERMAN_BOT_VOICE` | unset | `1` starts in voice mode (same as `--voice`) |
| `GERMAN_BOT_STT_MODEL` | `small` | `base` is ~3× faster and ~140 MB, noticeably worse German |
| `GERMAN_BOT_STT_COMPUTE` | `int8` | `float32` if int8 misbehaves on your CPU |
| `GERMAN_BOT_STT_CONFIDENCE` | `0.55` | Below this a word is flagged `unklar` |
| `GERMAN_BOT_STT_SILENCE` | `1.8` | Seconds of quiet that end a recording |
| `GERMAN_BOT_STT_MAX_SECONDS` | `20` | Hard cap on one take |

Because it speaks the OpenAI protocol, you can point `GERMAN_BOT_BASE_URL` at LM
Studio (`:1234/v1`), llama.cpp (`:8080/v1`) or vLLM (`:8000/v1`) without code
changes.

## Layout

```
main.py              entry point
frau_agent/agent.py  model wiring, level, system prompt
frau_agent/cli.py    terminal loop, streaming, /commands, voice
frau_agent/stt.py    microphone, Whisper, transcript timings + feedback
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Mikrofon nicht verfügbar` | macOS prompts once for mic access; if you dismissed it, enable your terminal under System Settings → Privacy & Security → Microphone, then restart the terminal |
| `Nichts gehört` | You spoke quieter than the room's noise floor, or waited longer than the 4 s lead-in. Raise input gain, or speak right after the prompt |
| Takes stop too soon | `GERMAN_BOT_STT_SILENCE=2.5` gives more thinking room |
| Wrong words in the transcript | Try `GERMAN_BOT_STT_MODEL=medium`; `base`/`tiny` hallucinate on short German answers |
| No `unklar` flags ever | Expected on clean speech — Whisper is usually above 0.9, and flagging nothing is the right answer |

