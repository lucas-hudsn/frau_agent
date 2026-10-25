"""Local speech-to-text for the microphone, with the signal you need for feedback.

Design notes
------------
* **Whistle** (Cactus Compute) is a 16.9 MB speech model in a single `.cact`
  file on the CPU, driven by a small native engine. No GPU, no cloud, no API
  key. Engine and weights are fetched once from Hugging Face into
  `~/.cache/cactus-needle/` and then work offline.
* We ask Whistle for **word timestamps**, not just a transcript. Timings give us
  speaking pace and pauses; per-word probabilities give us the words the
  recogniser was unsure about. That is what makes feedback possible from a
  recogniser that is otherwise a black box.
* One pass covers 30 s of audio. Longer clips are windowed by the engine, which
  cuts at the last full sentence, so `MAX_SECONDS` is clamped there: a take is
  short by design and we would rather cap it than lose its middle.
* Honest caveat: an ASR probability is *not* a pronunciation score. A word can
  be low-confidence because the microphone was far away, because the word is
  rare, or because the learner said it wrong. So the feedback we surface is
  "the recogniser struggled here, say it again" rather than "your
  pronunciation is 62% correct". Anything stronger needs a forced aligner and
  a reference recording — much heavier, much less forgiving.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy is only needed once we touch the microphone
    import numpy as np

SAMPLE_RATE = 16_000
BLOCK = 1024  # 64 ms per callback: small enough to stop on silence promptly
# A tuned `whistle.cact`; unset runs the published weights.
WEIGHTS = os.getenv("GERMAN_BOT_STT_WEIGHTS") or None
# Below this per-word probability we flag the word. Whistle is usually 0.9+ on
# clear speech, so 0.55 is deliberately conservative: we only point at words the
# recogniser genuinely stumbled on.
CONFIDENCE = float(os.getenv("GERMAN_BOT_STT_CONFIDENCE", "0.55"))
# A gap longer than this between two words counts as a hesitation.
PAUSE = 0.6

# Recording guards.
LEAD_IN = 4.0  # seconds to wait for the learner to start talking
SILENCE = float(os.getenv("GERMAN_BOT_STT_SILENCE", "1.8"))
MAX_SECONDS = min(float(os.getenv("GERMAN_BOT_STT_MAX_SECONDS", "20")), 30.0)

# The app promises no network after setup; the engine would otherwise post an
# anonymous usage count per transcription. Set before `needle` is imported.
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")


class STTError(RuntimeError):
    """Speech input failed in a way the caller should report, not crash on."""


class NoSpeechError(STTError):
    """The microphone heard nothing worth transcribing."""


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class Transcript:
    text: str
    words: tuple[Word, ...]
    audio_seconds: float

    # -- pace ---------------------------------------------------------------
    @property
    def span(self) -> float:
        """Seconds between the first word starting and the last one ending."""
        if not self.words:
            return 0.0
        return self.words[-1].end - self.words[0].start

    @property
    def wpm(self) -> float:
        """Words per minute of actual speech, silence at the edges excluded."""
        if self.span < 0.5:
            return 0.0
        return len(self.words) / self.span * 60

    # -- what went wrong ----------------------------------------------------
    def uncertain(self, threshold: float = CONFIDENCE) -> list[Word]:
        return [w for w in self.words if w.probability < threshold]

    def pauses(self, minimum: float = PAUSE) -> list[float]:
        """Gaps between words, as (seconds), longest first."""
        gaps = [
            b.start - a.end
            for a, b in zip(self.words, self.words[1:])
            if b.start - a.end > minimum
        ]
        return sorted(gaps, reverse=True)

    def summary(self, threshold: float = CONFIDENCE) -> list[tuple[str, str]]:
        """Feedback as (text, rich-style) pairs, ready for the terminal.

        Kept as data so this module owns the thresholds and the CLI owns the
        rendering.
        """
        out: list[tuple[str, str]] = []

        wpm = self.wpm
        if wpm == 0:
            pace = "too short for a pace"
        elif wpm < 60:
            pace = f"{wpm:.0f} words/min — a bit halting"
        elif wpm > 165:
            pace = f"{wpm:.0f} words/min — quite fast"
        else:
            pace = f"{wpm:.0f} words/min — good pace"
        out.append((pace, "dim"))

        longest = self.pauses()
        if longest and longest[0] >= 1.0:
            out.append((f"longest pause {longest[0]:.1f} s", "dim"))

        unsure = self.uncertain(threshold)
        if unsure:
            words = ", ".join(w.text.strip(".,!?;:…") for w in unsure[:3])
            more = "" if len(unsure) <= 3 else f" (+{len(unsure) - 3})"
            out.append((f"unclear: {words}{more}", "yellow"))
        return out


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
def _rms(block) -> float:
    if not len(block):
        return 0.0
    return float(math.sqrt(float((block * block).mean())))


def record() -> np.ndarray:
    """Record until the learner stops talking; returns 16 kHz mono float32.

    Auto-stop (voice activity detection) instead of push-to-talk: for a language
    learner, hunting for a key to stop is extra friction, and we want both hands
    free for gesturing.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise STTError("sounddevice is missing: `uv add sounddevice`") from exc

    frames: list = []

    def callback(indata, _frames, _time, status) -> None:  # runs on PortAudio thread
        if status:  # overflow/underflow: drop the chunk rather than the take
            return
        frames.append(indata[:, 0].copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK,
            callback=callback,
        ):
            start = time.monotonic()
            seen = 0
            speech_at = last_loud_at = 0.0
            noise = 0.01

            while True:
                sd.sleep(40)
                now = time.monotonic() - start
                if now < 0.5:  # calibrate against the room, not against the voice
                    noise = min(noise, max(_rms(frames[-1]), 1e-4)) if frames else noise
                    continue

                loud = max((_rms(f) for f in frames[seen:]), default=0.0)
                seen = len(frames)
                threshold = max(noise * 3.5, 0.008)

                if not speech_at:
                    if loud > threshold:
                        speech_at = last_loud_at = now
                    elif now > LEAD_IN:
                        raise NoSpeechError("no speech detected")
                elif loud > threshold:
                    last_loud_at = now
                elif now - last_loud_at > SILENCE:
                    break

                if speech_at and now - speech_at > MAX_SECONDS:
                    break

        if not frames or not speech_at:
            raise NoSpeechError("no speech detected")
    except STTError:
        raise
    except Exception as exc:  # PortAudioError: no mic, or macOS said no
        raise STTError(f"microphone unavailable: {exc}") from exc

    audio = np.concatenate(frames)
    # Trim the lead-in and the trailing silence we waited out.
    mask = np.abs(audio) > 0.008
    if mask.any():
        first, last = int(np.argmax(mask)), len(mask) - int(np.argmax(mask[::-1]))
        audio = audio[max(0, first - SAMPLE_RATE // 4) : last + SAMPLE_RATE // 4]
    return audio


# --------------------------------------------------------------------------- #
# transcription
# --------------------------------------------------------------------------- #
def _weights_file() -> Path | None:
    """Where the weights live, without loading the engine."""
    if WEIGHTS:
        return Path(WEIGHTS)
    if local := os.environ.get("NEEDLE_WHISTLE_WEIGHTS"):
        return Path(local)
    try:
        from needle.agent import fetch
    except ImportError:  # cannot tell, so do not nag
        return None
    return Path(fetch.cache_dir(fetch.WHISTLE)) / fetch.base_weights(fetch.WHISTLE)


def model_cached() -> bool:
    """Whether the weights are already on disk. First run downloads them."""
    path = _weights_file()
    return path is None or path.exists()


def transcribe(audio: np.ndarray, language: str = "de") -> Transcript:
    """Run Whistle over a 16 kHz mono float32 array."""
    try:
        import needle
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise STTError("cactus-needle is missing: `uv add cactus-needle`") from exc

    try:
        result = needle.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            weights=WEIGHTS,
        )
    except Exception as exc:  # weights missing, malformed samples, engine error
        raise STTError(f"recognition failed: {exc}") from exc

    words = tuple(
        Word(w["word"].strip(), float(w["start"]), float(w["end"]), float(w["probability"]))
        for w in result.get("words") or ()
        if w["word"].strip()
    )
    if not words:  # silence and steady noise come back empty, not invented
        raise NoSpeechError("nothing understood")

    seconds = len(audio) / SAMPLE_RATE
    return Transcript(result.get("text") or _join(words), words, float(seconds))


def _join(words: Iterable[Word]) -> str:
    """Glue token-level words back into a sentence when the engine sent no text."""
    text = " ".join(w.text for w in words)
    for punct in (",", ".", "?", "!", ";", ":", "…"):
        text = text.replace(f" {punct}", punct)
    return text[:1].upper() + text[1:] if text else text
