"""Local speech-to-text for the microphone, with the signal you need for feedback.

Design notes
------------
* **faster-whisper** (CTranslate2) runs Whisper on the CPU. No cloud, no API key,
  ~140 MB for `base`, ~460 MB for `small` on disk; int8 halves that in RAM.
  The model is downloaded once from Hugging Face and then works offline.
* We ask Whisper for **word timestamps**, not just a transcript. Timings give us
  speaking pace and pauses; per-word probabilities give us the words the
  recogniser was unsure about. That is what makes feedback possible from a
  recogniser that is otherwise a black box.
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
MODEL_SIZE = os.getenv("GERMAN_BOT_STT_MODEL", "small")
COMPUTE_TYPE = os.getenv("GERMAN_BOT_STT_COMPUTE", "int8")
# Below this per-word probability we flag the word. Whisper is usually 0.9+
# on clear speech, so 0.55 is deliberately conservative: we only point at words
# the recogniser genuinely stumbled on.
CONFIDENCE = float(os.getenv("GERMAN_BOT_STT_CONFIDENCE", "0.55"))
# A gap longer than this between two words counts as a hesitation.
PAUSE = 0.6

# Recording guards.
LEAD_IN = 4.0  # seconds to wait for the learner to start talking
SILENCE = float(os.getenv("GERMAN_BOT_STT_SILENCE", "1.8"))
MAX_SECONDS = float(os.getenv("GERMAN_BOT_STT_MAX_SECONDS", "20"))


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
            pace = "zu kurz für ein Tempo"
        elif wpm < 60:
            pace = f"{wpm:.0f} Wörter/Min — etwas stockend"
        elif wpm > 165:
            pace = f"{wpm:.0f} Wörter/Min — recht schnell"
        else:
            pace = f"{wpm:.0f} Wörter/Min — gutes Tempo"
        out.append((pace, "dim"))

        longest = self.pauses()
        if longest and longest[0] >= 1.0:
            out.append((f"längste Pause {longest[0]:.1f} s", "dim"))

        unsure = self.uncertain(threshold)
        if unsure:
            words = ", ".join(w.text.strip(".,!?;:…") for w in unsure[:3])
            more = "" if len(unsure) <= 3 else f" (+{len(unsure) - 3})"
            out.append((f"unklar: {words}{more}", "yellow"))
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
        raise STTError("sounddevice fehlt: `uv add sounddevice`") from exc

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
                        raise NoSpeechError("keine Sprache erkannt")
                elif loud > threshold:
                    last_loud_at = now
                elif now - last_loud_at > SILENCE:
                    break

                if speech_at and now - speech_at > MAX_SECONDS:
                    break

        if not frames or not speech_at:
            raise NoSpeechError("keine Sprache erkannt")
    except STTError:
        raise
    except Exception as exc:  # PortAudioError: no mic, or macOS said no
        raise STTError(f"Mikrofon nicht verfügbar: {exc}") from exc

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
_models: dict[str, object] = {}


def _model(size: str = MODEL_SIZE):
    """Whisper is heavy to load, so keep one instance per size for the session."""
    from faster_whisper import WhisperModel  # imported lazily: slow dependency

    if size not in _models:
        _models[size] = WhisperModel(
            size,
            device="cpu",  # Metal is not wired up in CTranslate2; CPU is fine here
            compute_type=COMPUTE_TYPE,
            # Small utterances: a few decode threads are enough and keep the fans quiet.
            cpu_threads=min(4, os.cpu_count() or 4),
        )
    return _models[size]


def model_cached(size: str = MODEL_SIZE) -> bool:
    """Whether the weights are already on disk. First run downloads them."""
    if os.path.isdir(size):
        return True
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:  # cannot tell, so do not nag
        return True
    return (Path(HF_HUB_CACHE) / f"models--Systran--faster-whisper-{size}").exists()


def transcribe(audio: np.ndarray, size: str = MODEL_SIZE, language: str = "de") -> Transcript:
    """Run Whisper over a 16 kHz mono float32 array."""
    model = _model(size)
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=1,  # greedy: 1-2 s faster, and short turns do not need search
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,  # one utterance, no context to inherit
    )

    words: list[Word] = []
    for segment in segments:  # generator: this is where the work happens
        for word in segment.words or ():
            text = word.word.strip()
            if text:
                words.append(Word(text, float(word.start), float(word.end), float(word.probability)))

    if not words:
        raise NoSpeechError("nichts verstanden")

    seconds = len(audio) / SAMPLE_RATE
    return Transcript(_join(words), tuple(words), float(seconds))


def _join(words: Iterable[Word]) -> str:
    """Whisper emits token-level text; glue it back into a readable sentence."""
    text = " ".join(w.text for w in words)
    for punct in (",", ".", "?", "!", ";", ":", "…"):
        text = text.replace(f" {punct}", punct)
    return text[:1].upper() + text[1:] if text else text
