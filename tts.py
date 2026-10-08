"""Generate a WAV file from text using Pocket TTS."""

import argparse
import os
from collections.abc import Iterator
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Event, Lock


MODEL_DIR = Path(os.environ.get("POCKET_TTS_MODEL_DIR", "./model")).expanduser().resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face reads this setting during import, so configure it first.
os.environ["HF_HOME"] = str(MODEL_DIR)

import scipy.io.wavfile  # noqa: E402
from pocket_tts import TTSModel  # noqa: E402


_generation_lock = Lock()
ENGLISH_VOICES = (
    "alba",
    "anna",
    "azelma",
    "bill_boerst",
    "caro_davy",
    "charles",
    "cosette",
    "eponine",
    "eve",
    "fantine",
    "george",
    "jane",
    "javert",
    "jean",
    "marius",
    "mary",
    "michael",
    "paul",
    "peter_yearsley",
    "stuart_bell",
    "vera",
)


def read_script(value: str) -> str:
    path = Path(value)
    return path.read_text(encoding="utf-8").strip() if path.is_file() else value.strip()


@lru_cache(maxsize=1)
def get_tts() -> TTSModel:
    return TTSModel.load_model()


@lru_cache(maxsize=len(ENGLISH_VOICES))
def get_voice_state(voice: str):
    if voice not in ENGLISH_VOICES:
        raise ValueError(f"unknown voice: {voice}")
    return get_tts().get_state_for_audio_prompt(voice)


def get_sample_rate() -> int:
    return get_tts().sample_rate


def stream_audio_chunks(
    script: str,
    voice: str = "eve",
    stop: Event | None = None,
) -> Iterator[bytes]:
    if not script.strip():
        raise ValueError("script cannot be empty")

    with _generation_lock:
        model = get_tts()
        voice_state = get_voice_state(voice)
        for audio in model.generate_audio_stream(
            voice_state,
            script.strip(),
            stop=stop,
        ):
            pcm = (
                audio.detach()
                .cpu()
                .float()
                .contiguous()
                .numpy()
                .reshape(-1)
                .astype("<f4", copy=False)
            )
            yield pcm.tobytes()


def generate_wav(script: str, voice: str = "eve") -> bytes:
    if not script.strip():
        raise ValueError("script cannot be empty")

    with _generation_lock:
        model = get_tts()
        voice_state = get_voice_state(voice)
        audio = model.generate_audio(voice_state, script.strip())

    output = BytesIO()
    scipy.io.wavfile.write(output, model.sample_rate, audio.numpy())
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tts_output.wav from text.")
    parser.add_argument("script", help="Quoted text or a path to a UTF-8 text file")
    args = parser.parse_args()

    script = read_script(args.script)
    if not script:
        parser.error("script cannot be empty")

    output_path = Path("tts_output.wav").resolve()
    output_path.write_bytes(generate_wav(script))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
