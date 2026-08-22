"""FluidSynth command helpers for MIDI-to-audio control demos."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def render_command(
    midi_path: Path | str,
    soundfont_path: Path | str,
    output_wav: Path | str,
    sample_rate: int = 44100,
) -> list[str]:
    """Return a FluidSynth command without executing it."""
    return [
        "fluidsynth",
        "-ni",
        str(soundfont_path),
        str(midi_path),
        "-F",
        str(output_wav),
        "-r",
        str(int(sample_rate)),
    ]


def fluidsynth_available() -> bool:
    """Return true when the FluidSynth executable is on PATH."""
    return shutil.which("fluidsynth") is not None


def render_status(soundfont_path: Path | str | None = None) -> dict[str, object]:
    """Return a table-friendly MIDI renderer status row."""
    soundfont_ok = True if soundfont_path is None else Path(soundfont_path).expanduser().exists()
    return {
        "renderer": "fluidsynth",
        "command_available": fluidsynth_available(),
        "soundfont_available": soundfont_ok,
        "soundfont_path": "" if soundfont_path is None else str(soundfont_path),
    }


def render_midi(
    midi_path: Path | str,
    soundfont_path: Path | str,
    output_wav: Path | str,
    sample_rate: int = 44100,
) -> list[str]:
    """Render MIDI to WAV with FluidSynth and return the executed command."""
    command = render_command(midi_path, soundfont_path, output_wav, sample_rate=sample_rate)
    if not fluidsynth_available():
        raise FileNotFoundError("FluidSynth executable not found on PATH.")
    if not Path(soundfont_path).expanduser().exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return command
