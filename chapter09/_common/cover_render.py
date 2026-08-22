"""合成翻唱渲染:同一首民歌在四根变化轴(音色/调性/速度/织体)下的可控版本。

音色/调性/速度三轴只动声学表面,音高级骨架不变;织体轴增删伴奏声部,直接改和声内容。
速度轴必须用非整数倍变速并带轻微 rubato——固定 tempo 渲染下节拍跟踪几乎必对,
"节拍同步 vs 固定帧率"的消融就测不出差异(写作指南 §6.3)。
渲染走第 3 章工具链:pretty_midi 组 MIDI → fluidsynth CLI + TimGM6mb SoundFont → wav。
"""
from __future__ import annotations

import subprocess
import tempfile
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pretty_midi

from .melody import NoteEvent, load_midi_notes, smooth_random_curve

CODE_ROOT = Path(__file__).resolve().parents[2]

MELODY_PROGRAM = 73  # 长笛,作为原版音色
ACCOMP_PROGRAM = 48  # 弦乐合奏,伴奏声部
CHORD_WINDOW_S = 1.0  # 朴素配和弦的时间窗(120 BPM 下约两拍)


@dataclass
class CoverSpec:
    """一个翻唱版本的全部轴参数;默认值即原版(v0)。"""

    version: str = "v0_original"
    program: int = MELODY_PROGRAM  # 主旋律 GM 音色号
    transpose: int = 0  # 移调(半音)
    tempo_ratio: float = 1.0  # 速度倍率(>1 更快;速度轴取非整数倍)
    rubato: float = 0.0  # 节奏弹性强度(秒)
    texture: str = "none"  # none | octave | drone | triads | reharm
    seed: int | None = None


# 每首民歌 6 个版本:v0 原版 + 音色/调性/速度各一版 + 织体轴两端。
# 织体轴中间档(drone/triads)与移调、速度的扫描版本由 Notebook 按需调用 render_cover 生成。
VERSION_PRESETS: dict[str, dict] = {
    "v0_original": {},
    "v1_timbre": {"program": 40},  # 小提琴
    "v2_key": {"transpose": 4},
    "v3_tempo": {"tempo_ratio": 1.37, "rubato": 0.03},
    "v4_texture_octave": {"texture": "octave"},
    "v5_texture_reharm": {"texture": "reharm"},
}


def resolve_soundfont() -> Path:
    """TimGM6mb:优先项目 datasets 副本,否则用 pretty_midi 自带的那份(同第 3 章)。"""
    project_copy = CODE_ROOT / "datasets" / "soundfonts" / "TimGM6mb.sf2"
    if project_copy.exists():
        return project_copy
    return Path(pretty_midi.__file__).with_name("TimGM6mb.sf2")


def _estimate_tonic(notes: list[NoteEvent]) -> int:
    """以结束音的音高级作主音(民歌大多收束于主音)。"""
    return int(round(notes[-1].pitch)) % 12


def _chord_candidates(tonic: int, quality: str = "major") -> list[list[int]]:
    """主音上的 I/IV/V(大调)或 i/iv/v(平行小调)三和弦(音高级)。"""
    if quality == "minor":
        return [
            [(tonic + i) % 12 for i in (0, 3, 7)],
            [(tonic + i) % 12 for i in (5, 8, 0)],
            [(tonic + i) % 12 for i in (7, 10, 2)],
        ]
    return [
        [(tonic + i) % 12 for i in (0, 4, 7)],
        [(tonic + i) % 12 for i in (5, 9, 0)],
        [(tonic + i) % 12 for i in (7, 11, 2)],
    ]


def _accompaniment(notes: list[NoteEvent], texture: str) -> list[tuple[float, float, float]]:
    """生成伴奏声部 (pitch, start, end) 列表;朴素编配,仅供构造织体差异。

    octave:旋律低八度叠置(音高级不变,织体变密)。
    drone:主音与属音长音持续(低两个八度)。
    triads:每个时间窗按旋律音高级分布从 I/IV/V 中选重叠最多的和弦。
    reharm:同一套选和弦逻辑,但和弦改从平行小调的 i/iv/v 中选——和声色彩改变,
    伴奏引入的音高级随之改变(模拟重编曲对色度匹配的破坏)。
    """
    if texture == "none" or not notes:
        return []
    t0, t1 = notes[0].start, notes[-1].end
    tonic = _estimate_tonic(notes)
    base = tonic + 36  # 伴奏区:低两个八度附近

    if texture == "octave":
        return [(n.pitch - 12, n.start, n.end) for n in notes]
    if texture == "drone":
        return [(float(base), t0, t1), (float(base + 7), t0, t1)]

    candidates = _chord_candidates(tonic, quality="minor" if texture == "reharm" else "major")
    chords: list[tuple[float, float, float]] = []
    window_start = t0
    while window_start < t1:
        window_end = min(window_start + CHORD_WINDOW_S, t1)
        weights = np.zeros(12)
        for n in notes:
            overlap = max(0.0, min(n.end, window_end) - max(n.start, window_start))
            if overlap > 0:
                weights[int(round(n.pitch)) % 12] += overlap
        scores = [float(weights[pc].sum()) for pc in candidates]
        pick = int(np.argmax(scores))
        for pc in candidates[pick]:
            pitch = float(base + (pc - tonic) % 12)
            chords.append((pitch, window_start, window_end))
            if texture == "reharm":
                chords.append((pitch + 12, window_start, window_end))  # 加厚声部,和声权重更大
        window_start = window_end
    return chords


def build_cover_midi(notes: list[NoteEvent], spec: CoverSpec) -> pretty_midi.PrettyMIDI:
    """按轴参数变换旋律并加伴奏,组装成临时 MIDI 对象。"""
    rng = np.random.default_rng(spec.seed)
    transformed = [replace(n, pitch=n.pitch + spec.transpose) for n in notes]

    if spec.tempo_ratio != 1.0:
        transformed = [
            replace(n, start=n.start / spec.tempo_ratio, end=n.end / spec.tempo_ratio) for n in transformed
        ]
    if spec.rubato > 0 and transformed:
        ctrl_times = np.arange(0.0, transformed[-1].end + 0.5, 1.0 / 50.0)
        wander = smooth_random_curve(ctrl_times, spec.rubato, rng)
        warped: list[NoteEvent] = []
        for n in transformed:
            start = max(0.0, n.start + float(np.interp(n.start, ctrl_times, wander)))
            end = max(0.0, n.end + float(np.interp(n.end, ctrl_times, wander)))
            if warped:
                start = max(start, warped[-1].end)
            warped.append(replace(n, start=start, end=max(end, start + 0.04)))
        transformed = warped

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    melody = pretty_midi.Instrument(program=spec.program, name="melody")
    for n in transformed:
        melody.notes.append(pretty_midi.Note(80, int(round(n.pitch)), n.start, n.end))
    pm.instruments.append(melody)

    chords = _accompaniment(transformed, spec.texture)
    if chords:
        accomp = pretty_midi.Instrument(program=ACCOMP_PROGRAM, name="accompaniment")
        for pitch, start, end in chords:
            accomp.notes.append(pretty_midi.Note(55, int(round(pitch)), start, end))
        pm.instruments.append(accomp)
    return pm


def render_cover(
    midi_path: Path | str,
    spec: CoverSpec,
    out_wav: Path | str,
    sr: int = 22050,
    soundfont: Path | str | None = None,
) -> dict:
    """渲染一个版本到 wav,返回记录各轴取值的元信息 dict。"""
    soundfont = Path(soundfont) if soundfont else resolve_soundfont()
    notes = load_midi_notes(midi_path)
    pm = build_cover_midi(notes, spec)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_midi = tmp.name
    try:
        pm.write(tmp_midi)
        cmd = ["fluidsynth", "-F", str(out_wav), "-r", str(sr), "-g", "2.0", "-ni", str(soundfont), tmp_midi]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"FluidSynth 渲染失败:\n{result.stderr}")
    finally:
        Path(tmp_midi).unlink(missing_ok=True)

    return {
        "song": Path(midi_path).stem,
        "wav": str(out_wav),
        **{k: getattr(spec, k) for k in ("version", "program", "transpose", "tempo_ratio", "rubato", "texture")},
    }


def render_cover_set(
    midi_path: Path | str,
    out_dir: Path | str,
    presets: dict[str, dict] | None = None,
    sr: int = 22050,
) -> list[dict]:
    """渲染一首歌的整套预设版本(默认 6 版),返回每版元信息。"""
    presets = presets or VERSION_PRESETS
    out_dir = Path(out_dir)
    records = []
    for version, overrides in presets.items():
        spec = CoverSpec(version=version, seed=zlib.crc32(f"{midi_path}|{version}".encode()) % (2**31), **overrides)
        wav_path = out_dir / Path(midi_path).stem / f"{version}.wav"
        records.append(render_cover(midi_path, spec, wav_path, sr=sr))
    return records
