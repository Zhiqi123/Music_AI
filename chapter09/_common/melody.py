"""旋律音符序列的读取与裁剪(供合成哼唱、合成翻唱共用)。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.ndimage import gaussian_filter1d


@dataclass
class NoteEvent:
    """单个音符事件,pitch 为 MIDI 音高(允许小数,表示微偏),时间单位秒。"""

    pitch: float
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_midi_notes(path: Path | str) -> list[NoteEvent]:
    """读取单旋律 MIDI 为按开始时间排序的音符序列。

    Essen 与中国民歌库均为单声部;若存在交叠(重叠发音),截断先发的音符,
    保证输出单调不重叠,便于后续按"音高轮廓"处理。
    """
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(
        (NoteEvent(float(n.pitch), float(n.start), float(n.end)) for inst in pm.instruments for n in inst.notes),
        key=lambda n: (n.start, n.pitch),
    )
    mono: list[NoteEvent] = []
    for note in notes:
        if mono and note.start < mono[-1].end:
            mono[-1].end = note.start
            if mono[-1].duration <= 0:
                mono.pop()
        mono.append(note)
    return mono


def smooth_random_curve(
    times: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    corr_s: float = 0.5,
    rate: float = 50.0,
) -> np.ndarray:
    """平滑随机扰动曲线:高斯噪声经高斯核低通,标准差归一到 strength。

    合成哼唱的节奏漂移与合成翻唱的 rubato 共用;strength <= 0 时返回零曲线。
    """
    if strength <= 0 or times.size == 0:
        return np.zeros_like(times)
    noise = rng.standard_normal(times.size)
    smooth = gaussian_filter1d(noise, sigma=corr_s * rate)
    smooth /= np.std(smooth) + 1e-12
    return strength * smooth


def crop_notes(notes: list[NoteEvent], start: float, end: float) -> list[NoteEvent]:
    """截取 [start, end) 时间窗内的音符并平移到 0 起点(模拟从歌曲任意处开始的查询)。"""
    cropped = [
        replace(n, start=max(n.start, start) - start, end=min(n.end, end) - start)
        for n in notes
        if n.end > start and n.start < end
    ]
    return [n for n in cropped if n.duration > 0]


def transpose_notes(notes: list[NoteEvent], semitones: float) -> list[NoteEvent]:
    """整体移调(哼唱查询与库内旋律常不在同调,用于构造查询而非归一)。"""
    return [replace(n, pitch=n.pitch + semitones) for n in notes]
