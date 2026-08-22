"""合成哼唱渲染器:把 MIDI 音符序列渲染成带哼唱特征的波形。

哼唱与 MIDI 直渲的差异集中在四处,每处一个可控旋钮:
逐音恒定音高偏移(音分)、平滑的节奏漂移(秒)、音符间滑音(秒)、颤音(音分/Hz)。
扰动强度可扫描,ground truth(实际渲染的音高/时间)随结果一并返回。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .melody import NoteEvent, smooth_random_curve


@dataclass
class HummingParams:
    sr: int = 22050
    detune_cents: float = 25.0  # 每个音符随机音高偏移的标准差(音分)
    tempo_wander: float = 0.03  # 节奏漂移强度:平滑时间扰动的标准差(秒)
    glide_s: float = 0.06  # 相邻音符间滑音时长(秒)
    vibrato_rate_hz: float = 5.0
    vibrato_depth_cents: float = 30.0  # 颤音深度(音分,单音内渐入)
    amplitude: float = 0.25
    noise_level: float = 0.0  # 呼吸噪声(相对振幅)
    seed: int | None = None


@dataclass
class HummingResult:
    wav: np.ndarray  # float32 单声道
    sr: int
    notes: list[NoteEvent]  # 实际渲染的音符(已含音高偏移与时间扰动)
    times: np.ndarray  # 逐采样时间轴
    pitch_track: np.ndarray  # 逐采样真值音高(MIDI 单位,无声段为 nan)


def _time_warp(times: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """平滑随机时间扰动(复用 melody 模块的共享实现)。"""
    return smooth_random_curve(times, strength, rng, corr_s=0.5, rate=50.0)


def _warp_notes(
    notes: list[NoteEvent], params: HummingParams, rng: np.random.Generator
) -> list[NoteEvent]:
    """先施加节奏漂移(时间轴扰动),再施加逐音音高偏移,保持时序单调。"""
    if not notes:
        return []
    rate = 50  # 时间扰动控制率(Hz)
    t_end = notes[-1].end + 0.5
    ctrl_times = np.arange(0.0, t_end, 1.0 / rate)
    wander = _time_warp(ctrl_times, params.tempo_wander, rng)

    warped: list[NoteEvent] = []
    for note in notes:
        start = max(0.0, note.start + float(np.interp(note.start, ctrl_times, wander)))
        end = max(0.0, note.end + float(np.interp(note.end, ctrl_times, wander)))
        if warped:
            start = max(start, warped[-1].end)
        end = max(end, start + 0.04)  # 最短音长,防止扰动后退化
        detuned = note.pitch + rng.normal(0.0, params.detune_cents / 100.0)
        warped.append(NoteEvent(detuned, start, end))
    return warped


def render_humming(notes: list[NoteEvent], params: HummingParams | None = None) -> HummingResult:
    """把音符序列渲染为哼唱波形;返回波形与实际渲染参数(ground truth)。"""
    params = params or HummingParams()
    rng = np.random.default_rng(params.seed)
    performed = _warp_notes(notes, params, rng)

    total = int(np.ceil((performed[-1].end + 0.2) * params.sr)) if performed else params.sr // 2
    times = np.arange(total) / params.sr
    pitch_track = np.full(total, np.nan)
    envelope = np.zeros(total)
    n = np.arange(total)

    attack_s, release_s = 0.015, 0.03
    for i, note in enumerate(performed):
        i0 = int(note.start * params.sr)
        i1 = min(int(note.end * params.sr), total)
        if i1 <= i0:
            continue
        seg = slice(i0, i1)
        t_in = (n[seg] - i0) / params.sr  # 音内时间
        dur = note.duration

        pitch = np.full(i1 - i0, note.pitch)
        if i > 0 and params.glide_s > 0:  # 从前一音滑入
            prev_pitch = performed[i - 1].pitch
            w = np.clip(t_in / params.glide_s, 0.0, 1.0)
            pitch = prev_pitch + (note.pitch - prev_pitch) * w
        if params.vibrato_depth_cents > 0:  # 颤音在音内渐入
            ramp = np.clip(3.0 * t_in / max(dur, 1e-6), 0.0, 1.0)
            phase0 = rng.uniform(0.0, 2.0 * np.pi)
            pitch = pitch + (params.vibrato_depth_cents / 100.0) * ramp * np.sin(
                2.0 * np.pi * params.vibrato_rate_hz * t_in + phase0
            )
        pitch_track[seg] = pitch

        env = np.ones(i1 - i0)
        n_attack = min(int(attack_s * params.sr), env.size)
        n_release = min(int(release_s * params.sr), env.size)
        if n_attack > 0:
            env[:n_attack] = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_attack) / n_attack)
        if n_release > 0:
            env[-n_release:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n_release) / n_release)
        envelope[seg] = env * (1.0 + rng.uniform(-0.15, 0.15))

    freq = 440.0 * 2.0 ** ((np.nan_to_num(pitch_track, nan=69.0) - 69.0) / 12.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / params.sr
    wav = params.amplitude * envelope * np.sin(phase)
    if params.noise_level > 0:
        wav = wav + params.noise_level * params.amplitude * rng.standard_normal(total)
    return HummingResult(wav.astype(np.float32), params.sr, performed, times, pitch_track)
