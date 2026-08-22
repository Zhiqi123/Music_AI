"""第一章八个音符事件的单一数据源。"""

from __future__ import annotations


BASE_NOTE_EVENTS = (
    {"onset_beat": 0.0, "pitch_midi": 60, "duration_beat": 1.0, "velocity_midi": 72},
    {"onset_beat": 1.0, "pitch_midi": 62, "duration_beat": 0.5, "velocity_midi": 68},
    {"onset_beat": 1.5, "pitch_midi": 64, "duration_beat": 0.5, "velocity_midi": 80},
    {"onset_beat": 2.0, "pitch_midi": 67, "duration_beat": 1.0, "velocity_midi": 90},
    {"onset_beat": 3.0, "pitch_midi": 69, "duration_beat": 1.0, "velocity_midi": 76},
    {"onset_beat": 4.0, "pitch_midi": 67, "duration_beat": 0.5, "velocity_midi": 84},
    {"onset_beat": 4.5, "pitch_midi": 64, "duration_beat": 0.5, "velocity_midi": 78},
    {"onset_beat": 5.0, "pitch_midi": 60, "duration_beat": 2.0, "velocity_midi": 70},
)


def make_note_events(transpose_semitones: int = 0) -> list[dict[str, float | int]]:
    """返回独立的事件字典，并按半音数移调 MIDI 音高。"""
    if isinstance(transpose_semitones, bool) or not isinstance(transpose_semitones, int):
        raise TypeError("transpose_semitones 必须为整数")

    events = [dict(event) for event in BASE_NOTE_EVENTS]
    for event in events:
        event["pitch_midi"] = int(event["pitch_midi"]) + transpose_semitones

    if not all(0 <= event["pitch_midi"] <= 127 for event in events):
        raise ValueError("移调后的 MIDI 音高必须在 0 至 127 之间")
    return events
