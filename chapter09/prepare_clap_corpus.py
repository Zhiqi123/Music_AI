"""一次性数据制备:构建 9.5 文本检索音频的 CLAP 检索语料(约 200 片段)。

四来源混建(全部库内已有,不进新数据集):
  CTIS 乐器独奏(胡琴家族作细粒度层 + 六类常见乐器)+ 二胡快/慢属性片段
  musdb18-hq 分轨(vocals/drums/bass/other,曲目与 9.2 指纹库同一批)
  GTZAN 十流派片段
  民歌合成渲染(cover_render 现场渲染三种音色)
所有片段统一为 48 kHz / 单声道 / 5.0 s / 峰值 0.5(CLAP 输入规格;采样率与时长先归一,
不让格式差混进检索信号),文案为模板化英文。产物落盘 CODE/datasets/clap_corpus_ch09/,
供 09_4 Notebook 直接读取;Notebook 运行时不再做采样与裁切。

运行(主 anaconda): python3 chapter09/prepare_clap_corpus.py   (从 CODE 根目录)
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from chapter09._common.cover_render import CoverSpec, render_cover  # noqa: E402

CTIS = CODE_ROOT / "datasets" / "CCMUSIC_CTIS" / "ccmusic_ctis_audio"
GTZAN = CODE_ROOT / "datasets" / "GTZAN" / "genres_original"
FINGERPRINT_MANIFEST = CODE_ROOT / "chapter09" / "data_manifests" / "fingerprint_library.csv"
FOLK_MIDI_DIR = CODE_ROOT / "datasets" / "melodies"
OUT_ROOT = CODE_ROOT / "datasets" / "clap_corpus_ch09"

SR = 48000
CLIP_S = 5.0
MIN_SRC_S = 4.0  # 短于此的源片段弃用
PEAK = 0.5
SEED = 42

# 胡琴家族(细粒度层):文件夹名 → 英文类名(拼音,CLAP 文本塔是否认识正是实验要测的)
HUQIN = {"二胡": "erhu", "板胡": "banhu", "中胡": "zhonghu", "高音板胡": "gaoyin banhu", "壮剧土胡": "tuhu"}
HUQIN_PER_CLASS = 8
# 其他常见乐器(丰富粗粒度层)
CTIS_OTHER = {"唢呐": "suona", "琵琶": "pipa", "中阮": "zhongruan", "古筝": "guzheng", "洞箫": "dongxiao", "高音键笙": "sheng"}
OTHER_PER_CLASS = 7
MUSDB_STEMS = {
    "vocals": "isolated singing voice",
    "drums": "drums only",
    "bass": "a bass track",
    "other": "other accompaniment instruments",
}
GTZAN_PER_GENRE = 4
FOLK_PROGRAMS = [(73, "flute"), (40, "violin"), (0, "piano")]


def safe_info(path: Path) -> "sf._SoundFileInfo | None":
    """sf.info 的容错封装(库内有已知的损坏文件,如 GTZAN jazz.00054.wav)。"""
    try:
        return sf.info(path)
    except Exception:
        return None


def loudest_window(path: Path, win_s: float, scan_sr: int = 8000) -> float:
    """低采样率快扫 RMS,返回能量最高的 win_s 窗口起点(避免裁到静默段)。"""
    y, sr = librosa.load(path, sr=scan_sr, mono=True)
    if len(y) <= int(win_s * sr):
        return 0.0
    hop = int(0.25 * sr)
    win = int(win_s * sr)
    energy = np.convolve(y**2, np.ones(win) / win, mode="valid")
    return float(np.argmax(energy[::hop]) * hop / sr)


def load_normalized_clip(path: Path) -> np.ndarray | None:
    """裁最响 5 s、重采样到 48 kHz 单声道、峰值归一;源损坏或不足 4 s 返回 None。"""
    info = safe_info(path)
    if info is None or info.duration < MIN_SRC_S:
        return None
    start = loudest_window(path, CLIP_S) if info.duration > CLIP_S + 0.5 else 0.0
    y, _ = librosa.load(path, sr=SR, mono=True, offset=start, duration=CLIP_S)
    if len(y) < int(CLIP_S * SR):
        y = np.pad(y, (0, int(CLIP_S * SR) - len(y)))
    peak = float(np.abs(y).max())
    if peak < 1e-4:
        return None
    return (y / peak * PEAK).astype(np.float32)


def origin_duration(path: Path) -> str:
    info = safe_info(path)
    return f"{info.duration:.3f}" if info else ""


def write_clip(y: np.ndarray, group: str, name: str) -> str:
    out = OUT_ROOT / group / f"{name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, SR)
    return str(out.relative_to(OUT_ROOT))


def collect_ctis(rng: random.Random) -> list[dict]:
    rows = []
    for folder, en in {**HUQIN, **CTIS_OTHER}.items():
        quota = HUQIN_PER_CLASS if folder in HUQIN else OTHER_PER_CLASS
        candidates = [p for p in sorted((CTIS / folder).glob("*.wav")) if (i := safe_info(p)) and i.duration >= MIN_SRC_S]
        for p in rng.sample(candidates, min(quota, len(candidates))):
            y = load_normalized_clip(p)
            if y is None:
                continue
            group = "ctis_huqin" if folder in HUQIN else "ctis_other"
            clip_id = f"{group}_{en.replace(' ', '')}_{len(rows):03d}"
            rows.append({
                "clip_id": clip_id,
                "wav_path": write_clip(y, group, clip_id),
                "source_group": group,
                "category": en,
                "caption_en": f"a solo {en} performance",
                "attribute": "",
                "origin_path": str(p.relative_to(CODE_ROOT)),
                "origin_duration_sec": origin_duration(p),
            })
    # 二胡快/慢属性片段(属性级实验层:文件名带"速度快/慢"标注)
    for attr, keyword in (("fast", "速度快"), ("slow", "速度慢")):
        for p in sorted((CTIS / "二胡").glob(f"*【二胡】*{keyword}*.wav")):
            y = load_normalized_clip(p)
            if y is None:
                continue
            clip_id = f"ctis_huqin_erhu_{attr}_{len(rows):03d}"
            rows.append({
                "clip_id": clip_id,
                "wav_path": write_clip(y, "ctis_huqin", clip_id),
                "source_group": "ctis_huqin",
                "category": "erhu",
                "caption_en": "a solo erhu performance",
                "attribute": attr,
                "origin_path": str(p.relative_to(CODE_ROOT)),
                "origin_duration_sec": origin_duration(p),
            })
    return rows


def collect_musdb() -> list[dict]:
    rows = []
    with FINGERPRINT_MANIFEST.open(encoding="utf-8") as f:
        tracks = [r["path"] for r in csv.DictReader(f) if r["role"] == "library"]
    for track_rel in tracks:
        track_dir = CODE_ROOT / track_rel.removesuffix("/mixture.wav")
        for stem, caption in MUSDB_STEMS.items():
            p = track_dir / f"{stem}.wav"
            y = load_normalized_clip(p)
            if y is None:
                continue
            track_name = track_dir.name.replace(" ", "_")
            clip_id = f"musdb_{stem}_{track_name[:24]}"
            rows.append({
                "clip_id": clip_id,
                "wav_path": write_clip(y, "musdb", clip_id),
                "source_group": "musdb",
                "category": stem,
                "caption_en": caption,
                "attribute": "",
                "origin_path": str(p.relative_to(CODE_ROOT)),
                "origin_duration_sec": origin_duration(p),
            })
    return rows


def collect_gtzan(rng: random.Random) -> list[dict]:
    rows = []
    for genre_dir in sorted(GTZAN.iterdir()):
        if not genre_dir.is_dir():
            continue
        files = sorted(genre_dir.glob("*.wav"))
        for p in rng.sample(files, min(GTZAN_PER_GENRE, len(files))):
            y = load_normalized_clip(p)
            if y is None:
                continue
            clip_id = f"gtzan_{genre_dir.name}_{p.stem.split('.')[-1]}"
            rows.append({
                "clip_id": clip_id,
                "wav_path": write_clip(y, "gtzan", clip_id),
                "source_group": "gtzan",
                "category": genre_dir.name,
                "caption_en": f"a {genre_dir.name} music excerpt",
                "attribute": "",
                "origin_path": str(p.relative_to(CODE_ROOT)),
                "origin_duration_sec": origin_duration(p),
            })
    return rows


def collect_folk_synth() -> list[dict]:
    rows = []
    tmp_dir = OUT_ROOT / "_tmp_render"
    for midi in sorted(FOLK_MIDI_DIR.glob("*.midi")):
        for program, timbre in FOLK_PROGRAMS:
            rec = render_cover(midi, CoverSpec(version=f"folk_{timbre}", program=program, seed=0), tmp_dir / f"{midi.stem}_{timbre}.wav", sr=SR)
            y = load_normalized_clip(Path(rec["wav"]))
            if y is None:
                continue
            clip_id = f"folk_{timbre}_{len(rows):03d}"
            rows.append({
                "clip_id": clip_id,
                "wav_path": write_clip(y, "folk_synth", clip_id),
                "source_group": "folk_synth",
                "category": f"synth_{timbre}",
                "caption_en": f"a synthesized Chinese folk melody on {timbre}",
                "attribute": "",
                "origin_path": str(midi.relative_to(CODE_ROOT)),
                "origin_duration_sec": "",
            })
    for p in tmp_dir.glob("*.wav"):
        p.unlink()
    tmp_dir.rmdir()
    return rows


def main() -> None:
    rng = random.Random(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += collect_ctis(rng)
    rows += collect_musdb()
    rows += collect_gtzan(rng)
    rows += collect_folk_synth()

    fieldnames = ["clip_id", "wav_path", "source_group", "category", "caption_en", "attribute", "origin_path", "origin_duration_sec"]
    with (OUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"共 {len(rows)} 片段 → {OUT_ROOT}")
    for group in ("ctis_huqin", "ctis_other", "musdb", "gtzan", "folk_synth"):
        n = sum(1 for r in rows if r["source_group"] == group)
        cats = sorted({r["category"] for r in rows if r["source_group"] == group})
        print(f"  {group:10s} {n:3d} 类清单 {cats}")
    attrs = [r for r in rows if r["attribute"]]
    print(f"  属性片段 {len(attrs)}(fast {sum(1 for r in attrs if r['attribute']=='fast')}, slow {sum(1 for r in attrs if r['attribute']=='slow')})")


if __name__ == "__main__":
    main()
