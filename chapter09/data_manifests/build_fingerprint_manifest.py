"""生成 9.2 指纹库选曲清单:从 musdb18-hq train 按时长均匀取 10 首,另定 1 首库外负例。

产物: CODE/chapter09/data_manifests/fingerprint_library.csv
选曲规则: 按时长排序后等距取样(覆盖短曲到长曲,体现曲库时长差异),种子无关、完全确定。
负例: 从 test split 取一首,保证不在库内(9.2 的"查无此曲"查询用)。
注: 库内另有 9.4 合成翻唱的 8 首民歌原版渲染,Notebook 运行时经 cover_render 现场生成加入(共用音频批次)。
"""
from __future__ import annotations

import csv
from pathlib import Path

import soundfile as sf

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = CHAPTER_ROOT.parent
MUSDB = CODE_ROOT / "datasets" / "musdb18-hq"
OUT = CHAPTER_ROOT / "data_manifests" / "fingerprint_library.csv"

N_LIBRARY = 10

# MUSDB 官方逐曲清单：
# https://github.com/sigsep/website/blob/master/content/datasets/assets/tracklist.csv
# 官方数据页另说明整套数据需申请访问，曲目仅限学术用途使用。
LICENSE_BY_TRACK = {
    "A Classic Education - NightOwl": "CC BY-NC-SA 4.0",
    "Atlantis Bound - It Was My Fault For Waiting": "Restricted (academic use)",
    "Chris Durban - Celebrate": "Restricted (academic use)",
    "Giselle - Moss": "Restricted (academic use)",
    "Invisible Familiars - Disturbing Wildlife": "CC BY-NC-SA 4.0",
    "Jay Menon - Through My Eyes": "Restricted (academic use)",
    "Lushlife - Toynbee Suite": "CC BY-NC-SA 4.0",
    "Mu - Too Bright": "Restricted (academic use)",
    "Music Delta - Gospel": "CC BY-NC-SA 4.0",
    "Secret Mountains - High Horse": "CC BY-NC-SA 4.0",
    "Wall Of Death - Femme": "Restricted (academic use)",
}


def license_for(track: str) -> str:
    """返回官方逐曲清单中的许可状态；新选曲需先补充核对。"""
    return LICENSE_BY_TRACK.get(track, "Verify against official per-track list")


def track_durations(split: str) -> list[tuple[str, float]]:
    rows = []
    for track_dir in sorted((MUSDB / split).iterdir()):
        mixture = track_dir / "mixture.wav"
        if mixture.exists():
            rows.append((track_dir.name, sf.info(mixture).duration))
    return rows


def main() -> None:
    train = track_durations("train")
    # 同艺人只留一首(取该艺人时长中位者),避免库内曲目音色同源
    by_artist: dict[str, list[tuple[str, float]]] = {}
    for name, dur in train:
        by_artist.setdefault(name.split(" - ")[0], []).append((name, dur))
    unique = sorted((tracks[len(tracks) // 2] for tracks in by_artist.values()), key=lambda r: r[1])
    # 按时长排序等距取 10 首(含最短与最长)
    indices = [round(i * (len(unique) - 1) / (N_LIBRARY - 1)) for i in range(N_LIBRARY)]
    library = [unique[i] for i in indices]

    test = track_durations("test")
    negative = test[len(test) // 2]  # test 中间时长的一首,作库外负例

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track", "path", "split", "duration_sec", "role", "license_status", "notes"])
        for name, dur in library:
            writer.writerow([
                name,
                f"datasets/musdb18-hq/train/{name}/mixture.wav",
                "train",
                f"{dur:.3f}",
                "library",
                license_for(name),
                "指纹库曲目(完整曲)",
            ])
        writer.writerow([
            negative[0],
            f"datasets/musdb18-hq/test/{negative[0]}/mixture.wav",
            "test",
            f"{negative[1]:.3f}",
            "negative_query",
            license_for(negative[0]),
            "库外音频,查询应返回无匹配",
        ])

    print(f"库内 {len(library)} 首 + 负例 1 首 → {OUT}")
    for name, dur in library:
        print(f"  {dur:7.1f}s  {name}")
    print(f"  负例: {negative[0]} ({negative[1]:.1f}s)")
    print(f"  总库音频时长 {sum(d for _, d in library):.0f}s")


if __name__ == "__main__":
    main()
