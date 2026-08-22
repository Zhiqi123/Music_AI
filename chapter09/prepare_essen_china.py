"""一次性数据制备:Essen 中国民歌子集 krn → MIDI 转换与选样。

来源: CODE/external/essen-folksong-collection/asia/china/(han/shanxi/natmin/xinhua,共 2241 个 .krn)
产物: CODE/datasets/essen_china_midi/{子库}/*.mid + manifest.csv(如有失败另出 conversion_failures.csv)
选样: natmin(206)与 xinhua(10)两个小子库全保留;han/shanxi 按原库比例(1223:802)随机抽 120/78,
      非 natmin 部分约 200 首,合计约 415 首。随机种子固定,结果可复现。
运行: CODE/venv_ch09_dataprep/bin/python CODE/chapter09/prepare_essen_china.py
说明: verovio 仅在本数据制备环节使用,第九章 Notebook 运行时不依赖。
"""

from __future__ import annotations

import base64
import csv
import random
import re
from pathlib import Path

import verovio

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC = CODE_ROOT / "external" / "essen-folksong-collection" / "asia" / "china"
DST = CODE_ROOT / "datasets" / "essen_china_midi"

SUBCORPORA = ("han", "shanxi", "natmin", "xinhua")
SAMPLE_QUOTA = {"han": 120, "shanxi": 78}  # 原库比例 1223:802 ≈ 60:40
KEEP_ALL = ("natmin", "xinhua")  # 小池全保留
SEED = 42

META_PATTERNS = {
    "title": re.compile(r"^!!!OTL:\s*(.*)$"),
    "area": re.compile(r"^!!!ARE:\s*(.*)$"),
    "ethnic_group": re.compile(r"^!!\s*Ethnic Group:\s*(.*)$"),
}
PITCH_RE = re.compile(r"[a-gA-G]")


def read_krn_text(path: Path) -> str:
    """读取 .krn 文本;老 Essen 文件混有 DOS CP437 编码(德语变音字符与不间断空格)。"""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp437").replace("\xa0", " ")


def parse_metadata(text: str) -> dict[str, str]:
    meta = {"title": "", "area": "", "ethnic_group": ""}
    for line in text.splitlines():
        for key, pattern in META_PATTERNS.items():
            m = pattern.match(line)
            if m:
                meta[key] = m.group(1).strip()
    return meta


def count_kern_notes(text: str) -> int:
    """数 **kern 数据行里的音符 token(小节线、解释项、小节大括号不计)。"""
    n = 0
    for line in text.splitlines():
        if not line or line[0] in "!*=":
            continue
        for token in line.replace("{", " ").replace("}", " ").split():
            if PITCH_RE.search(token):
                n += 1
    return n


def convert_all() -> tuple[list[dict], list[dict]]:
    """全部 2241 个文件先过一遍 verovio,成功项进入候选池,失败项记录原因。"""
    toolkit = verovio.toolkit()
    pool: list[dict] = []
    failures: list[dict] = []
    for sub in SUBCORPORA:
        for krn in sorted((SRC / sub).glob("*.krn")):
            text = read_krn_text(krn)
            record = {
                "subcorpus": sub,
                "krn_name": krn.name,
                "source_krn": str(krn.relative_to(CODE_ROOT)),
                "n_kern_notes": count_kern_notes(text),
                **parse_metadata(text),
            }
            try:
                if not toolkit.loadFile(str(krn)):
                    raise ValueError("verovio loadFile 返回 False")
                raw = base64.b64decode(toolkit.renderToMIDI())
                if not raw.startswith(b"MThd"):
                    raise ValueError("MIDI 头无效")
            except Exception as exc:  # 单文件失败不中断整批
                failures.append({**record, "error": str(exc)})
                continue
            record["midi_bytes"] = raw
            pool.append(record)
    return pool, failures


def select(pool: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    selected: list[dict] = []
    for sub in SUBCORPORA:
        candidates = [r for r in pool if r["subcorpus"] == sub]
        if sub in KEEP_ALL or len(candidates) <= SAMPLE_QUOTA.get(sub, 0):
            chosen = candidates
        else:
            chosen = rng.sample(candidates, SAMPLE_QUOTA[sub])
        selected.extend(sorted(chosen, key=lambda r: r["krn_name"]))
    return selected


def main() -> None:
    pool, failures = convert_all()
    selected = select(pool)

    DST.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in selected:
        sub_dir = DST / record["subcorpus"]
        sub_dir.mkdir(exist_ok=True)
        midi_name = Path(record["krn_name"]).with_suffix(".mid").name
        (sub_dir / midi_name).write_bytes(record["midi_bytes"])
        rows.append(
            {
                "midi_path": str((sub_dir / midi_name).relative_to(DST)),
                "subcorpus": record["subcorpus"],
                "source_krn": record["source_krn"],
                "title": record["title"],
                "area": record["area"],
                "ethnic_group": record["ethnic_group"],
                "n_kern_notes": record["n_kern_notes"],
            }
        )

    fieldnames = ["midi_path", "subcorpus", "source_krn", "title", "area", "ethnic_group", "n_kern_notes"]
    with (DST / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        with (DST / "conversion_failures.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subcorpus", "krn_name", "source_krn", "error"])
            writer.writeheader()
            writer.writerows({k: r[k] for k in ("subcorpus", "krn_name", "source_krn", "error")} for r in failures)

    print(f"候选池 {len(pool)} / 2241,转换失败 {len(failures)}")
    for sub in SUBCORPORA:
        n_pool = sum(1 for r in pool if r["subcorpus"] == sub)
        n_sel = sum(1 for r in selected if r["subcorpus"] == sub)
        print(f"  {sub:7s} 候选 {n_pool:4d} → 选样 {n_sel:4d}")
    print(f"合计选样 {len(selected)} 首,落盘 {DST}")


if __name__ == "__main__":
    main()
