"""CLAP zero-shot 的 prompt 模板与 CTMP 类名英文映射。"""
from __future__ import annotations

INSTRUMENT_EN: dict[str, tuple[str, str]] = {
    "二胡": ("erhu", "bowed string instrument"),
    "琵琶": ("pipa", "plucked string instrument"),
    "中阮": ("zhongruan", "plucked string instrument"),
    "笛子": ("dizi", "bamboo flute"),
    "唢呐": ("suona", "double-reed wind instrument"),
    "笙":   ("sheng", "free-reed mouth organ"),
}

PROMPT_TEMPLATES: dict[str, str] = {
    "naive_label":   "{name}",
    "recording_of":  "a recording of {name}",
    "descriptive":   "the sound of {name}, a Chinese {family}",
    "domain_scoped": "Chinese traditional music, solo {name} performance",
}


def render_prompts(template_name: str) -> list[str]:
    """按某模板渲染 6 个 CTMP 类的 text prompts，顺序与 CTMP_CLASSES 一致。"""
    from chapter06._common.ctmp_loader import CTMP_CLASSES
    template = PROMPT_TEMPLATES[template_name]
    out = []
    for cls in CTMP_CLASSES:
        name, family = INSTRUMENT_EN[cls]
        out.append(template.format(name=name, family=family))
    return out
