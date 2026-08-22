"""Conditioning and trainability tables for Chapter 8 pretrained runners."""
from __future__ import annotations

from pathlib import Path

from _common.tables import write_rows


CONDITIONING_FIELDS = [
    "model_id",
    "notebook",
    "current_demo_conditions",
    "where_the_conditions_come_from",
    "generation_mechanism",
    "audio_conditioning",
    "image_conditioning",
    "trainability_in_chapter",
    "practical_training_path",
]


CONDITIONING_ROWS = [
    {
        "model_id": "musicgen",
        "notebook": "08_6b_musicgen_inference.ipynb",
        "current_demo_conditions": "text prompt; author-audio continuation prompt when the configured audio file exists",
        "where_the_conditions_come_from": "configs/musicgen_inference.yaml and CODE/datasets/audio_author/chapter_08_author",
        "generation_mechanism": "T5 text embedding conditions a codec-token language model; audio continuation prefixes the generated token stream.",
        "audio_conditioning": "audio continuation demonstrated through MusicGen.generate_continuation when the author audio prompt exists.",
        "image_conditioning": "not a native MusicGen condition in this notebook.",
        "trainability_in_chapter": "fine-tuning reference path",
        "practical_training_path": "08_8 exports AudioCraft JSONL and Dora command from audio/text pairs.",
    },
    {
        "model_id": "audioldm2",
        "notebook": "08_6c_audioldm2_inference.ipynb",
        "current_demo_conditions": "text prompt; negative prompt; duration; diffusion steps",
        "where_the_conditions_come_from": "configs/audioldm2_inference.yaml",
        "generation_mechanism": "text encoders and language model condition latent diffusion; a vocoder decodes the latent audio.",
        "audio_conditioning": "not used by this runner; this notebook keeps the stable text-to-audio path.",
        "image_conditioning": "not native to AudioLDM2 text-to-audio inference.",
        "trainability_in_chapter": "not trained locally",
        "practical_training_path": "training from scratch is large-scale; use official/Diffusers fine-tuning recipes outside the teaching run.",
    },
    {
        "model_id": "stable_audio_open",
        "notebook": "08_6d_stable_audio_open_inference.ipynb",
        "current_demo_conditions": "text prompt; duration; seed; diffusion steps; CFG scale; sigma range; sampler type",
        "where_the_conditions_come_from": "configs/stable_audio_open_inference.yaml",
        "generation_mechanism": "stable-audio-tools builds prompt, seconds_start, and seconds_total conditioning, then samples a latent diffusion model and decodes stereo audio.",
        "audio_conditioning": "not used by this runner; this notebook keeps the official text-to-audio path.",
        "image_conditioning": "not native to Stable Audio Open text-to-audio inference.",
        "trainability_in_chapter": "not trained locally",
        "practical_training_path": "large-scale training or fine-tuning requires the provider/official training stack and license review.",
    },
    {
        "model_id": "yue",
        "notebook": "08_6e_yue_full_song_generation.ipynb",
        "current_demo_conditions": "genre text file; lyrics text file; segment count; output directory",
        "where_the_conditions_come_from": "data_manifests/yue_genre.example.txt and data_manifests/yue_prompt.example.txt",
        "generation_mechanism": "official YuE infer.py consumes structured song text and writes full-song audio segments.",
        "audio_conditioning": "not used by the command template.",
        "image_conditioning": "not part of the YuE runner.",
        "trainability_in_chapter": "not trained locally",
        "practical_training_path": "use the official YuE training or inference repository on a large CUDA setup.",
    },
    {
        "model_id": "ace_step",
        "notebook": "08_6f_ace_step_generation.ipynb",
        "current_demo_conditions": "style prompt; lyrics; duration; guidance settings; seed",
        "where_the_conditions_come_from": "configs/ace_step_inference.yaml",
        "generation_mechanism": "ACE-Step pipeline maps prompt and lyric controls into a diffusion-style music generation process.",
        "audio_conditioning": "author audio is used for LoRA personalization in 08_8, not direct inference conditioning here.",
        "image_conditioning": "not part of the ACE-Step runner.",
        "trainability_in_chapter": "LoRA personalization plan",
        "practical_training_path": "08_8 validates author-audio manifest and writes an ACE-Step LoRA training plan.",
    },
]


def build_conditioning_rows(model_id: str | None = None) -> list[dict[str, object]]:
    """Return rows explaining how pretrained demos are conditioned."""
    rows = [dict(row) for row in CONDITIONING_ROWS]
    if model_id is not None:
        rows = [row for row in rows if row["model_id"] == model_id]
    return rows


def write_conditioning_table(output_csv: Path | str) -> list[dict[str, object]]:
    """Write and return the pretrained conditioning matrix."""
    rows = build_conditioning_rows()
    write_rows(output_csv, rows, fieldnames=CONDITIONING_FIELDS)
    return rows
