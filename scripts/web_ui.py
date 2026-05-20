#!/usr/bin/env python3
"""Optional local Gradio UI for the Irodori-TTS-MLX generation CLI."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VOICE_DESIGN_WEIGHTS_REPO = "t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign"
VOICE_DESIGN_WEIGHTS_REVISION = "bf877a3beb7d921dc6bfb2b6812d02be07f39f2a"
V3_WEIGHTS_REPO = "t0yohei/Irodori-TTS-MLX-500M-v3"
V3_WEIGHTS_REVISION = "078ffb11ffad92e6dde237a6abef730f4341b359"
DEFAULT_CODEC_ARTIFACT_REPO = "t0yohei/Irodori-TTS-MLX-DACVAE-Codec"
MLX_CODEC_RUNTIME_MODES = {"mlx"}

ARTIFACT_PRESETS = {
    "VoiceDesign hosted": {
        "weights_repo": VOICE_DESIGN_WEIGHTS_REPO,
        "weights_revision": VOICE_DESIGN_WEIGHTS_REVISION,
        "caption": "落ち着いた女性の声",
        "no_ref": True,
        "force_no_reference": True,
        "supports_caption": True,
    },
    "v3 hosted": {
        "weights_repo": V3_WEIGHTS_REPO,
        "weights_revision": V3_WEIGHTS_REVISION,
        "caption": "",
        "no_ref": True,
        "force_no_reference": False,
        "supports_caption": False,
    },
}


@dataclass(frozen=True)
class WebGenerationConfig:
    artifact_preset: str = "VoiceDesign hosted"
    weights_repo: str = ""
    weights_revision: str = ""
    weights_dir: str = ""
    weights: str = ""
    text: str = "こんにちは。今日は良い天気です。"
    caption: str = "落ち着いた女性の声"
    ref_wav: str | None = None
    no_ref: bool = True
    preset: str = "balanced"
    num_steps: int | None = None
    seed: int = 0
    seconds: float | None = None
    duration_scale: float = 1.0
    codec_runtime_mode: str = "mlx"
    codec_artifact_repo: str = DEFAULT_CODEC_ARTIFACT_REPO
    codec_artifact_revision: str = ""
    codec_artifact_dir: str = ""
    codec_path: str = ""
    codec_device: str = "cpu"
    cfg_scale_text: float | None = None
    cfg_scale_caption: float | None = None
    cfg_scale_speaker: float | None = None
    cfg_guidance_mode: str = ""
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    output_dir: str = ""


def _clean_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _optional_float(value: object) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    return float(text)


def _optional_int(value: object) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    return int(text)


def _float_or_default(value: object, default: float) -> float:
    return default if value is None else float(value)


def _int_or_default(value: object, default: int) -> int:
    return default if value is None else int(value)


def _with_artifact_preset(config: WebGenerationConfig) -> WebGenerationConfig:
    preset = ARTIFACT_PRESETS.get(config.artifact_preset)
    if preset is None:
        return config
    return WebGenerationConfig(
        **{
            **config.__dict__,
            "weights_repo": config.weights_repo or str(preset["weights_repo"]),
            "weights_revision": config.weights_revision or str(preset["weights_revision"] or ""),
            "caption": config.caption if preset["supports_caption"] and config.caption else str(preset["caption"]),
            "no_ref": True if preset["force_no_reference"] else bool(config.no_ref),
        }
    )


def _add_optional(argv: list[str], flag: str, value: object) -> None:
    text = _clean_text(value)
    if text:
        argv.extend([flag, text])


def build_generate_argv(config: WebGenerationConfig, *, output_wav: str, metadata_json: str) -> list[str]:
    config = _with_artifact_preset(config)
    argv: list[str] = []
    weight_sources = [
        ("--weights-repo", config.weights_repo),
        ("--weights-dir", config.weights_dir),
        ("--weights", config.weights),
    ]
    selected_weights = [(flag, value) for flag, value in weight_sources if _clean_text(value)]
    if len(selected_weights) != 1:
        raise ValueError("Select exactly one weights source: hosted repo, hosted-layout directory, or local .npz.")
    flag, value = selected_weights[0]
    argv.extend([flag, _clean_text(value)])
    if flag == "--weights-repo":
        _add_optional(argv, "--weights-revision", config.weights_revision)

    if not _clean_text(config.text):
        raise ValueError("Text is required.")
    argv.extend(["--output-wav", output_wav, "--text", _clean_text(config.text)])
    _add_optional(argv, "--caption", config.caption)
    if config.no_ref:
        argv.append("--no-ref")
    else:
        _add_optional(argv, "--ref-wav", config.ref_wav)
    _add_optional(argv, "--preset", config.preset)
    if config.num_steps is not None:
        argv.extend(["--num-steps", str(int(config.num_steps))])
    argv.extend(["--seed", str(int(config.seed))])
    if config.seconds is not None:
        argv.extend(["--seconds", str(float(config.seconds))])
    argv.extend(["--duration-scale", str(float(config.duration_scale))])
    argv.extend(["--codec-runtime-mode", config.codec_runtime_mode])
    local_codec_source = bool(_clean_text(config.codec_path) or _clean_text(config.codec_artifact_dir))
    needs_codec_artifact = config.codec_runtime_mode in MLX_CODEC_RUNTIME_MODES
    codec_artifact_repo = config.codec_artifact_repo if needs_codec_artifact and not local_codec_source else ""
    _add_optional(argv, "--codec-artifact-repo", codec_artifact_repo)
    if _clean_text(codec_artifact_repo):
        _add_optional(argv, "--codec-artifact-revision", config.codec_artifact_revision)
    _add_optional(argv, "--codec-artifact-dir", config.codec_artifact_dir)
    _add_optional(argv, "--codec-path", config.codec_path)
    _add_optional(argv, "--codec-device", config.codec_device)
    if config.cfg_scale_text is not None:
        argv.extend(["--cfg-scale-text", str(float(config.cfg_scale_text))])
    if config.cfg_scale_caption is not None:
        argv.extend(["--cfg-scale-caption", str(float(config.cfg_scale_caption))])
    if config.cfg_scale_speaker is not None:
        argv.extend(["--cfg-scale-speaker", str(float(config.cfg_scale_speaker))])
    _add_optional(argv, "--cfg-guidance-mode", config.cfg_guidance_mode)
    argv.extend(
        [
            "--cfg-min-t",
            str(float(config.cfg_min_t)),
            "--cfg-max-t",
            str(float(config.cfg_max_t)),
            "--metadata-json",
            metadata_json,
            "--json",
        ]
    )
    return argv


def run_generation(config: WebGenerationConfig) -> tuple[str | None, str, str]:
    output_root = Path(config.output_dir).expanduser() if _clean_text(config.output_dir) else Path(tempfile.mkdtemp(prefix="irodori-web-"))
    output_root.mkdir(parents=True, exist_ok=True)
    output_wav = str(output_root / "irodori-web-output.wav")
    metadata_path = output_root / f"irodori-web-metadata-{uuid.uuid4().hex}.json"
    try:
        argv = build_generate_argv(config, output_wav=output_wav, metadata_json=str(metadata_path))
    except Exception as exc:
        return None, "", f"error: {exc}"

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.generate_wav", *argv],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    log_text = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    if completed.returncode != 0:
        return None, "", f"generation failed with exit code {completed.returncode}\n{log_text}".strip()
    metadata_text = ""
    if metadata_path.exists():
        try:
            metadata_text = json.dumps(json.loads(metadata_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2, sort_keys=True)
        except json.JSONDecodeError:
            metadata_text = metadata_path.read_text(encoding="utf-8")
    return output_wav, metadata_text, log_text


def _import_gradio() -> Any:
    try:
        return importlib.import_module("gradio")
    except ImportError as exc:
        raise RuntimeError('Gradio is required for irodori-tts-web. Install with: pip install -e ".[runtime,web]"') from exc


def build_ui() -> Any:
    gr = _import_gradio()
    with gr.Blocks(title="Irodori-TTS-MLX") as demo:
        gr.Markdown("# Irodori-TTS-MLX")
        gr.Markdown("Optional local Web UI for the existing irodori-tts-generate CLI. Do not upload reference audio unless you have rights to use it.")
        with gr.Row():
            artifact_preset = gr.Dropdown(["VoiceDesign hosted", "v3 hosted", "Custom"], value="VoiceDesign hosted", label="Artifact preset")
            weights_repo = gr.Textbox(label="Weights repo", placeholder=VOICE_DESIGN_WEIGHTS_REPO)
            weights_revision = gr.Textbox(label="Weights revision")
        with gr.Row():
            weights_dir = gr.Textbox(label="Weights directory/archive")
            weights = gr.Textbox(label="Local weights .npz")
        text = gr.Textbox(label="Text", value="こんにちは。今日は良い天気です。", lines=3)
        caption = gr.Textbox(label="Caption / style text", value="落ち着いた女性の声")
        with gr.Row():
            ref_wav = gr.Audio(label="Reference audio", type="filepath")
            no_ref = gr.Checkbox(label="No reference", value=True)
        with gr.Row():
            preset = gr.Dropdown(["ultra-fast", "fast", "balanced", "quality"], value="balanced", label="Generation preset")
            num_steps = gr.Textbox(label="Override num steps")
            seed = gr.Number(label="Seed", value=0, precision=0)
            seconds = gr.Textbox(label="Seconds")
            duration_scale = gr.Number(label="Duration scale", value=1.0)
        with gr.Accordion("Codec", open=False):
            codec_runtime_mode = gr.Dropdown(["mlx"], value="mlx", label="Codec runtime mode")
            codec_artifact_repo = gr.Textbox(label="Codec artifact repo", value=DEFAULT_CODEC_ARTIFACT_REPO)
            codec_artifact_revision = gr.Textbox(label="Codec artifact revision")
            codec_artifact_dir = gr.Textbox(label="Codec artifact directory/archive")
            codec_path = gr.Textbox(label="Codec .npz path")
            codec_device = gr.Dropdown(["cpu", "mps", "cuda"], value="cpu", label="PyTorch codec device")
        with gr.Accordion("Sampling", open=False):
            cfg_scale_text = gr.Number(label="CFG scale text override")
            cfg_scale_caption = gr.Number(label="CFG scale caption override")
            cfg_scale_speaker = gr.Number(label="CFG scale speaker override")
            cfg_guidance_mode = gr.Dropdown(["", "independent", "joint", "alternating", "reduced"], value="", label="CFG guidance mode override")
            cfg_min_t = gr.Number(label="CFG min t", value=0.5)
            cfg_max_t = gr.Number(label="CFG max t", value=1.0)
        output_dir = gr.Textbox(label="Output directory", placeholder="Leave blank to use a temporary directory")
        run = gr.Button("Generate", variant="primary")
        audio = gr.Audio(label="Generated audio", type="filepath")
        metadata = gr.Code(label="Metadata", language="json")
        logs = gr.Textbox(label="Logs", lines=8)

        def _submit(*values: Any) -> tuple[str | None, str, str]:
            config = WebGenerationConfig(
                artifact_preset=str(values[0]),
                weights_repo=_clean_text(values[1]),
                weights_revision=_clean_text(values[2]),
                weights_dir=_clean_text(values[3]),
                weights=_clean_text(values[4]),
                text=_clean_text(values[5]),
                caption=_clean_text(values[6]),
                ref_wav=values[7],
                no_ref=bool(values[8]),
                preset=_clean_text(values[9]),
                num_steps=_optional_int(values[10]),
                seed=_int_or_default(values[11], 0),
                seconds=_optional_float(values[12]),
                duration_scale=_float_or_default(values[13], 1.0),
                codec_runtime_mode=str(values[14]),
                codec_artifact_repo=_clean_text(values[15]),
                codec_artifact_revision=_clean_text(values[16]),
                codec_artifact_dir=_clean_text(values[17]),
                codec_path=_clean_text(values[18]),
                codec_device=str(values[19]),
                cfg_scale_text=_optional_float(values[20]),
                cfg_scale_caption=_optional_float(values[21]),
                cfg_scale_speaker=_optional_float(values[22]),
                cfg_guidance_mode=_clean_text(values[23]),
                cfg_min_t=_float_or_default(values[24], 0.5),
                cfg_max_t=_float_or_default(values[25], 1.0),
                output_dir=_clean_text(values[26]),
            )
            return run_generation(config)

        inputs = [
            artifact_preset,
            weights_repo,
            weights_revision,
            weights_dir,
            weights,
            text,
            caption,
            ref_wav,
            no_ref,
            preset,
            num_steps,
            seed,
            seconds,
            duration_scale,
            codec_runtime_mode,
            codec_artifact_repo,
            codec_artifact_revision,
            codec_artifact_dir,
            codec_path,
            codec_device,
            cfg_scale_text,
            cfg_scale_caption,
            cfg_scale_speaker,
            cfg_guidance_mode,
            cfg_min_t,
            cfg_max_t,
            output_dir,
        ]
        run.click(_submit, inputs=inputs, outputs=[audio, metadata, logs])
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the optional local Irodori-TTS-MLX Web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface for the local Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the local Gradio server.")
    parser.add_argument("--share", action="store_true", help="Ask Gradio to create a public share link.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the UI in a browser after launch.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=bool(args.share), inbrowser=bool(args.inbrowser), pwa=True)
    return 0


def cli_main() -> int:
    try:
        return main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
