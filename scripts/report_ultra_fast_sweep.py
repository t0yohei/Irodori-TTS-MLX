#!/usr/bin/env python3
"""Build a consolidated low-step/CFG sweep report from persistent benchmark summaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UltraFastSweepReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class SweepCandidate:
    label: str
    num_steps: int
    cfg_guidance_mode: str
    cfg_scale_text: float
    request_wall_ms: float
    sample_rf_ms: float | None
    decode_dacvae_ms: float | None
    decode_dacvae_model_ms: float | None
    audio_write_ms: float | None
    output_duration_seconds: float | None
    process_wall_seconds: float | None
    throughput_rps: float | None
    quality_proxy: str
    summary_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ultra-fast low-step/CFG sweep report.")
    parser.add_argument("summaries", nargs="+", help="persistent-batch-summary.json files")
    parser.add_argument("--output", required=True, help="Markdown report path")
    parser.add_argument("--issue-url", default="https://github.com/t0yohei/Irodori-TTS-MLX/issues/217")
    parser.add_argument("--parent-url", default="https://github.com/t0yohei/Irodori-TTS-MLX/issues/220")
    parser.add_argument("--baseline-url", default="2026-05-14-apple-silicon-num-steps-v3-text.md")
    parser.add_argument("--persistent-baseline-url", default="2026-05-18-apple-silicon-persistent-batch-runtime-cleanup.md")
    return parser.parse_args()


def _median_metric(summary: dict[str, Any], name: str) -> float | None:
    metric = (summary.get("aggregates") or {}).get(name)
    if not metric:
        return None
    value = metric.get("median")
    return None if value is None else float(value)


def _quality_proxy(num_steps: int, cfg_scale_text: float, *, request_wall_ms: float) -> str:
    if num_steps < 6:
        return "experimental-high-risk"
    if cfg_scale_text == 0:
        return "experimental-unguided"
    if num_steps <= 6:
        return "experimental-fastest"
    if request_wall_ms < 1500:
        return "plausibly-usable"
    return "control-or-anchor"


def parse_summary(path: str | Path) -> SweepCandidate:
    source = Path(path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    invocation = summary.get("invocation") or {}
    process = summary.get("process") or {}
    request_wall_ms = _median_metric(summary, "measured_total_to_decode_ms")
    if request_wall_ms is None:
        raise UltraFastSweepReportError(f"{source}: missing aggregates.measured_total_to_decode_ms.median")
    num_steps = int(invocation["num_steps"])
    cfg_scale_text = float(invocation.get("cfg_scale_text", 3.0))
    mode = str(invocation.get("cfg_guidance_mode") or "independent")
    return SweepCandidate(
        label=str(invocation.get("case_label") or source.parent.name or source.stem),
        num_steps=num_steps,
        cfg_guidance_mode=mode,
        cfg_scale_text=cfg_scale_text,
        request_wall_ms=float(request_wall_ms),
        sample_rf_ms=_median_metric(summary, "measured_sample_rf_ms"),
        decode_dacvae_ms=_median_metric(summary, "measured_decode_dacvae_ms"),
        decode_dacvae_model_ms=_median_metric(summary, "measured_decode_dacvae_model_ms"),
        audio_write_ms=_median_metric(summary, "measured_audio_write_ms"),
        output_duration_seconds=_median_metric(summary, "measured_output_duration_seconds"),
        process_wall_seconds=None if process.get("wall_seconds") is None else float(process["wall_seconds"]),
        throughput_rps=None if process.get("measured_generation_throughput_rps") is None else float(process["measured_generation_throughput_rps"]),
        quality_proxy=_quality_proxy(num_steps, cfg_scale_text, request_wall_ms=float(request_wall_ms)),
        summary_path=str(source),
    )


def _ms(value: float | None) -> str:
    return "" if value is None else f"{value:.1f} ms"


def _seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.2f} s"


def build_report(candidates: list[SweepCandidate], *, args: argparse.Namespace) -> str:
    ranked = sorted(candidates, key=lambda item: item.request_wall_ms)
    fastest = ranked[0]
    preset_note = (
        "No candidate is suitable for a public ultra-fast preset from latency alone; keep the setting experimental "
        "until subjective MOS/parity checks accept the quality loss."
        if fastest.quality_proxy.startswith("experimental")
        else "The fastest candidate is plausibly usable, but still needs subjective quality review before a public preset."
    )
    lines = [
        "# Apple Silicon ultra-fast low-step/CFG sweep",
        "",
        f"Issue: [#217]({args.issue_url})",
        f"Parent: [#220]({args.parent_url})",
        "",
        "## Summary",
        "",
        f"- Fastest measured request latency: {fastest.label} at {_ms(fastest.request_wall_ms)}.",
        f"- Fastest settings: num_steps={fastest.num_steps}, cfg_guidance_mode={fastest.cfg_guidance_mode}, cfg_scale_text={fastest.cfg_scale_text:g}.",
        f"- Public preset recommendation: {preset_note}",
        f"- Baselines: [#64 v3 one-shot]({args.baseline_url}) and [persistent mlx-decode baseline]({args.persistent_baseline_url}).",
        "",
        "## Ranked Candidates",
        "",
        "| Rank | Case | Steps | CFG mode | CFG text | Request wall | sample_rf | DACVAE decode | Decode model | Audio write | Output duration | Quality proxy |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, item in enumerate(ranked, start=1):
        lines.append(
            "| {rank} | {label} | {steps} | {mode} | {cfg:g} | {wall} | {sample} | {decode} | {decode_model} | {write} | {duration} | {quality} |".format(
                rank=rank,
                label=item.label,
                steps=item.num_steps,
                mode=item.cfg_guidance_mode,
                cfg=item.cfg_scale_text,
                wall=_ms(item.request_wall_ms),
                sample=_ms(item.sample_rf_ms),
                decode=_ms(item.decode_dacvae_ms),
                decode_model=_ms(item.decode_dacvae_model_ms),
                write=_ms(item.audio_write_ms),
                duration=_seconds(item.output_duration_seconds),
                quality=item.quality_proxy,
            )
        )
    lines.extend([
        "",
        "## Evidence Fields",
        "",
        "Each source summary must include invocation.num_steps, invocation.cfg_guidance_mode, invocation.cfg_scale_text, process.wall_seconds, aggregates.measured_total_to_decode_ms, aggregates.measured_sample_rf_ms, aggregates.measured_decode_dacvae_ms, and, for current reports, aggregates.measured_audio_write_ms plus aggregates.measured_output_duration_seconds.",
        "",
        "## Source Summaries",
        "",
    ])
    for item in ranked:
        lines.append(f"- {item.label}: {item.summary_path}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    candidates = [parse_summary(path) for path in args.summaries]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(candidates, args=args), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UltraFastSweepReportError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc

