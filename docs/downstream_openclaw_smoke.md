# Downstream OpenClaw smoke path

Issue: [#146](https://github.com/t0yohei/Irodori-TTS-MLX/issues/146)
Parent: [#123 TOY-5 v0.2 cross-repo delivery](https://github.com/t0yohei/Irodori-TTS-MLX/issues/123)

This page defines the downstream local-assistant/OpenClaw smoke path for an
Irodori-TTS-MLX v0.2 build. It is intentionally a smoke contract, not a full
quality benchmark: the pass condition is that the selected OpenClaw entry point
can be fed a reproducible Irodori-TTS-MLX WAV artifact with machine-readable
metadata and clear fallback behavior.

## Downstream consumer and entry point

Use this smoke path from the downstream OpenClaw/local-assistant repository or
checkout that will consume Irodori-TTS-MLX. Keep the downstream repository URL,
private checkout path, and host-specific service manager details in that
downstream project; this public repository records only the portable contract.

The downstream entry point should be one of:

- a local-assistant TTS playback path that can consume a local WAV file; or
- an OpenAI-compatible TTS endpoint that accepts `POST /v1/audio/speech` and
  returns `audio/wav`.

For TOY-5, the required Irodori-TTS-MLX smoke artifact is produced by this
repository's irodori-tts-generate CLI, then handed to the OpenClaw/local-
assistant playback boundary as a local WAV file. This keeps the smoke path
independent from private OpenClaw configuration while still validating the
downstream artifact shape that local-assistant needs: a playable WAV plus
metadata that identifies the checkpoint family, duration mode, codec backend,
and generation timing fields.

## Required inputs

Set these paths for the smoke run:

    export IRODORI_MLX_REPO=/path/to/Irodori-TTS-MLX
    export IRODORI_UPSTREAM_REPO=/path/to/Irodori-TTS
    export IRODORI_SMOKE_DIR=/tmp/irodori-openclaw-smoke
    export IRODORI_WEIGHTS=/models/irodori-tts-mlx/weights.npz
    export IRODORI_MODEL_CONFIG=/models/irodori-tts-mlx/model_config.json
    export OPENCLAW_CONSUMER_REPO=/path/to/openclaw-local-assistant-consumer

The recommended first downstream smoke uses v3 no-reference generation because
it avoids committing or distributing a reference speaker WAV:

    export IRODORI_TEXT='OpenClaw local assistant smoke test.'
    export IRODORI_CODEC_MODE=persistent
    export IRODORI_PRESET=fast

Use a hosted layout instead of direct .npz weights only when the target hosted
repository has irodori_mlx_manifest.json with license_review.status: "approved":

    export IRODORI_WEIGHTS_REPO=t0yohei/Irodori-TTS-MLX-500M-v3

## Irodori-TTS-MLX smoke command

Create the output directory outside the repository so generated audio, cached
checkpoints, and codec artifacts cannot be committed accidentally:

    mkdir -p "$IRODORI_SMOKE_DIR"
    cd "$IRODORI_MLX_REPO"

    PYTHONPATH="$IRODORI_UPSTREAM_REPO:${PYTHONPATH:-}" \
    irodori-tts-generate \
      --weights "$IRODORI_WEIGHTS" \
      --model-config-json "$IRODORI_MODEL_CONFIG" \
      --text "$IRODORI_TEXT" \
      --no-reference \
      --codec-runtime-mode "$IRODORI_CODEC_MODE" \
      --preset "$IRODORI_PRESET" \
      --output "$IRODORI_SMOKE_DIR/openclaw-smoke.wav" \
      --metadata-json "$IRODORI_SMOKE_DIR/openclaw-smoke.metadata.json" \
      --json > "$IRODORI_SMOKE_DIR/openclaw-smoke.stdout.json"

Hosted-layout variant:

    PYTHONPATH="$IRODORI_UPSTREAM_REPO:${PYTHONPATH:-}" \
    irodori-tts-generate \
      --weights-repo "$IRODORI_WEIGHTS_REPO" \
      --text "$IRODORI_TEXT" \
      --no-reference \
      --codec-runtime-mode "$IRODORI_CODEC_MODE" \
      --preset "$IRODORI_PRESET" \
      --output "$IRODORI_SMOKE_DIR/openclaw-smoke.wav" \
      --metadata-json "$IRODORI_SMOKE_DIR/openclaw-smoke.metadata.json" \
      --json > "$IRODORI_SMOKE_DIR/openclaw-smoke.stdout.json"

For VoiceDesign, replace the v3 no-reference inputs with an approved
VoiceDesign hosted/local layout and add a caption:

    --caption 'calm, close, natural assistant voice' \
    --no-reference

## Expected metadata

The smoke run passes the Irodori-TTS-MLX side only when all of these checks
hold:

    test -s "$IRODORI_SMOKE_DIR/openclaw-smoke.wav"
    python - <<'PY'
    import json
    import os
    import wave
    from pathlib import Path

    root = Path(os.environ["IRODORI_SMOKE_DIR"])
    metadata_payload = json.loads((root / "openclaw-smoke.metadata.json").read_text())
    stdout_payload = json.loads((root / "openclaw-smoke.stdout.json").read_text())
    metadata = metadata_payload["result"]
    stdout_result = stdout_payload["result"]

    required = [
        "checkpoint_family",
        "checkpoint_capabilities",
        "duration_mode",
        "codec_backend",
        "codec_encode_backend",
        "codec_decode_backend",
        "resolved_seconds",
        "timings_ms",
    ]
    missing = [name for name in required if name not in metadata]
    if missing:
        raise SystemExit(f"missing metadata fields: {missing}")
    if metadata["checkpoint_family"] not in {"v3", "voicedesign", "base_v2"}:
        raise SystemExit(f"unexpected checkpoint_family: {metadata['checkpoint_family']!r}")
    if metadata["duration_mode"] not in {"predicted", "fallback", "manual"}:
        raise SystemExit(f"unexpected duration_mode: {metadata['duration_mode']!r}")
    if stdout_result.get("output_wav") != str(root / "openclaw-smoke.wav"):
        raise SystemExit("stdout output_wav does not match smoke WAV path")

    with wave.open(str(root / "openclaw-smoke.wav"), "rb") as wav:
        if wav.getnframes() <= 0:
            raise SystemExit("empty WAV")
        if wav.getframerate() <= 0:
            raise SystemExit("invalid sample rate")

    print(json.dumps({
        "ok": True,
        "checkpoint_family": metadata["checkpoint_family"],
        "duration_mode": metadata["duration_mode"],
        "codec_decode_backend": metadata["codec_decode_backend"],
        "wav": str(root / "openclaw-smoke.wav"),
    }, ensure_ascii=False))
    PY

## OpenClaw playback boundary check

The OpenClaw-side smoke should consume the generated WAV through the same local
assistant playback boundary used for local TTS replies. If the downstream
consumer exposes an OpenAI-compatible TTS adapter, validate it with a generic
request like this:

    cd "$OPENCLAW_CONSUMER_REPO"

    export OPENCLAW_TTS_BASE_URL=http://127.0.0.1:5058/v1
    export OPENCLAW_TTS_MODEL=irodori-tts-mlx-smoke

    python - <<'PY'
    import json
    import os
    import urllib.request
    from pathlib import Path

    root = Path(os.environ["IRODORI_SMOKE_DIR"])
    payload = {
        "model": os.environ["OPENCLAW_TTS_MODEL"],
        "input": os.environ["IRODORI_TEXT"],
        "response_format": "wav",
    }
    req = urllib.request.Request(
        os.environ["OPENCLAW_TTS_BASE_URL"].rstrip("/") + "/audio/speech",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        audio = resp.read()
        content_type = resp.headers.get_content_type()
    if content_type != "audio/wav":
        raise SystemExit(f"unexpected content type: {content_type}")
    if not audio:
        raise SystemExit("empty downstream audio response")
    out = root / "openclaw-playback.wav"
    out.write_bytes(audio)
    print(json.dumps({"ok": True, "path": str(out), "contentType": content_type}))
    PY

If the downstream integration consumes local files instead of an HTTP endpoint,
feed `$IRODORI_SMOKE_DIR/openclaw-smoke.wav` to that playback boundary and
record the command in the downstream repository. The local-assistant side passes
when it reports a successful playback or no-play validation with a non-empty WAV
path and `audio/wav` content.

## Fallback behavior

- Hosted weights unavailable, unapproved, or outside the audited families: use
  direct local conversion and --weights "$IRODORI_WEIGHTS" with
  --model-config-json "$IRODORI_MODEL_CONFIG".
- Missing upstream irodori_tts or PyTorch DACVAE runtime: keep
  --codec-runtime-mode persistent or subprocess, fix PYTHONPATH or install
  upstream Irodori-TTS in the same environment, and expect the command to fail
  before writing a WAV with an actionable import/runtime error.
- Missing MLX codec artifact for --codec-runtime-mode mlx, mlx-decode, or
  mlx-decode-subprocess: pass --codec-path for a local DACVAE codec .npz, or
  fall back to persistent.
- Missing MLX runtime dependencies: install this package with the documented
  runtime extras before retrying.
- Missing local speaker or playback tools: keep --no-play and validate the WAV
  and JSON metadata only.

## Artifact policy

Do not commit generated audio, downloaded checkpoint caches, converted
weights.npz, DACVAE codec artifacts, Hugging Face cache snapshots, or secrets.
Commit only compact Markdown/JSON smoke summaries that omit private cache paths
when a PR or issue needs evidence.

The smoke report should name the exact Irodori-TTS-MLX commit or PR head, the
downstream OpenClaw checkout or PR head, and whether the run used direct local
weights or an approved hosted layout.
