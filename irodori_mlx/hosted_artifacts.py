from __future__ import annotations

from dataclasses import dataclass

from .config import CHECKPOINT_FAMILY_V3, CHECKPOINT_FAMILY_VOICEDESIGN_V2


@dataclass(frozen=True)
class HostedRfDitArtifact:
    """Publication status for a hosted RF-DiT weights artifact."""

    family: str
    upstream_checkpoint: str
    repo_id: str | None
    revision: str | None
    publication_status: str
    license_review_status: str
    review_reference: str
    issue_url: str
    parent_issue_url: str
    blocker: str | None = None

    @property
    def is_approved_public(self) -> bool:
        return (
            self.publication_status == "approved-public"
            and self.license_review_status == "approved"
            and self.repo_id is not None
            and self.revision is not None
        )


HOSTED_RF_DIT_ARTIFACTS: dict[str, HostedRfDitArtifact] = {
    CHECKPOINT_FAMILY_VOICEDESIGN_V2: HostedRfDitArtifact(
        family=CHECKPOINT_FAMILY_VOICEDESIGN_V2,
        upstream_checkpoint="Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        repo_id="t0yohei/Irodori-TTS-MLX-500M-v2-VoiceDesign",
        revision="bf877a3beb7d921dc6bfb2b6812d02be07f39f2a",
        publication_status="approved-public",
        license_review_status="approved",
        review_reference=(
            "https://github.com/t0yohei/Irodori-TTS-MLX/blob/main/"
            "docs/preconverted_weights_redistribution_audit.md"
        ),
        issue_url="https://github.com/t0yohei/Irodori-TTS-MLX/issues/157",
        parent_issue_url="https://github.com/t0yohei/Irodori-TTS-MLX/issues/160",
    ),
    CHECKPOINT_FAMILY_V3: HostedRfDitArtifact(
        family=CHECKPOINT_FAMILY_V3,
        upstream_checkpoint="Aratako/Irodori-TTS-500M-v3",
        repo_id="t0yohei/Irodori-TTS-MLX-500M-v3",
        revision="078ffb11ffad92e6dde237a6abef730f4341b359",
        publication_status="approved-public",
        license_review_status="approved",
        review_reference=(
            "https://github.com/t0yohei/Irodori-TTS-MLX/blob/main/"
            "docs/preconverted_weights_redistribution_audit.md"
        ),
        issue_url="https://github.com/t0yohei/Irodori-TTS-MLX/issues/187",
        parent_issue_url="https://github.com/t0yohei/Irodori-TTS-MLX/issues/160",
    ),
}


def hosted_rf_dit_artifacts() -> dict[str, HostedRfDitArtifact]:
    """Return the v0.2 hosted RF-DiT artifact publication contract."""

    return dict(HOSTED_RF_DIT_ARTIFACTS)


def approved_hosted_rf_dit_artifacts() -> dict[str, HostedRfDitArtifact]:
    """Return only artifacts that are approved for public --weights-repo use."""

    return {
        family: artifact
        for family, artifact in HOSTED_RF_DIT_ARTIFACTS.items()
        if artifact.is_approved_public
    }


def approved_hosted_rf_dit_repo(family: str) -> str:
    """Return the approved public repo id for a family, or explain the blocker."""

    artifact = HOSTED_RF_DIT_ARTIFACTS.get(family)
    if artifact is None:
        known = ", ".join(sorted(HOSTED_RF_DIT_ARTIFACTS))
        raise KeyError(f"unknown hosted RF-DiT family {family!r}; known families: {known}")
    if artifact.is_approved_public:
        assert artifact.repo_id is not None
        return artifact.repo_id
    blocker = artifact.blocker or "artifact is not approved for public hosted use"
    raise RuntimeError(blocker)
