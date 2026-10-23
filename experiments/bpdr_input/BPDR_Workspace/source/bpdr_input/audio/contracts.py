"""Versioned hand-off data structures, independent of future audio/model code."""

from dataclasses import asdict, dataclass
from enum import StrEnum
import re


class Status(StrEnum):
    ACCEPT_UNCHANGED = "ACCEPT_UNCHANGED"
    ACCEPT_PREPARED = "ACCEPT_PREPARED"
    RERECORD_REQUIRED = "RERECORD_REQUIRED"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


@dataclass(frozen=True)
class AcceptedAudio:
    path: str
    checksum: str
    sample_rate_hz: int
    sample_count: int
    channels: int = 1

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("Accepted audio requires an artifact path")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checksum):
            raise ValueError("Audio checksum must be a lowercase SHA-256 digest")
        if self.sample_rate_hz <= 0 or self.sample_count <= 0:
            raise ValueError("Sample rate and sample count must be positive")
        if self.channels != 1:
            raise ValueError("The initial hand-off contract requires mono audio")

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz


@dataclass(frozen=True)
class ProcessingResult:
    request_id: str
    source_id: str
    status: Status
    accepted_audio: AcceptedAudio | None = None
    reason_codes: tuple[str, ...] = ()
    user_message: str = ""
    policy_version: str = "draft-0.1"
    schema_version: str = "0.1"

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.source_id.strip():
            raise ValueError("Request and source identifiers are required")
        if not isinstance(self.status, Status):
            raise TypeError("Status must be one of the declared Status values")
        accepted = self.status in (Status.ACCEPT_UNCHANGED, Status.ACCEPT_PREPARED)
        if accepted != (self.accepted_audio is not None):
            raise ValueError("Only accepted statuses may contain accepted audio")
        if self.accepted_audio is not None and not isinstance(self.accepted_audio, AcceptedAudio):
            raise TypeError("Accepted audio must satisfy the AcceptedAudio contract")
        if not accepted and not self.reason_codes:
            raise ValueError("Non-accepted results require a reason code")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["status"] = self.status.value
        if self.accepted_audio is not None:
            result["accepted_audio"]["duration_seconds"] = self.accepted_audio.duration_seconds
        return result


def require_accepted_audio(result: ProcessingResult) -> AcceptedAudio:
    """Stop a consumer from falling back to rejected original input."""
    if result.accepted_audio is None:
        raise ValueError(f"Downstream processing blocked: {result.status.value}")
    return result.accepted_audio
