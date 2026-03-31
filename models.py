from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentStats:
    word_count: int = 0
    chunk_count: int = 0
    file_count: int = 0
    page_count: int = 0
    processed_files: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "DocumentStats":
        payload = payload or {}
        return cls(
            word_count=int(payload.get("word_count", 0)),
            chunk_count=int(payload.get("chunk_count", 0)),
            file_count=int(payload.get("file_count", 0)),
            page_count=int(payload.get("page_count", 0)),
            processed_files=list(payload.get("processed_files", [])),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceReference:
    label: str
    snippet: str


@dataclass
class RagPipelineResult:
    answer: str
    sources: list[SourceReference]
    debug_messages: list[str] = field(default_factory=list)
