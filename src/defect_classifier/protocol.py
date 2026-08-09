"""Validated loading of the frozen research protocol."""

from __future__ import annotations

import hashlib
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProtocolError(ValueError):
    """Raised when the frozen protocol is missing or internally inconsistent."""


@dataclass(frozen=True)
class TargetDefinition:
    mapping: dict[str, str]
    order: tuple[str, ...]


@dataclass(frozen=True)
class FrozenProtocol:
    path: Path
    fingerprint: str
    protocol_id: str
    version: int
    seed: int
    expected_raw_rows: int
    expected_eligible_rows: int
    raw_columns: tuple[str, ...]
    raw_delimiter: str
    raw_encoding: str
    accepted_labels: tuple[str, ...]
    excluded_labels: tuple[str, ...]
    targets: dict[str, TargetDefinition]
    text_fields: tuple[str, str]
    text_separator: str
    development_ratio: float
    cv_fold_count: int
    duplicate_field: str
    unlock_environment_variable: str
    unlock_value: str
    document: dict[str, Any]

    def map_severity(self, raw_label: str) -> dict[str, str]:
        """Map an accepted raw label for every task, failing closed otherwise."""
        if raw_label not in self.accepted_labels:
            raise ProtocolError(f"severity is not eligible under protocol v1: {raw_label!r}")
        return {task: definition.mapping[raw_label] for task, definition in self.targets.items()}

    def canonicalize_text(self, value: str | None) -> str:
        text = "" if value is None else value
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return unicodedata.normalize("NFC", text)

    def combine_text(self, summary: str | None, description: str | None) -> tuple[str, str, str]:
        canonical_summary = self.canonicalize_text(summary)
        canonical_description = self.canonicalize_text(description)
        return (
            canonical_summary,
            canonical_description,
            canonical_summary + self.text_separator + canonical_description,
        )

    def exact_text_hash(self, summary: str | None, description: str | None) -> str:
        canonical_summary = self.canonicalize_text(summary)
        canonical_description = self.canonicalize_text(description)
        digest = hashlib.sha256()
        digest.update(canonical_summary.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(canonical_description.encode("utf-8"))
        return digest.hexdigest()


def default_protocol_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "protocol_v1.toml"


def _target_definition(
    task: str, table: dict[str, Any], accepted: tuple[str, ...]
) -> TargetDefinition:
    raw_order = table.get("order")
    if (
        not isinstance(raw_order, list)
        or not raw_order
        or not all(isinstance(label, str) for label in raw_order)
    ):
        raise ProtocolError(f"targets.{task}.order must be a non-empty string array")
    mapping = {key: value for key, value in table.items() if key != "order"}
    if set(mapping) != set(accepted) or not all(
        isinstance(value, str) for value in mapping.values()
    ):
        raise ProtocolError(f"targets.{task} mapping must cover accepted labels exactly")
    if set(mapping.values()) != set(raw_order):
        raise ProtocolError(f"targets.{task}.order must contain every mapped class exactly")
    return TargetDefinition(mapping=mapping, order=tuple(raw_order))


def load_protocol(path: str | Path | None = None) -> FrozenProtocol:
    protocol_path = Path(path) if path is not None else default_protocol_path()
    try:
        raw_bytes = protocol_path.read_bytes()
        document = tomllib.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ProtocolError(f"cannot load protocol {protocol_path}: {exc}") from exc

    accepted = tuple(document.get("targets", {}).get("accepted_raw_labels", ()))
    excluded = tuple(document.get("targets", {}).get("excluded_raw_labels", ()))
    if len(accepted) != 6 or len(set(accepted)) != len(accepted):
        raise ProtocolError("accepted raw severity labels must contain six unique labels")
    if set(accepted) & set(excluded):
        raise ProtocolError("accepted and excluded severity labels overlap")
    targets = {
        task: _target_definition(task, document["targets"][task], accepted)
        for task in ("s6", "s3", "s2")
    }
    raw_schema = document.get("raw_schema", {})
    raw_columns = tuple(raw_schema.get("columns", ()))
    if len(raw_columns) != 42 or len(set(raw_columns)) != len(raw_columns):
        raise ProtocolError("raw_schema.columns must freeze 42 unique ordered columns")
    text = document.get("text", {})
    text_fields = tuple(text.get("fields", ()))
    if text_fields != ("Summary", "Description"):
        raise ProtocolError("protocol v1 text fields must be Summary and Description")
    ratio = document.get("split", {}).get("development_ratio")
    if not isinstance(ratio, float) or not 0 < ratio < 1:
        raise ProtocolError("split.development_ratio must be between zero and one")
    fold_count = document.get("temporal_cv", {}).get("fold_count")
    if fold_count != 3:
        raise ProtocolError("protocol v1 requires exactly three temporal CV folds")
    access = document.get("access_control", {})
    assertions = document.get("dataset_assertions", {})
    return FrozenProtocol(
        path=protocol_path.resolve(),
        fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
        protocol_id=str(document.get("protocol_id")),
        version=int(document.get("version", 0)),
        seed=int(document.get("seed", 0)),
        expected_raw_rows=int(assertions.get("expected_raw_rows", 0)),
        expected_eligible_rows=int(
            assertions.get("expected_eligible_rows_before_duplicate_overlap", 0)
        ),
        raw_columns=raw_columns,
        raw_delimiter=str(raw_schema.get("delimiter")),
        raw_encoding=str(raw_schema.get("encoding")),
        accepted_labels=accepted,
        excluded_labels=excluded,
        targets=targets,
        text_fields=(text_fields[0], text_fields[1]),
        text_separator=str(text.get("separator")),
        development_ratio=ratio,
        cv_fold_count=fold_count,
        duplicate_field=str(document.get("duplicates", {}).get("explicit_equivalence_field")),
        unlock_environment_variable=str(access.get("unlock_environment_variable")),
        unlock_value=str(access.get("unlock_value")),
        document=document,
    )
