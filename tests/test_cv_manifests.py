import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from defect_classifier.cv_manifests import (
    CvManifestError,
    _validate_fingerprints,
    _validate_integrity,
    load_cv_membership,
    persist_cv_memberships,
)
from defect_classifier.preparation import RowMeta, _build_components, _membership_fingerprint
from defect_classifier.protocol import load_protocol


def _row(project: str, issue_id: str, day: int, text_hash: str | None = None) -> RowMeta:
    return RowMeta(
        stable_id=f"{project}:{issue_id}",
        source_project=project,
        issue_id=issue_id,
        creation_time=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day),
        severity_raw="normal",
        target_s6="normal",
        target_s3="MEDIUM",
        target_s2="LOWER_IMPACT",
        dupe_of_raw="",
        exact_text_hash=text_hash or f"hash-{issue_id}",
    )


@pytest.fixture
def manifest_fixture(tmp_path: Path):
    protocol = load_protocol()
    rows = [_row("P", "1", 0), _row("P", "2", 1), _row("P", "3", 2)]
    memberships = {}
    expected = {}
    for family in ("within_project", "pooled"):
        for fold in range(1, 4):
            memberships[(family, fold, "TRAIN")] = {"P:1", "P:2"}
            memberships[(family, fold, "VALIDATION")] = {"P:3"}
            expected[f"{family}_fold_{fold}_training"] = _membership_fingerprint({"P:1", "P:2"})
            expected[f"{family}_fold_{fold}_validation"] = _membership_fingerprint({"P:3"})
    root = tmp_path / "manifests"
    persist_cv_memberships(root, memberships, rows)
    fingerprints = tmp_path / "fingerprints.json"
    fingerprints.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol.fingerprint,
                "cv_membership_sha256": expected,
            }
        ),
        encoding="utf-8",
    )
    return protocol, rows, memberships, expected, root, fingerprints


def test_exact_membership_persistence_and_load(manifest_fixture) -> None:
    protocol, _, _, _, root, fingerprints = manifest_fixture
    loaded = load_cv_membership(
        root,
        fingerprints,
        protocol,
        family="pooled",
        fold=2,
        role="TRAIN",
    )
    assert {row.stable_id for row in loaded} == {"P:1", "P:2"}
    assert all(row.source_project == "P" for row in loaded)
    assert all(row.issue_id in {"1", "2"} for row in loaded)


def test_persisted_fingerprints_equal_generated(manifest_fixture) -> None:
    _, _, memberships, expected, _, _ = manifest_fixture
    validation = _validate_fingerprints(memberships, expected)
    assert len(validation) == 12
    assert all(row["status"] == "MATCH" for row in validation)


def test_fingerprint_mismatch_fails_closed(manifest_fixture) -> None:
    protocol, _, _, _, root, fingerprints = manifest_fixture
    path = root / "pooled" / "fold_1" / "train.parquet"
    table = pq.read_table(path).slice(0, 1)
    pq.write_table(table, path)
    with pytest.raises(CvManifestError, match="fingerprint mismatch"):
        load_cv_membership(root, fingerprints, protocol, family="pooled", fold=1, role="TRAIN")


def test_missing_manifest_fails_closed(manifest_fixture) -> None:
    protocol, _, _, _, root, fingerprints = manifest_fixture
    path = root / "pooled" / "fold_1" / "train.parquet"
    path.rename(path.with_suffix(".moved"))
    with pytest.raises(CvManifestError, match="missing"):
        load_cv_membership(root, fingerprints, protocol, family="pooled", fold=1, role="TRAIN")


def test_protocol_drift_fails_closed(manifest_fixture) -> None:
    protocol, _, _, _, root, fingerprints = manifest_fixture
    changed = replace(protocol, fingerprint="different")
    with pytest.raises(CvManifestError, match="protocol drift"):
        load_cv_membership(root, fingerprints, changed, family="pooled", fold=1, role="TRAIN")


def test_integrity_rejects_train_validation_overlap(manifest_fixture) -> None:
    _, rows, memberships, _, _, _ = manifest_fixture
    union_find, _, _, _ = _build_components(rows)
    memberships[("pooled", 1, "VALIDATION")].add("P:1")
    with pytest.raises(CvManifestError, match="TRAIN/VALIDATION overlap"):
        _validate_integrity(memberships, rows, {row.stable_id for row in rows}, set(), union_find)


def test_integrity_rejects_locked_membership(manifest_fixture) -> None:
    _, rows, memberships, _, _, _ = manifest_fixture
    union_find, _, _, _ = _build_components(rows)
    with pytest.raises(CvManifestError, match="final locked-test"):
        _validate_integrity(memberships, rows, {row.stable_id for row in rows}, {"P:3"}, union_find)


def test_integrity_rejects_chronological_reversal(manifest_fixture) -> None:
    _, rows, memberships, _, _, _ = manifest_fixture
    union_find, _, _, _ = _build_components(rows)
    for family in ("within_project", "pooled"):
        memberships[(family, 1, "TRAIN")] = {"P:3"}
        memberships[(family, 1, "VALIDATION")] = {"P:1"}
    with pytest.raises(CvManifestError, match="chronology violated"):
        _validate_integrity(memberships, rows, {row.stable_id for row in rows}, set(), union_find)


def test_integrity_rejects_duplicate_component_leakage(tmp_path: Path) -> None:
    rows = [_row("P", "1", 0, "same"), _row("P", "2", 1, "same")]
    union_find, _, _, _ = _build_components(rows)
    memberships = {
        (family, fold, role): ({"P:1"} if role == "TRAIN" else {"P:2"})
        for family in ("within_project", "pooled")
        for fold in range(1, 4)
        for role in ("TRAIN", "VALIDATION")
    }
    with pytest.raises(CvManifestError, match="duplicate-component leakage"):
        _validate_integrity(memberships, rows, {"P:1", "P:2"}, set(), union_find)


def test_repeat_materialization_is_deterministic(manifest_fixture, tmp_path: Path) -> None:
    _, rows, memberships, _, root, _ = manifest_fixture
    before = {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.parquet")
    }
    second = tmp_path / "second"
    persist_cv_memberships(second, memberships, rows)
    after = {
        path.relative_to(second): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.rglob("*.parquet")
    }
    assert before == after


def test_loader_does_not_reconstruct_or_require_locked_artifacts(
    manifest_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, _, _, _, root, fingerprints = manifest_fixture
    monkeypatch.setattr(
        "defect_classifier.preparation._temporal_folds",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not reconstruct")),
    )
    loaded = load_cv_membership(
        root,
        fingerprints,
        protocol,
        family="within_project",
        fold=3,
        role="VALIDATION",
    )
    assert [row.stable_id for row in loaded] == ["P:3"]
