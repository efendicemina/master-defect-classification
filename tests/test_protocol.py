import hashlib
from pathlib import Path

import pytest

from defect_classifier.protocol import ProtocolError, load_protocol


def test_frozen_target_mappings_and_orders() -> None:
    protocol = load_protocol()
    assert protocol.targets["s6"].order == (
        "blocker",
        "critical",
        "major",
        "normal",
        "minor",
        "trivial",
    )
    assert protocol.targets["s3"].order == ("HIGH", "MEDIUM", "LOW")
    assert protocol.targets["s2"].order == ("HIGH_IMPACT", "LOWER_IMPACT")
    assert protocol.map_severity("blocker") == {
        "s6": "blocker",
        "s3": "HIGH",
        "s2": "HIGH_IMPACT",
    }
    assert protocol.map_severity("normal") == {
        "s6": "normal",
        "s3": "MEDIUM",
        "s2": "LOWER_IMPACT",
    }
    assert protocol.map_severity("minor") == {
        "s6": "minor",
        "s3": "LOW",
        "s2": "LOWER_IMPACT",
    }


def test_unknown_and_excluded_severity_fail_closed() -> None:
    protocol = load_protocol()
    with pytest.raises(ProtocolError):
        protocol.map_severity("future-label")
    with pytest.raises(ProtocolError):
        protocol.map_severity("enhancement")


def test_canonical_text_is_minimal_and_deterministic() -> None:
    protocol = load_protocol()
    summary, description, combined = protocol.combine_text("Cafe\u0301\r\n", None)
    assert summary == "Café\n"
    assert description == ""
    assert combined == "Café\n\n\n"
    expected = hashlib.sha256("Café\n\x00".encode()).hexdigest()
    assert protocol.exact_text_hash("Cafe\u0301\r\n", None) == expected


def test_protocol_fingerprint_is_file_sha256() -> None:
    protocol = load_protocol()
    assert protocol.fingerprint == hashlib.sha256(Path(protocol.path).read_bytes()).hexdigest()
