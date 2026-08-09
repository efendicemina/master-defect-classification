from pathlib import Path, PurePosixPath

from defect_classifier.catalogue import DatasetCatalogue
from defect_classifier.verification import verify_dataset


def test_verification_reports_present_and_missing_files(tmp_path: Path) -> None:
    present = tmp_path / "one" / "present.csv"
    present.parent.mkdir()
    present.write_bytes(b"fake")
    catalogue = DatasetCatalogue(
        projects={
            "PRESENT": PurePosixPath("one/present.csv"),
            "MISSING": PurePosixPath("two/missing.csv"),
        },
        version=1,
    )

    checks = verify_dataset(tmp_path, catalogue)

    assert checks[0].exists is True
    assert checks[0].size_bytes == 4
    assert checks[1].exists is False
    assert checks[1].size_bytes is None
