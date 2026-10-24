import hashlib
import zipfile

import pytest

from hyperweave.data import prepare

pytestmark = pytest.mark.phase1


def _csv_bytes(features):
    return (",".join(features + ["Label", "Attack"]) + "\n" +
            ",".join(["1"] * len(features) + ["0", "Benign"]) + "\n").encode()


def test_import_official_named_csv_from_nested_zip_and_refuse_overwrite(tmp_path):
    features = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"] + [f"f{i}" for i in range(41)]
    content = _csv_bytes(features)
    archive = tmp_path / "download.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("release/data/NF-UNSW-NB15-v2.csv", content)
    raw = tmp_path / "raw"
    result = prepare.install_nf(archive, raw, "unsw", features,
                                expected_sha1=hashlib.sha1(content).hexdigest())
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert (raw / "NF-UNSW-NB15-v2.csv").read_bytes() == content
    assert prepare.readiness(raw)["nf_v2"] == {"unsw": True, "cse": False}
    with pytest.raises(FileExistsError):
        prepare.install_nf(archive, raw, "unsw", features)


def test_import_rejects_wrong_hash_without_leaving_dataset_or_part(tmp_path):
    features = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"] + [f"f{i}" for i in range(41)]
    source = tmp_path / "NF-UNSW-NB15-v2.csv"
    source.write_bytes(_csv_bytes(features))
    raw = tmp_path / "raw"
    with pytest.raises(ValueError, match="SHA-1"):
        prepare.install_nf(source, raw, "unsw", features, expected_sha1="wrong")
    assert list(raw.iterdir()) == []


def test_fashion_import_verifies_all_four_files(tmp_path, monkeypatch):
    source = tmp_path / "downloaded"
    source.mkdir()
    checksums = {}
    for name in prepare.FASHION_MD5:
        payload = name.encode()
        (source / name).write_bytes(payload)
        checksums[name] = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(prepare, "FASHION_MD5", checksums)
    installed = prepare.install_fashion(source, tmp_path / "raw")
    assert len(installed) == 4
    assert all((tmp_path / "raw" / "fashion" / name).is_file() for name in checksums)
    (source / next(iter(checksums))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        prepare.verify_fashion(source)
