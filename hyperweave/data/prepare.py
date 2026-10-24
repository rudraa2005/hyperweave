"""Import official local dataset archives without treating them as research results."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

from .flows import validate_nf_header


NF_FILES = {
    "unsw": ("NF-UNSW-NB15-v2.csv", "nf_unsw_nb15_v2"),
    "cse": ("NF-CSE-CIC-IDS2018-v2.csv", "nf_cse_cic_ids2018_v2"),
}
UNSW_SHA1 = "dde9a9e38009ec60121cbb4fd884bf2c7cf5c0b0"
FASHION_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def _hash_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _source_stream(source, expected_name):
    source = Path(source)
    if not zipfile.is_zipfile(source):
        if source.name != expected_name:
            raise ValueError(f"expected {expected_name}, got {source.name}")
        with source.open("rb") as stream:
            yield stream
        return
    with zipfile.ZipFile(source) as archive:
        members = [name for name in archive.namelist() if Path(name).name == expected_name]
        if len(members) != 1:
            raise ValueError(f"archive must contain exactly one {expected_name}")
        with archive.open(members[0]) as stream:
            yield stream


def install_nf(source, output, key, features, *, expected_sha1=None):
    if key not in NF_FILES:
        raise ValueError("unknown NF-v2 dataset")
    filename, dataset = NF_FILES[key]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / filename
    if target.exists():
        raise FileExistsError(f"dataset already installed: {target}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix=f".{filename}.",
                                         suffix=".part", delete=False) as staged:
            temporary = Path(staged.name)
            sha256, sha1 = hashlib.sha256(), hashlib.sha1()
            with _source_stream(source, filename) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    staged.write(block)
                    sha256.update(block)
                    sha1.update(block)
        validate_nf_header(temporary, dataset, features)
        if expected_sha1 is not None and sha1.hexdigest() != expected_sha1:
            raise ValueError(f"{filename} SHA-1 differs from the audited official file")
        os.replace(temporary, target)
        return {"path": str(target), "size_bytes": target.stat().st_size,
                "sha256": sha256.hexdigest(), "sha1": sha1.hexdigest(),
                "source": str(Path(source)), "status": "imported_header_verified_not_audited"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_fashion(directory):
    directory = Path(directory)
    results = {}
    for filename, expected in FASHION_MD5.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Fashion-MNIST file is missing: {path}")
        actual = _hash_file(path, "md5")
        if actual != expected:
            raise ValueError(f"Fashion-MNIST MD5 mismatch: {filename}")
        results[filename] = {"path": str(path), "md5": actual,
                             "size_bytes": path.stat().st_size}
    return results


def install_fashion(source_dir, output):
    verified = verify_fashion(source_dir)
    target_dir = Path(output) / "fashion"
    target_dir.mkdir(parents=True, exist_ok=True)
    installed = {}
    for filename, detail in verified.items():
        source = Path(detail["path"])
        target = target_dir / filename
        if source.resolve() != target.resolve():
            if target.exists():
                raise FileExistsError(f"Fashion-MNIST file already installed: {target}")
            with tempfile.NamedTemporaryFile(dir=target_dir, prefix=f".{filename}.",
                                             suffix=".part", delete=False) as staged:
                temporary = Path(staged.name)
            try:
                shutil.copyfile(source, temporary)
                if _hash_file(temporary, "md5") != detail["md5"]:
                    raise ValueError(f"Fashion-MNIST copy failed checksum verification: {filename}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        if _hash_file(target, "md5") != detail["md5"]:
            raise ValueError(f"Fashion-MNIST copy failed checksum verification: {filename}")
        installed[filename] = {**detail, "path": str(target)}
    return installed


def readiness(raw_dir):
    raw_dir = Path(raw_dir)
    runs_dir = raw_dir.parent.parent / "runs"
    return {"nf_v2": {key: (raw_dir / name).is_file()
                      for key, (name, _) in NF_FILES.items()},
            "fashion_mnist": {name: (raw_dir / "fashion" / name).is_file()
                              for name in FASHION_MD5},
            "split_manifest": any(runs_dir.glob("group_split_*/manifest.json")),
            "phase4_report": any(runs_dir.glob("phase4_group_*/report.json"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsw", help="local official UNSW CSV or ZIP")
    parser.add_argument("--cse", help="local official CSE CSV or ZIP")
    parser.add_argument("--fashion-dir", help="directory containing four official Fashion-MNIST gzip files")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--features", default="configs/nf_v2_features.txt")
    args = parser.parse_args()
    output = Path(args.output)
    if not any((args.unsw, args.cse, args.fashion_dir)):
        print(json.dumps(readiness(output), indent=2))
        return
    features = Path(args.features).read_text(encoding="utf-8").splitlines()
    result = {"kind": "local_dataset_import", "performance_evaluation": "not_run"}
    if args.unsw:
        result["unsw"] = install_nf(args.unsw, output, "unsw", features,
                                    expected_sha1=UNSW_SHA1)
    if args.cse:
        result["cse"] = install_nf(args.cse, output, "cse", features)
    if args.fashion_dir:
        result["fashion_mnist"] = install_fashion(args.fashion_dir, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "import_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True),
                                                  encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
