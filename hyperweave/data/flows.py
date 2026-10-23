import hashlib
import csv
import ipaddress
from pathlib import Path

import pandas as pd


DATASETS = {"nf_unsw_nb15_v2", "nf_cse_cic_ids2018_v2"}
IPV4_FEATURES = frozenset({"IPV4_SRC_ADDR", "IPV4_DST_ADDR"})


class MissingSplitMetadataError(ValueError):
    pass


def csv_header(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        return next(csv.reader(source))


def validate_nf_header(path, dataset: str, features: list[str]):
    if dataset not in DATASETS or len(features) != 43 or len(set(features)) != 43:
        raise ValueError("expected known NF-v2 dataset and 43 unique ordered feature names")
    header = csv_header(path)
    if len(header) != len(set(header)):
        raise ValueError("duplicate CSV column names")
    missing = (set(features) | {"Label"}) - set(header)
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return header


def encode_nf_features(frame: pd.DataFrame, features: list[str]):
    """Deterministic IPv4-to-uint32 conversion; other features stay numeric."""
    out = frame.copy()
    for name in features:
        if name in IPV4_FEATURES:
            try:
                out[name] = out[name].map(lambda value: int(ipaddress.IPv4Address(str(value)))).astype("uint32")
            except (ipaddress.AddressValueError, ValueError, TypeError) as exc:
                raise ValueError(f"invalid IPv4 value in {name}") from exc
        elif not pd.api.types.is_numeric_dtype(out[name]):
            out[name] = pd.to_numeric(out[name], errors="raise")
        if out[name].isna().any():
            raise ValueError(f"missing feature values in {name}")
    return out


def iter_nf_csv_chunks(path, dataset: str, features: list[str], *, chunksize=100000):
    header = validate_nf_header(path, dataset, features)
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("chunksize must be positive")
    columns = features + ["Label"] + (["Attack"] if "Attack" in header else [])
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        chunk = encode_nf_features(chunk, features)
        labels = pd.to_numeric(chunk["Label"], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError("Label must be binary 0/1")
        yield chunk.loc[:, columns]


def load_nf_csv(path, dataset: str, features: list[str], *, campaign_col: str, time_col: str):
    """Load a supplied NF-v2 CSV; fail closed if metadata or feature schema is absent."""
    header = validate_nf_header(path, dataset, features)
    required = set(features) | {"Label", campaign_col, time_col}
    if not {campaign_col, time_col} <= set(header):
        raise MissingSplitMetadataError(
            f"campaign/time columns missing: {sorted({campaign_col, time_col} - set(header))}; "
            "do not use a random-flow or row-order split")
    if not required <= set(header):
        raise ValueError(f"missing required columns: {required - set(header)}")
    df = pd.read_csv(path, usecols=list(required))
    df = encode_nf_features(df, features)
    labels = pd.to_numeric(df["Label"], errors="raise")
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("Label must contain both binary classes 0 and 1")
    if df[[campaign_col, time_col]].isna().any().any():
        raise ValueError("campaign/time metadata cannot be missing")
    return df.loc[:, features + ["Label", campaign_col, time_col]].copy()


def deduplicate(df: pd.DataFrame, features: list[str]):
    """Keep first occurrence per flow feature vector, before any split."""
    conflict = df.groupby(features, dropna=False)["Label"].nunique()
    if (conflict > 1).any():
        raise ValueError("identical flow features have conflicting labels")
    return df.drop_duplicates(subset=features, keep="first").reset_index(drop=True)


def split_by_campaign(df: pd.DataFrame, campaign_col: str, time_col: str):
    """Disjoint chronological campaign groups: A-train, A-val, B, sealed."""
    groups = df.groupby(campaign_col, sort=False)[time_col].min().sort_values().index.tolist()
    if len(groups) < 4:
        raise ValueError("at least four distinct campaign groups required")
    # Equal-sized group partitions; no flow-level randomization.
    n = len(groups)
    first = max(1, (n - 1) // 2)
    cuts = [first, max(first + 1, n - 2), n - 1]
    if not cuts[0] < cuts[1] < cuts[2]:
        raise ValueError("insufficient campaigns for four splits")
    name_groups = {"a_train": groups[:cuts[0]], "a_val": groups[cuts[0]:cuts[1]],
                   "b": groups[cuts[1]:cuts[2]], "sealed": groups[cuts[2]:]}
    return {name: df.loc[df[campaign_col].isin(ids)].copy() for name, ids in name_groups.items()}


def validate_disjoint(splits: dict, features: list[str], campaign_col: str):
    campaigns, rows = set(), set()
    for part in splits.values():
        current = set(part[campaign_col])
        if campaigns & current:
            raise ValueError("campaign overlap")
        campaigns.update(current)
        hashes = set(pd.util.hash_pandas_object(part[features], index=False).tolist())
        if len(hashes) != len(part) or rows & hashes:
            raise ValueError("duplicate flow within or across splits")
        rows.update(hashes)


def validate_chronological(splits: dict, time_col: str):
    order = ("a_train", "a_val", "b", "sealed")
    for earlier, later in zip(order, order[1:]):
        if splits[earlier][time_col].max() >= splits[later][time_col].min():
            raise ValueError("time windows overlap across splits")


def dataset_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

