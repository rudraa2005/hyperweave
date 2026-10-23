import hashlib
from pathlib import Path

import pandas as pd


DATASETS = {"nf_unsw_nb15_v2", "nf_cse_cic_ids2018_v2"}


def load_nf_csv(path, dataset: str, features: list[str], *, campaign_col: str, time_col: str):
    """Load a supplied NF-v2 CSV; fail closed if metadata or feature schema is absent."""
    if dataset not in DATASETS or len(features) != 43 or len(set(features)) != 43:
        raise ValueError("expected known NF-v2 dataset and 43 unique ordered feature names")
    df = pd.read_csv(path)
    required = set(features) | {"Label", campaign_col, time_col}
    if not required <= set(df.columns):
        raise ValueError(f"missing required columns: {required - set(df.columns)}")
    if df[features].isna().any().any():
        raise ValueError("missing feature values")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in features):
        raise ValueError("non-numeric feature; encoding must be specified before experiment")
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

