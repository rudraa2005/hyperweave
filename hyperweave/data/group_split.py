"""Streaming, deterministic endpoint-pair-disjoint NF-v2 sampling."""
import argparse
import json
from pathlib import Path

import numpy as np

from .flows import dataset_hash, iter_nf_csv_chunks


SALT = np.uint64(20261024)
RANK_SALT = np.uint64(0xA0761D6478BD642F)
LIMIT = 6000


def _mix64(x):
    x = np.asarray(x, dtype=np.uint64).copy()
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def endpoint_group_hash(src, dst):
    a, b = np.asarray(src, np.uint64), np.asarray(dst, np.uint64)
    if a.shape != b.shape:
        raise ValueError("endpoint arrays must match")
    return _mix64(np.minimum(a, b) * np.uint64(0x9E3779B185EBCA87) ^
                  np.maximum(a, b) * np.uint64(0xC2B2AE3D27D4EB4F) ^ SALT)


def partition_for_group(group_hash):
    bucket = np.asarray(group_hash, np.uint64) % np.uint64(100)
    return np.where(bucket < 70, 0, np.where(bucket < 90, 1, np.where(bucket < 95, 2, 3)))


def _retain(pool, x, group, rank, limit):
    if len(x) == 0:
        return pool
    if pool is not None:
        x = np.concatenate((pool[0], x))
        group = np.concatenate((pool[1], group))
        rank = np.concatenate((pool[2], rank))
    if len(rank) > limit:
        ids = np.argpartition(rank, limit - 1)[:limit]
    else:
        ids = np.arange(len(rank))
    return x[ids], group[ids], rank[ids]


def sample_group_disjoint(unsw_csv, cse_csv, features, output, *, limit=LIMIT, chunksize=100000,
                          min_val_benign=1000):
    if len(features) != 43 or len(set(features)) != 43 or limit < 1:
        raise ValueError("need ordered 43-feature manifest and positive limit")
    if not {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"} <= set(features):
        raise ValueError("endpoint features required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh split directory")
    output.mkdir(parents=True, exist_ok=True)
    pools = {(split, label): None for split in range(4) for label in (0, 1)}
    paths = ((unsw_csv, "nf_unsw_nb15_v2", (0, 1)),
             (cse_csv, "nf_cse_cic_ids2018_v2", (2, 3)))
    for path, dataset, accepted in paths:
        offset = 0
        src_idx, dst_idx = features.index("IPV4_SRC_ADDR"), features.index("IPV4_DST_ADDR")
        for chunk in iter_nf_csv_chunks(path, dataset, features, chunksize=chunksize):
            x = chunk[features].to_numpy(dtype=np.float64)
            y = chunk["Label"].to_numpy(dtype=np.uint8)
            group = endpoint_group_hash(x[:, src_idx].astype(np.uint64), x[:, dst_idx].astype(np.uint64))
            split = partition_for_group(group)
            rank = _mix64(np.arange(offset, offset + len(x), dtype=np.uint64) ^ RANK_SALT)
            offset += len(x)
            for part in accepted:
                for label in (0, 1):
                    ids = np.flatnonzero((split == part) & (y == label))
                    if len(ids) > limit:
                        ids = ids[np.argpartition(rank[ids], limit - 1)[:limit]]
                    pools[part, label] = _retain(pools[part, label], x[ids], group[ids], rank[ids], limit)
    names = ("a_train", "a_val", "b", "sealed")
    manifest = {"kind": "endpoint_pair_group_disjoint_v1", "salt": int(SALT),
                "sample_limit_per_class_partition": limit,
                "source_sha256": {"nf_unsw_nb15_v2": dataset_hash(unsw_csv),
                                  "nf_cse_cic_ids2018_v2": dataset_hash(cse_csv)},
                "features": list(features), "partitions": {}}
    group_sets = []
    for part, name in enumerate(names):
        if any(pools[part, label] is None for label in (0, 1)):
            raise ValueError(f"{name} lacks a class under preregistered group split")
        x = np.vstack([pools[part, label][0] for label in (0, 1)])
        y = np.r_[np.zeros(len(pools[part, 0][0]), np.uint8),
                  np.ones(len(pools[part, 1][0]), np.uint8)]
        group = np.concatenate([pools[part, label][1] for label in (0, 1)])
        # np.unique keeps the first of identical feature vectors after ranking.
        unique, first, inverse = np.unique(x, axis=0, return_index=True, return_inverse=True)
        low = np.ones(len(unique), dtype=np.uint8)
        high = np.zeros(len(unique), dtype=np.uint8)
        np.minimum.at(low, inverse, y)
        np.maximum.at(high, inverse, y)
        if np.any(low != high):
            raise ValueError(f"{name} has conflicting labels for identical features")
        first.sort()
        x, y, group = x[first], y[first], group[first]
        if not np.any(y == 0) or not np.any(y == 1):
            raise ValueError(f"{name} loses a class after deduplication")
        group_sets.append(set(map(int, group)))
        destination = output / ("private" if name == "sealed" else "public")
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination / f"{name}.npz", x=x, y=y, group=group)
        manifest["partitions"][name] = {"rows": int(len(y)), "benign": int(np.sum(y == 0)),
                                         "attack": int(np.sum(y == 1)), "groups": len(group_sets[-1]),
                                         "dataset": "nf_unsw_nb15_v2" if part < 2 else "nf_cse_cic_ids2018_v2"}
    for i in range(4):
        for j in range(i + 1, 4):
            if group_sets[i] & group_sets[j]:
                raise ValueError("endpoint pair group overlap")
    if manifest["partitions"]["a_val"]["benign"] < min_val_benign:
        raise ValueError("A-val cannot resolve 0.1% FPR")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsw", required=True)
    parser.add_argument("--cse", required=True)
    parser.add_argument("--features", default="configs/nf_v2_features.txt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = sample_group_disjoint(args.unsw, args.cse, Path(args.features).read_text().splitlines(), args.output)
    print(json.dumps(report["partitions"], indent=2))


if __name__ == "__main__":
    main()
