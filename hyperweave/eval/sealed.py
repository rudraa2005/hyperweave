"""Trusted evaluator: raw effects stay in a private store, reply is coarse."""
import json
import sqlite3
from pathlib import Path


CHECKS = frozenset({"general", "shift", "evasion", "latency", "regression"})


class SealedValidator:
    def __init__(self, private_path, evaluator, *, query_limit=10):
        if not isinstance(query_limit, int) or query_limit < 1:
            raise ValueError("query limit must be positive")
        self.private_path = Path(private_path)
        self.private_path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self.query_limit = query_limit
        with sqlite3.connect(self.private_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY, candidate TEXT NOT NULL, check_kind TEXT NOT NULL, raw_effect REAL, reply TEXT NOT NULL)")

    def count(self):
        with sqlite3.connect(self.private_path) as db:
            return db.execute("SELECT COUNT(*) FROM queries").fetchone()[0]

    def query(self, candidate_run_id, *, check="general"):
        if not isinstance(candidate_run_id, str) or not candidate_run_id:
            raise TypeError("candidate run ID must be text")
        if check not in CHECKS:
            raise ValueError("unknown validator check")
        with sqlite3.connect(self.private_path) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM queries").fetchone()[0] >= self.query_limit:
                raise RuntimeError("validator query budget exhausted")
            evaluator = self.evaluator.get(check) if isinstance(self.evaluator, dict) else self.evaluator
            effect = None if evaluator is None else evaluator(candidate_run_id)
            if effect is not None and (not isinstance(effect, (int, float)) or not -1 <= effect <= 1):
                raise ValueError("validator effect must be a detection-rate delta in [-1,1]")
            if effect is None:
                reply = {"verdict": "inconclusive", "bucket": "unknown"}
            elif effect > .02:
                reply = {"verdict": "accept", "bucket": "better"}
            elif effect < -.02:
                reply = {"verdict": "reject", "bucket": "worse"}
            else:
                reply = {"verdict": "inconclusive", "bucket": "similar"}
            db.execute("INSERT INTO queries(candidate, check_kind, raw_effect, reply) VALUES (?, ?, ?, ?)",
                       (candidate_run_id, check, effect, json.dumps(reply, sort_keys=True)))
        return reply
