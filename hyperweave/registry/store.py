import json
import sqlite3
from pathlib import Path


class Registry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, record TEXT NOT NULL)")

    def _connect(self):
        return sqlite3.connect(self.path)

    def write(self, record: dict):
        required = {"run_id", "config", "seed", "dataset_hash", "git_commit", "metrics", "checkpoint_path"}
        if not required <= record.keys():
            raise ValueError(f"missing registry fields: {required - record.keys()}")
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        with self._connect() as db:
            db.execute("INSERT INTO runs (run_id, record) VALUES (?, ?)", (record["run_id"], payload))

    def read(self, run_id):
        with self._connect() as db:
            row = db.execute("SELECT record FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

