import json
import sqlite3
from pathlib import Path


class Registry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (event_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)")

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

    def all_runs(self):
        with self._connect() as db:
            rows = db.execute("SELECT record FROM runs ORDER BY run_id").fetchall()
        return [json.loads(row[0]) for row in rows]

    def append_event(self, kind: str, payload: dict):
        if not kind or not isinstance(payload, dict):
            raise ValueError("event requires kind and object payload")
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self._connect() as db:
            cursor = db.execute("INSERT INTO events (kind, payload) VALUES (?, ?)", (kind, encoded))
            return cursor.lastrowid

    def events(self, kind=None):
        with self._connect() as db:
            if kind is None:
                rows = db.execute("SELECT event_id, kind, payload FROM events ORDER BY event_id").fetchall()
            else:
                rows = db.execute("SELECT event_id, kind, payload FROM events WHERE kind = ? ORDER BY event_id", (kind,)).fetchall()
        return [{"event_id": row[0], "kind": row[1], "payload": json.loads(row[2])} for row in rows]

