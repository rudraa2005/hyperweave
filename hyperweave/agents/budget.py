"""Transactional, persistent run-budget accounting."""
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class HumanDecision:
    approved: bool
    actor: str

    def __post_init__(self):
        if not isinstance(self.approved, bool) or not isinstance(self.actor, str) or not self.actor:
            raise TypeError("explicit human approval/rejection and actor required")


class BudgetLedger:
    def __init__(self, registry, *, limits=None):
        self.registry = registry
        with sqlite3.connect(registry.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS budgets (condition TEXT PRIMARY KEY, limit_runs INTEGER NOT NULL, used_runs INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS reservations (proposal_id TEXT PRIMARY KEY, condition TEXT NOT NULL, cost INTEGER NOT NULL, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS decisions (proposal_id TEXT PRIMARY KEY, approved INTEGER NOT NULL, actor TEXT NOT NULL)")
            for condition, limit in (limits or {}).items():
                if not isinstance(limit, int) or limit < 0:
                    raise ValueError("budget limit must be a nonnegative integer")
                db.execute("INSERT OR IGNORE INTO budgets VALUES (?, ?, 0)", (condition, limit))
                existing = db.execute("SELECT limit_runs FROM budgets WHERE condition=?", (condition,)).fetchone()[0]
                if existing != limit:
                    raise ValueError("predeclared budget cannot be changed")

    def remaining(self, condition):
        with sqlite3.connect(self.registry.path) as db:
            row = db.execute("SELECT limit_runs-used_runs FROM budgets WHERE condition=?", (condition,)).fetchone()
        if row is None:
            raise ValueError("unknown budget condition")
        return row[0]

    def reserve(self, condition, proposal_id, cost, *, approved, actor):
        if not isinstance(condition, str) or not isinstance(proposal_id, str) or not proposal_id:
            raise TypeError("condition and proposal ID must be text")
        if not isinstance(cost, int) or isinstance(cost, bool) or cost < 1:
            raise TypeError("cost must be positive integer runs")
        if not isinstance(approved, bool) or not isinstance(actor, str) or not actor:
            raise TypeError("explicit boolean decision and actor required")
        refused = False
        with sqlite3.connect(self.registry.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT limit_runs, used_runs FROM budgets WHERE condition=?", (condition,)).fetchone()
            if row is None:
                raise ValueError("unknown budget condition")
            if db.execute("SELECT 1 FROM decisions WHERE proposal_id=?", (proposal_id,)).fetchone():
                raise ValueError("proposal already decided")
            db.execute("INSERT INTO decisions VALUES (?, ?, ?)", (proposal_id, int(approved), actor))
            if not approved or row[0] - row[1] < cost:
                refused = True
            else:
                db.execute("UPDATE budgets SET used_runs=used_runs+? WHERE condition=?", (cost, condition))
                db.execute("INSERT INTO reservations VALUES (?, ?, ?, 'reserved')", (proposal_id, condition, cost))
        self.registry.append_event("human_decision", {"proposal_id": proposal_id, "approved": approved, "actor": actor})
        if refused:
            if approved:
                self.registry.append_event("budget_refusal", {"proposal_id": proposal_id, "condition": condition, "cost": cost})
            return False
        self.registry.append_event("budget_reservation", {"proposal_id": proposal_id, "condition": condition, "cost": cost})
        return True

    def start(self, proposal_id):
        with sqlite3.connect(self.registry.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM reservations WHERE proposal_id=?", (proposal_id,)).fetchone()
            if row is None or row[0] != "reserved":
                raise PermissionError("execution requires an approved reservation")
            db.execute("UPDATE reservations SET state='running' WHERE proposal_id=?", (proposal_id,))
        self.registry.append_event("action_started", {"proposal_id": proposal_id})

    def finish(self, proposal_id, *, success):
        with sqlite3.connect(self.registry.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM reservations WHERE proposal_id=?", (proposal_id,)).fetchone()
            if row is None or row[0] != "running":
                raise ValueError("proposal was not started or already finished")
            db.execute("UPDATE reservations SET state=? WHERE proposal_id=?",
                       ("completed" if success else "failed", proposal_id))
        self.registry.append_event("action_status", {"proposal_id": proposal_id,
                                                     "status": "completed" if success else "failed"})

    def release_unstarted(self, proposal_id):
        """Rollback a reservation only before execution starts."""
        with sqlite3.connect(self.registry.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT condition, cost, state FROM reservations WHERE proposal_id=?", (proposal_id,)).fetchone()
            if row is None or row[2] != "reserved":
                raise ValueError("only unstarted reservations can be released")
            db.execute("UPDATE reservations SET state='released' WHERE proposal_id=?", (proposal_id,))
            db.execute("UPDATE budgets SET used_runs=used_runs-? WHERE condition=?", (row[1], row[0]))
        self.registry.append_event("rollback", {"proposal_id": proposal_id, "released_runs": row[1]})
