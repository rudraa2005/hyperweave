import ast
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from hyperweave.models import AffineMLP, FunctionalNet
from hyperweave.eval import score_binary
from hyperweave.registry import Registry
from hyperweave.seed import seed_everything

pytestmark = pytest.mark.global_gate


def test_determinism():
    x = np.random.default_rng(7).normal(size=(32, 3))
    y = (x[:, 0] > 0).astype(int)
    for cls, args in ((AffineMLP, (3, 5)), (FunctionalNet, (3, 3, 2))):
        outputs = []
        for _ in range(2):
            seed_everything(11)
            model = cls(*args, seed=11).fit(x, y, epochs=3, seed=11)
            outputs.append(model.logits(x))
        np.testing.assert_array_equal(*outputs)
        assert score_binary(y, outputs[0], 0) == score_binary(y, outputs[1], 0)


def test_registry_immutable(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite")
    record = dict(run_id="r1", config={}, seed=11, dataset_hash="sha256:x", git_commit="none",
                  metrics={}, checkpoint_path="checkpoint.npz")
    registry.write(record)
    with pytest.raises(sqlite3.IntegrityError):
        registry.write(record)
    assert registry.read("r1") == record


def test_no_test_leak():
    root = Path(__file__).resolve().parents[2]
    for file in (root / "hyperweave" / "agents").rglob("*.py"):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert "sealed" not in ast.unparse(node).lower()
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "sealed/" not in node.value.lower()

