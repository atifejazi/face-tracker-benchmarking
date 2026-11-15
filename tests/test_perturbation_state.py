from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from perturb_avatar import state_from_export


class PerturbationExportTests(unittest.TestCase):
    def write_mesh(self, path, shape, expr_width=100):
        np.savez(path, shape=shape, static_offset=np.zeros((1, 4, 3), np.float32),
                 expr=np.zeros((1, expr_width), np.float32), rotation=np.zeros((1, 3)),
                 neck_pose=np.zeros((1, 3)), jaw_pose=np.zeros((1, 3)),
                 eyes_pose=np.zeros((1, 6)), translation=np.zeros((1, 3)))

    def test_export_reads_pre_ga_state_across_splits(self):
        with TemporaryDirectory() as directory:
            a, b = (Path(directory) / name for name in ("0.npz", "1.npz"))
            for path in (a, b):
                self.write_mesh(path, np.zeros(300, np.float32))
            splits = {"train": [({"timestep_index": 0}, None, a, "0", 0)],
                      "test": [({"timestep_index": 1}, None, b, "0", 1)]}
            flame = SimpleNamespace(v_template=np.zeros((4, 3)))
            result = state_from_export(flame, splits)
            self.assertEqual(result["expr"].shape, (2, 100))
            self.assertEqual(result["dynamic_offset"].shape, (2, 4, 3))
            self.assertEqual(result["static_offset"].shape, (4, 3))

            self.write_mesh(b, np.ones(300, np.float32))
            with self.assertRaisesRegex(ValueError, "identity varies"):
                state_from_export(flame, splits)

    def test_export_rejects_incompatible_expression_basis(self):
        with TemporaryDirectory() as directory:
            a = Path(directory) / "0.npz"
            self.write_mesh(a, np.zeros(300, np.float32), expr_width=50)
            splits = {"train": [({"timestep_index": 0}, None, a, "0", 0)]}
            flame = SimpleNamespace(v_template=np.zeros((4, 3)))
            with self.assertRaisesRegex(ValueError, "100 expression coefficients"):
                state_from_export(flame, splits)


if __name__ == "__main__":
    unittest.main()
