import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_tracking import FIELDS, summarize, validate_metrics, write_csv


class TrackingAuditTests(unittest.TestCase):
    def row(self, subject, frame, value):
        return dict(dataset="MultiREX", subject=subject, sequence="ROM", camera="front",
                    frame=str(frame), tracker="mica", protocol="pilot", motion_split="unknown",
                    view_split="unknown", capture="monocular", region="mouth", metric="error",
                    unit="mm", value=value)

    def test_summary_weights_identities_equally_and_preserves_unknown(self):
        with TemporaryDirectory() as directory:
            metrics, output = (Path(directory) / name for name in ("metrics.csv", "grouped.csv"))
            rows = [self.row("001", frame, 1) for frame in range(4)] + [self.row("002", 0, 5)]
            write_csv(metrics, FIELDS, rows)
            summarize(SimpleNamespace(metrics=metrics, output=output, group_by=["glasses"],
                                      clip_labels=None, frame_labels=None))
            with output.open(newline="") as file:
                result, = csv.DictReader(file)
            self.assertEqual(result["glasses"], "unknown")
            self.assertEqual(result["n_identities"], "2")
            self.assertEqual(result["n_observations"], "5")
            self.assertEqual(float(result["mean"]), 3)

    def test_duplicate_frame_metric_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate metric row"):
            validate_metrics([self.row("001", 0, 1), self.row("001", 0, 2)])


if __name__ == "__main__":
    unittest.main()
