"""Public command failure semantics and safe feature-file loading, without CUDA."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from opendlss_cuda import cli


class CLITests(unittest.TestCase):
    def test_help_and_import_without_cuda(self):
        code = "import ctypes; ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('CUDA load')); from opendlss_cuda.cli import main; main(['--help'])"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run-features", result.stdout)

    def test_feature_header_and_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npy"
            shape = (3, 4, 16)
            g = {"full_height": 3, "full_width": 4}
            data = np.zeros(shape, np.float32)
            data[0, 0, 0] = -0.0
            np.save(path, data)
            mapped = cli.load_features(path, g)
            self.assertEqual(mapped.tobytes(), data.tobytes())
            del mapped
            for bad in (np.zeros(shape, np.float16), np.zeros((1, 4, 16), np.float32),
                        np.full(shape, np.nan, np.float32), np.full(shape, 65505, np.float32),
                        np.asfortranarray(data)):
                np.save(path, bad)
                with self.assertRaises(ValueError):
                    cli.load_features(path, g)
            np.save(path, data)
            with path.open("ab") as stream:
                stream.write(b"extra")
            with self.assertRaisesRegex(ValueError, "trailing"):
                cli.load_features(path, g)
            np.save(path, data)
            with path.open("r+b") as stream:
                stream.truncate(path.stat().st_size - 1)
            with self.assertRaisesRegex(ValueError, "truncated"):
                cli.load_features(path, g)

    def test_parity_status_exit_codes_and_fresh_destination(self):
        for status, code in (("passed", 0), ("mismatch", 1), ("unavailable", 3)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary) / "report"
                argv = ["parity", "--model", "unused", "--library", "unused", "--fixture", "unused", "--out", str(out)]
                with patch("opendlss_cuda.fixture.run_parity", return_value={"status": status}), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(argv), code)
                content = (out / "report.json").read_bytes()
                self.assertEqual(json.loads(content)["status"], status)
                with patch("opendlss_cuda.fixture.run_parity") as run, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(argv), 2)
                    run.assert_not_called()
                self.assertEqual(content, (out / "report.json").read_bytes())

    def test_invalid_parity_never_publishes_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "report"
            with patch("opendlss_cuda.fixture.run_parity", side_effect=ValueError("missing boundary")), contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["parity", "--model", "unused", "--library", "unused", "--fixture", "unused", "--out", str(out)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads((out / "report.json").read_text())["status"], "invalid")

    def test_report_write_failure_returns_error_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "report"
            stderr, stdout = io.StringIO(), io.StringIO()
            with patch("opendlss_cuda.fixture.run_parity", return_value={"status": "passed"}), \
                    patch.object(cli, "write_report", side_effect=OSError("disk full")) as write, \
                    contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                code = cli.main(["parity", "--model", "unused", "--library", "unused", "--fixture", "unused", "--out", str(out)])
            self.assertEqual(code, 2)
            self.assertEqual(write.call_count, 1)
            self.assertEqual(stdout.getvalue(), "")
            report = json.loads(stderr.getvalue())
            self.assertEqual(report["status"], "runtime_error")
            self.assertEqual(report["error"], "disk full")
            self.assertFalse((out / "report.json").exists())

    def test_existing_partial_is_preserved_without_success_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "report"
            partial = out / "report.json.partial"
            def run(**kwargs):
                partial.write_bytes(b"interrupted receipt")
                return {"status": "passed"}
            stderr = io.StringIO()
            with patch("opendlss_cuda.fixture.run_parity", side_effect=run), \
                    patch.object(cli, "write_report", wraps=cli.write_report) as write, \
                    contextlib.redirect_stderr(stderr):
                code = cli.main(["parity", "--model", "unused", "--library", "unused", "--fixture", "unused", "--out", str(out)])
            self.assertEqual(code, 2)
            self.assertEqual(write.call_count, 1)
            report = json.loads(stderr.getvalue())
            self.assertEqual(report["status"], "runtime_error")
            self.assertEqual(report["error_type"], "FileExistsError")
            self.assertEqual(partial.read_bytes(), b"interrupted receipt")
            self.assertFalse((out / "report.json").exists())

    def test_operator_error_survives_error_report_write_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "report"
            stderr = io.StringIO()
            with patch("opendlss_cuda.fixture.run_parity", side_effect=RuntimeError("operator failed")), \
                    patch.object(cli, "write_report", side_effect=OSError("disk full")) as write, \
                    contextlib.redirect_stderr(stderr):
                code = cli.main(["parity", "--model", "unused", "--library", "unused", "--fixture", "unused", "--out", str(out)])
            self.assertEqual(code, 2)
            self.assertEqual(write.call_count, 1)
            report = json.loads(stderr.getvalue())
            self.assertEqual(report["status"], "runtime_error")
            self.assertEqual(report["error_type"], "RuntimeError")
            self.assertEqual(report["error"], "operator failed")
            self.assertEqual(report["report_write_error"], {"error_type": "OSError", "error": "disk full"})
            self.assertFalse((out / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
