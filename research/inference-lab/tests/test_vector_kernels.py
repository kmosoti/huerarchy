import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_lab.common import ROOT
from research_lab.vector_kernels import check_matches, reference_scores, write_matrix


class NativeVectorTests(unittest.TestCase):
    def test_worker_binary_format_signed_values_and_normalization(self):
        binary = ROOT / "target/release/vector-search"
        if not binary.exists():
            self.skipTest("build the native worker with cargo build --release")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            codes = np.array([[-128, 2, -3, 4, -5, 6, -7, 8, 9],
                              [127, -2, 3, -4, 5, -6, 7, -8, -9],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.int8)
            queries = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
            corpus, question = directory / "codes.vec", directory / "queries.vec"
            write_matrix(corpus, codes)
            write_matrix(question, queries)
            run = subprocess.run([str(binary), str(corpus), str(question), "3"], check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in run.stdout.splitlines()]
            summary = rows.pop()
            error, _ = check_matches(rows, reference_scores(codes, queries, np.array([0, 1])))
            self.assertLess(error, 2e-6)
            self.assertEqual(rows[0]["ids"][0], 0)
            self.assertEqual(rows[1]["ids"][0], 2)
            self.assertEqual(summary["corpus_bytes"], codes.nbytes + codes.shape[0]*4)
            # A truncated payload must fail before attempting to search.
            corpus.write_bytes(corpus.read_bytes()[:-1])
            result = subprocess.run([str(binary), str(corpus), str(question), "3"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_writer_rejects_wrong_dtype_shape_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.vec"
            for array in (np.zeros((0, 2), dtype=np.int8), np.zeros(2, dtype=np.float32)):
                with self.assertRaises(ValueError):
                    write_matrix(path, array)
            with self.assertRaises(TypeError):
                write_matrix(path, np.ones((2, 3), dtype=np.float64))
            with self.assertRaises(ValueError):
                write_matrix(path, np.array([[np.nan]], dtype=np.float32))

    def test_conformance_detects_wrong_scores_and_omitted_matches(self):
        references = {0: np.array([0.9, 0.8, 0.1])}
        with self.assertRaises(AssertionError):
            check_matches([{"query": 0, "ids": [0], "scores": [0.5]}], references)
        with self.assertRaises(AssertionError):
            check_matches([{"query": 0, "ids": [1], "scores": [0.8]}], references)
        error, _ = check_matches([{"query": 0, "ids": [0], "scores": [0.9]}], references)
        self.assertEqual(error, 0.0)


if __name__ == "__main__":
    unittest.main()
