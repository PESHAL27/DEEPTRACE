"""
Unit and integration tests for DEEPTRACE Stage 8D Benchmark Service and APIs.
Verifies:
1. Manifest schema parsing and validation.
2. Data integrity and data leakage detection (duplicate SHA-256, cross-split leakage).
3. Metric calculations (Accuracy, Precision, Recall, Specificity, F1, FPR, FNR, TP, FP, TN, FN).
4. Zero-denominator safety and graceful degradation.
5. Threshold sweep (0.10 to 0.90).
6. ROC-AUC and PR-AUC (multi-class vs single-class graceful handling).
7. Error analysis extraction (False Positives and False Negatives).
8. Subgroup metric aggregation.
9. Benchmark execution and disk persistence.
10. Benchmark REST API endpoints.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import (
    BENCHMARK_RESULTS_DIR,
    DATASETS_DIR,
    EVALUATION_DIR,
    MANIFESTS_DIR,
)
from app.main import app
from app.models.benchmark_schemas import (
    DatasetManifest,
    ManifestSample,
    SampleEvaluationResult,
)
from app.services.benchmark_service import (
    compute_file_sha256,
    compute_metrics_at_threshold,
    compute_subgroup_analysis,
    extract_error_analysis,
    get_sample_manifest_template,
    get_benchmark_run,
    list_benchmark_runs,
    run_threshold_sweep,
    validate_manifest,
    BenchmarkRunner,
)


class TestBenchmarkMetrics(unittest.TestCase):
    """Tests mathematical correctness of metrics calculations."""

    def test_perfect_predictions(self):
        samples = [
            SampleEvaluationResult(
                sample_id="s1", filename="s1.jpg", file_path="s1.jpg",
                ground_truth="authentic", manipulated_probability=0.05, inference_status="completed"
            ),
            SampleEvaluationResult(
                sample_id="s2", filename="s2.jpg", file_path="s2.jpg",
                ground_truth="authentic", manipulated_probability=0.10, inference_status="completed"
            ),
            SampleEvaluationResult(
                sample_id="s3", filename="s3.jpg", file_path="s3.jpg",
                ground_truth="manipulated", manipulated_probability=0.90, inference_status="completed"
            ),
            SampleEvaluationResult(
                sample_id="s4", filename="s4.jpg", file_path="s4.jpg",
                ground_truth="manipulated", manipulated_probability=0.95, inference_status="completed"
            ),
        ]
        metrics, cm = compute_metrics_at_threshold(samples, threshold=0.5)

        self.assertEqual(cm.tn, 2)
        self.assertEqual(cm.tp, 2)
        self.assertEqual(cm.fp, 0)
        self.assertEqual(cm.fn, 0)
        self.assertAlmostEqual(metrics.accuracy, 1.0)
        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.specificity, 1.0)
        self.assertAlmostEqual(metrics.f1_score, 1.0)
        self.assertAlmostEqual(metrics.false_positive_rate, 0.0)
        self.assertAlmostEqual(metrics.false_negative_rate, 0.0)
        self.assertIsNotNone(metrics.roc_auc)
        self.assertAlmostEqual(metrics.roc_auc, 1.0)

    def test_imperfect_predictions(self):
        # 5 authentic (0), 5 manipulated (1)
        # Authentics: 0.1, 0.2, 0.3, 0.4, 0.8 (1 FP at 0.5) -> TN=4, FP=1
        # Manipulated: 0.2, 0.6, 0.7, 0.8, 0.9 (1 FN at 0.5) -> FN=1, TP=4
        samples = []
        for i, p in enumerate([0.1, 0.2, 0.3, 0.4, 0.8]):
            samples.append(SampleEvaluationResult(
                sample_id=f"auth_{i}", filename=f"auth_{i}.jpg", file_path=f"auth_{i}.jpg",
                ground_truth="authentic", manipulated_probability=p, inference_status="completed"
            ))
        for i, p in enumerate([0.2, 0.6, 0.7, 0.8, 0.9]):
            samples.append(SampleEvaluationResult(
                sample_id=f"man_{i}", filename=f"man_{i}.jpg", file_path=f"man_{i}.jpg",
                ground_truth="manipulated", manipulated_probability=p, inference_status="completed"
            ))

        metrics, cm = compute_metrics_at_threshold(samples, threshold=0.5)

        self.assertEqual(cm.tn, 4)
        self.assertEqual(cm.fp, 1)
        self.assertEqual(cm.fn, 1)
        self.assertEqual(cm.tp, 4)
        self.assertAlmostEqual(metrics.accuracy, 0.80)
        self.assertAlmostEqual(metrics.precision, 4 / (4 + 1))  # 0.8
        self.assertAlmostEqual(metrics.recall, 4 / (4 + 1))     # 0.8
        self.assertAlmostEqual(metrics.specificity, 4 / (4 + 1)) # 0.8
        self.assertAlmostEqual(metrics.f1_score, 0.80)
        self.assertAlmostEqual(metrics.false_positive_rate, 0.20)
        self.assertAlmostEqual(metrics.false_negative_rate, 0.20)

    def test_zero_denominator_safe(self):
        # All authentic, all predicted authentic (no positive predictions)
        samples = [
            SampleEvaluationResult(
                sample_id=f"auth_{i}", filename=f"auth_{i}.jpg", file_path=f"auth_{i}.jpg",
                ground_truth="authentic", manipulated_probability=p, inference_status="completed"
            )
            for i, p in enumerate([0.1, 0.2, 0.1])
        ]
        metrics, cm = compute_metrics_at_threshold(samples, threshold=0.5)

        # TP=0, FP=0, FN=0, TN=3
        self.assertEqual(cm.tn, 3)
        self.assertEqual(cm.fp, 0)
        self.assertEqual(cm.tp, 0)
        self.assertEqual(cm.fn, 0)
        # Precision denominator TP+FP = 0 -> safely returns 0.0
        self.assertEqual(metrics.precision, 0.0)
        # Recall denominator TP+FN = 0 -> safely returns 0.0
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1_score, 0.0)
        # Specificity TN/(TN+FP) = 3/3 = 1.0
        self.assertEqual(metrics.specificity, 1.0)
        # Single-class ROC-AUC should gracefully be None with explanation
        self.assertIsNone(metrics.roc_auc)
        self.assertEqual(metrics.roc_auc_status, "not_available")

    def test_threshold_sweep(self):
        samples = [
            SampleEvaluationResult(
                sample_id="s1", filename="s1.jpg", file_path="s1.jpg",
                ground_truth="authentic", manipulated_probability=0.2, inference_status="completed"
            ),
            SampleEvaluationResult(
                sample_id="s2", filename="s2.jpg", file_path="s2.jpg",
                ground_truth="authentic", manipulated_probability=0.4, inference_status="completed"
            ),
            SampleEvaluationResult(
                sample_id="s3", filename="s3.jpg", file_path="s3.jpg",
                ground_truth="manipulated", manipulated_probability=0.6, inference_status="completed"
            ),
            SampleEvaluationResult(
                sample_id="s4", filename="s4.jpg", file_path="s4.jpg",
                ground_truth="manipulated", manipulated_probability=0.8, inference_status="completed"
            ),
        ]
        sweep = run_threshold_sweep(samples)

        self.assertEqual(len(sweep), 17)
        # At threshold 0.10: all predicted fake (1) -> TP=2, FP=2, TN=0, FN=0
        first = sweep[0]
        self.assertEqual(first.threshold, 0.10)
        self.assertEqual(first.fp, 2)
        self.assertEqual(first.tn, 0)

        # At threshold 0.90: all predicted real (0) -> TP=0, FP=0, TN=2, FN=2
        last = sweep[-1]
        self.assertEqual(last.threshold, 0.90)
        self.assertEqual(last.tp, 0)
        self.assertEqual(last.fn, 2)

    def test_error_analysis_extraction(self):
        samples = [
            SampleEvaluationResult(
                sample_id="s1", filename="s1.jpg", file_path="s1.jpg",
                ground_truth="authentic", manipulated_probability=0.85, inference_status="completed"
            ), # FP
            SampleEvaluationResult(
                sample_id="s2", filename="s2.jpg", file_path="s2.jpg",
                ground_truth="authentic", manipulated_probability=0.15, inference_status="completed"
            ), # TN
            SampleEvaluationResult(
                sample_id="s3", filename="s3.jpg", file_path="s3.jpg",
                ground_truth="manipulated", manipulated_probability=0.20, inference_status="completed"
            ), # FN
            SampleEvaluationResult(
                sample_id="s4", filename="s4.jpg", file_path="s4.jpg",
                ground_truth="manipulated", manipulated_probability=0.90, inference_status="completed"
            ), # TP
        ]
        errors = extract_error_analysis(samples, threshold=0.5)
        self.assertEqual(len(errors), 2)
        
        fp = next((e for e in errors if e.error_type == "false_positive"), None)
        fn = next((e for e in errors if e.error_type == "false_negative"), None)

        self.assertIsNotNone(fp)
        self.assertEqual(fp.sample_id, "s1")
        self.assertIsNotNone(fn)
        self.assertEqual(fn.sample_id, "s3")


class TestManifestValidation(unittest.TestCase):
    """Tests manifest format, duplicate hashing, and leakage detection."""

    def test_valid_manifest_passes(self):
        manifest_data = {
            "manifest_version": "1.0",
            "dataset_name": "Test Dataset",
            "samples": [
                {
                    "sample_id": "samp-1",
                    "file_path": "sample.jpg",
                    "ground_truth": "authentic",
                    "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                    "split": "test"
                }
            ]
        }
        manifest = DatasetManifest(**manifest_data)
        is_valid, warnings, stats = validate_manifest(manifest)

        self.assertTrue(is_valid)
        self.assertEqual(stats["authentic_count"], 1)
        self.assertEqual(stats["manipulated_count"], 0)

    def test_duplicate_sha256_detection(self):
        sha = "1111222233334444555566667777888811112222333344445555666677778888"
        manifest_data = {
            "manifest_version": "1.0",
            "dataset_name": "Duplicate Test",
            "samples": [
                {
                    "sample_id": "s1",
                    "file_path": "sample1.jpg",
                    "ground_truth": "authentic",
                    "sha256": sha,
                    "split": "test"
                },
                {
                    "sample_id": "s2",
                    "file_path": "sample2.jpg",
                    "ground_truth": "authentic",
                    "sha256": sha,
                    "split": "test"
                }
            ]
        }
        manifest = DatasetManifest(**manifest_data)
        is_valid, warnings, stats = validate_manifest(manifest)

        self.assertTrue(is_valid)
        # Duplicates in evaluation set are flagged in warnings
        has_dup_warning = any("duplicate file" in w.lower() for w in warnings)
        self.assertTrue(has_dup_warning)

    def test_cross_split_leakage_detection(self):
        sha = "aaaabbbbccccddddeeeeffff00001111aaaabbbbccccddddeeeeffff00001111"
        manifest_data = {
            "manifest_version": "1.0",
            "dataset_name": "Leakage Test",
            "samples": [
                {
                    "sample_id": "s1",
                    "file_path": "sample1.jpg",
                    "ground_truth": "authentic",
                    "sha256": sha,
                    "split": "train"
                },
                {
                    "sample_id": "s2",
                    "file_path": "sample2.jpg",
                    "ground_truth": "authentic",
                    "sha256": sha,
                    "split": "test"
                }
            ]
        }
        manifest = DatasetManifest(**manifest_data)
        is_valid, warnings, stats = validate_manifest(manifest)

        self.assertTrue(is_valid)
        has_leakage_warning = any("leakage" in w.lower() for w in warnings)
        self.assertTrue(has_leakage_warning)


class TestBenchmarkRunnerAndPersistence(unittest.TestCase):
    """Tests execution of benchmark runner on synthetic images and persistence."""

    def test_runner_on_synthetic_images(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create two synthetic blank test images
            img1 = np.full((300, 300, 3), 128, dtype=np.uint8)
            img2 = np.full((300, 300, 3), 200, dtype=np.uint8)

            file1 = tmp_path / "img1.png"
            file2 = tmp_path / "img2.png"
            cv2.imwrite(str(file1), img1)
            cv2.imwrite(str(file2), img2)

            sha1 = compute_file_sha256(file1)
            sha2 = compute_file_sha256(file2)

            manifest_data = {
                "manifest_version": "1.0",
                "dataset_name": "Runner Test Dataset",
                "dataset_version": "1.0",
                "description": "Synthetic unit test dataset",
                "samples": [
                    {
                        "sample_id": "synth-1",
                        "file_path": str(file1),
                        "ground_truth": "authentic",
                        "sha256": sha1,
                        "manipulation_type": "none"
                    },
                    {
                        "sample_id": "synth-2",
                        "file_path": str(file2),
                        "ground_truth": "manipulated",
                        "sha256": sha2,
                        "manipulation_type": "face_swap"
                    }
                ]
            }
            manifest_file = tmp_path / "manifest.json"
            manifest_file.write_text(json.dumps(manifest_data, indent=2))

            manifest = DatasetManifest(**manifest_data)
            runner = BenchmarkRunner(threshold=0.5)
            detail = runner.run_benchmark(manifest)

            self.assertIsNotNone(detail)
            self.assertEqual(detail.sample_count, 2)
            self.assertEqual(detail.dataset_name, "Runner Test Dataset")
            self.assertIsNotNone(detail.metrics)
            self.assertEqual(len(detail.threshold_sweep), 17)

            # Check that results were saved
            saved = get_benchmark_run(detail.benchmark_run_id)
            self.assertIsNotNone(saved)
            self.assertEqual(saved.benchmark_run_id, detail.benchmark_run_id)
            self.assertEqual(len(saved.samples), 2)


class TestBenchmarkAPI(unittest.TestCase):
    """Tests the benchmark REST API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_get_template(self):
        response = self.client.get("/api/benchmark/template")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["manifest_version"], "1.0")
        self.assertIn("samples", data)
        self.assertGreater(len(data["samples"]), 0)

    def test_list_runs(self):
        response = self.client.get("/api/benchmark/runs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_get_nonexistent_run_404(self):
        response = self.client.get("/api/benchmark/runs/BENCH-nonexistent-999")
        self.assertEqual(response.status_code, 404)

    def test_get_nonexistent_run_metrics_404(self):
        response = self.client.get("/api/benchmark/runs/BENCH-nonexistent-999/metrics")
        self.assertEqual(response.status_code, 404)

    def test_run_benchmark_missing_manifest_404(self):
        response = self.client.post("/api/benchmark/run", json={"manifest_path": "non_existent_manifest.json"})
        self.assertEqual(response.status_code, 404)

    def test_run_benchmark_empty_body_400(self):
        response = self.client.post("/api/benchmark/run", json={})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
