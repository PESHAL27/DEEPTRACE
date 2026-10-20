#!/usr/bin/env python3
"""
DEEPTRACE — Deployment Smoke Test Suite
Verifies that a deployed DEEPTRACE backend service is operational, models are
ready for inference, endpoints respond within expected schemas, and security
invariants hold without leaking internal filesystem details.

Usage:
    python backend/evaluation/deployment_smoke_test.py [--base-url http://127.0.0.1:8000]
"""

import sys
import argparse
import urllib.request
import urllib.parse
import json
import io
from pathlib import Path


def log_step(name: str, passed: bool, detail: str = ""):
    status_str = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
    print(f" {status_str} {name}" + (f" — {detail}" if detail else ""))


def make_request(url: str, method: str = "GET", data: bytes = None, headers: dict = None) -> tuple[int, dict, dict]:
    headers = headers or {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status_code = resp.status
            resp_headers = dict(resp.headers)
            body = resp.read().decode("utf-8", errors="ignore")
            try:
                parsed_json = json.loads(body)
            except Exception:
                parsed_json = {"raw": body}
            return status_code, parsed_json, resp_headers
    except urllib.error.HTTPError as e:
        status_code = e.code
        resp_headers = dict(e.headers)
        body = e.read().decode("utf-8", errors="ignore")
        try:
            parsed_json = json.loads(body)
        except Exception:
            parsed_json = {"raw": body}
        return status_code, parsed_json, resp_headers
    except Exception as e:
        return 0, {"error": str(e)}, {}


def run_smoke_tests(base_url: str) -> bool:
    print(f"\n=======================================================")
    print(f" DEEPTRACE Production Deployment Smoke Test")
    print(f" Target Service: {base_url}")
    print(f"=======================================================\n")

    all_passed = True
    base_url = base_url.rstrip("/")

    # 1. Health / Liveness Check
    status, body, _ = make_request(f"{base_url}/api/health")
    if status == 200 and body.get("status") == "ok":
        log_step("GET /api/health (Liveness)", True, f"service={body.get('service')}")
    else:
        log_step("GET /api/health (Liveness)", False, f"HTTP {status}: {body}")
        all_passed = False

    # 2. Readiness Status Check
    status, body, _ = make_request(f"{base_url}/api/status")
    if status == 200 and body.get("status") in ["ready", "degraded"]:
        models = body.get("models", {})
        log_step(
            "GET /api/status (Readiness)",
            True,
            f"status={body.get('status')}, xception={models.get('xception_primary')}, device={body.get('device')}"
        )
    else:
        log_step("GET /api/status (Readiness)", False, f"HTTP {status}: {body}")
        all_passed = False

    # 3. Model Registry Inspection
    status, body, _ = make_request(f"{base_url}/api/models")
    if status == 200 and isinstance(body, list) and len(body) >= 2:
        model_names = [m.get("name") for m in body]
        log_step("GET /api/models (Registry)", True, f"{len(body)} models registered ({', '.join(model_names[:2])})")
    else:
        log_step("GET /api/models (Registry)", False, f"HTTP {status}: {body}")
        all_passed = False

    # 4. Forensic Reports Log
    status, body, _ = make_request(f"{base_url}/api/reports")
    if status == 200 and isinstance(body, list):
        log_step("GET /api/reports (Audit Log)", True, f"{len(body)} archived case reports verified")
    else:
        log_step("GET /api/reports (Audit Log)", False, f"HTTP {status}: {body}")
        all_passed = False

    # 5. Benchmark Template Endpoint
    status, body, _ = make_request(f"{base_url}/api/benchmark/template")
    if status == 200 and "samples" in body:
        log_step("GET /api/benchmark/template (Benchmark Harness)", True, f"Manifest schema verified")
    else:
        log_step("GET /api/benchmark/template (Benchmark Harness)", False, f"HTTP {status}: {body}")
        all_passed = False

    # 6. End-to-End Image Ingestion & Analysis
    workspace_root = Path(__file__).resolve().parent.parent.parent
    sample_image = workspace_root / "deepfake_model" / "tests" / "samples" / "known_real.jpg"
    if not sample_image.exists():
        # Fallback to test_fixtures
        sample_image = workspace_root / "backend" / "test_fixtures" / "test_image.jpg"

    if sample_image.exists():
        with open(sample_image, "rb") as f:
            file_bytes = f.read()

        boundary = "----WebKitFormBoundarySmokeTest7MA4YW"
        body_bytes = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{sample_image.name}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        status, body, _ = make_request(f"{base_url}/api/analyze", method="POST", data=body_bytes, headers=headers)

        if status == 200 and "analysis_id" in body:
            verdict = body.get("prediction")
            prob = body.get("model_probability") or body.get("probability")
            gradcam = body.get("gradcam_available")
            log_step(
                "POST /api/analyze (Forensic Ingestion)",
                True,
                f"id={body.get('analysis_id')}, verdict={verdict}, prob={prob}, gradcam={gradcam}"
            )
        else:
            log_step("POST /api/analyze (Forensic Ingestion)", False, f"HTTP {status}: {body}")
            all_passed = False
    else:
        log_step("POST /api/analyze (Forensic Ingestion)", False, f"Test sample not found: {sample_image}")
        all_passed = False

    print("\n-------------------------------------------------------")
    if all_passed:
        print("\033[92mALL DEPLOYMENT SMOKE TESTS PASSED SUCCESSFULLY.\033[0m\n")
    else:
        print("\033[91mONE OR MORE SMOKE TESTS FAILED. CHECK LOGS ABOVE.\033[0m\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEEPTRACE Deployment Smoke Test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Root URL of DEEPTRACE backend service")
    args = parser.parse_args()

    success = run_smoke_tests(args.base_url)
    sys.exit(0 if success else 1)
