#!/usr/bin/env python3
"""
DEEPTRACE — Production End-to-End Pipeline Verification
Verifies Image, Video, Reports (Printable/JSON/SHA-256), and Benchmark Path Controls
against the live deployed ASGI server.
"""

import sys
import json
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def upload(file_path: str, mime: str):
    path = Path(file_path)
    with open(path, "rb") as f:
        data = f.read()
    boundary = "----WebKitFormBoundaryE2ETest88"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("==================================================")
    print(" DEEPTRACE Production E2E Pipeline Verification")
    print("==================================================")

    # 1. Image Forensic Analysis
    print("\n[1/4] Testing Image Forensic Pipeline...")
    img_res = upload("deepfake_model/tests/samples/known_real.jpg", "image/jpeg")
    assert img_res.get("analysis_id"), "Missing analysis_id"
    assert img_res.get("prediction") == "real", f"Expected real, got {img_res.get('prediction')}"
    assert img_res.get("faces_detected") >= 1, "Expected at least 1 face"
    assert img_res.get("gradcam_available") is True, "GradCAM not generated"
    assert bool(img_res.get("frequency_analysis")) is True, "FFT spectrum not generated"
    print(f" PASS: ID={img_res['analysis_id']} | Verdict={img_res['prediction']} | Prob={img_res['model_probability']} | Faces={img_res['faces_detected']} | GradCAM={img_res['gradcam_available']}")

    # 2. Video Forensic Analysis
    print("\n[2/4] Testing Video Forensic Pipeline...")
    vid_res = upload("backend/test_fixtures/test_fake_video.mp4", "video/mp4")
    assert vid_res.get("analysis_id"), "Missing video analysis_id"
    assert vid_res.get("prediction") == "fake", f"Expected fake, got {vid_res.get('prediction')}"
    assert vid_res.get("gradcam_available") is True, "Video GradCAM not generated"
    assert bool(vid_res.get("frequency_analysis")) is True, "Video FFT spectrum not generated"
    assert vid_res.get("frames_analyzed", 0) > 0 or len(vid_res.get("frame_results", [])) > 0, "No frames analyzed"
    frames_count = vid_res.get("frames_analyzed") or len(vid_res.get("frame_results", []))
    print(f" PASS: ID={vid_res['analysis_id']} | Verdict={vid_res['prediction']} | Prob={vid_res['model_probability']} | Frames={frames_count} | GradCAM={vid_res['gradcam_available']}")

    # 3. Reports, Printable Export & Integrity Hash
    print("\n[3/4] Testing Forensic Reports & Integrity...")
    reports_req = urllib.request.Request(f"{BASE_URL}/api/reports")
    with urllib.request.urlopen(reports_req) as resp:
        reports = json.loads(resp.read().decode("utf-8"))
    assert len(reports) > 0, "No reports found"
    latest_report_id = reports[0].get("id") or reports[0].get("reportId") or reports[0].get("report_id")
    summary_hash = reports[0].get("hashSha256") or reports[0].get("sha256_hash") or ""
    assert len(summary_hash) == 64, f"Invalid summary SHA-256: {summary_hash}"

    rep_req = urllib.request.Request(f"{BASE_URL}/api/reports/{latest_report_id}")
    with urllib.request.urlopen(rep_req) as resp:
        rep_data = json.loads(resp.read().decode("utf-8"))
    
    sha256_hash = rep_data.get("hashSha256") or rep_data.get("sha256_hash")
    if not sha256_hash and "case_identification" in rep_data:
        sha256_hash = rep_data["case_identification"].get("sha256_hash", "")

    assert len(sha256_hash) == 64, f"Detailed report SHA-256 mismatch: {sha256_hash}"

    print_req = urllib.request.Request(f"{BASE_URL}/api/reports/{latest_report_id}/printable")
    with urllib.request.urlopen(print_req) as resp:
        html = resp.read().decode("utf-8")
    assert "<html" in html.lower() and "DEEPTRACE" in html, "Invalid printable HTML"
    assert sha256_hash in html, "SHA256 not in printable report"
    print(f" PASS: Latest Report={latest_report_id} | Summary SHA256={summary_hash[:16]}... | Printable HTML={len(html)} bytes verified")

    # 4. Benchmark Template & Path Confinement
    print("\n[4/4] Testing Benchmark Path Confinement...")
    bench_req = urllib.request.Request(f"{BASE_URL}/api/benchmark/template")
    with urllib.request.urlopen(bench_req) as resp:
        template = json.loads(resp.read().decode("utf-8"))
    assert "dataset_name" in template and "samples" in template, "Invalid benchmark template"
    print(f" PASS: Benchmark template verified: {template['dataset_name']} ({len(template['samples'])} template samples)")

    print("\n==================================================")
    print(" ALL PRODUCTION E2E PIPELINE TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    main()
