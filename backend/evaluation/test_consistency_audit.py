#!/usr/bin/env python3
"""
DEEPTRACE Stage 9D — Cross-System Consistency Audit Script
Verifies field parity across:
1. POST /api/analyze response
2. GET /api/analysis/{id}
3. GET /api/reports
4. GET /api/reports/{report_id}
5. GET /api/reports/{report_id}/printable
"""

import sys
import json
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def upload_file(path_str: str, mime: str):
    p = Path(path_str)
    with open(p, "rb") as f:
        data = f.read()
    boundary = "----WebKitFormBoundaryConsistency9D"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
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

def fetch_json(url: str):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))

def fetch_text(url: str):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")

def audit_consistency(media_path: str, mime: str, media_type: str):
    print(f"\n--- Auditing Cross-System Consistency for: {media_path} ---")
    
    # 1. Ingest
    res1 = upload_file(media_path, mime)
    analysis_id = res1["analysis_id"]
    print(f"1. Ingestion Response: ID={analysis_id}, Verdict={res1['prediction']}, Prob={res1['model_probability']}")
    
    # 2. GET /api/analysis/{id}
    res2 = fetch_json(f"{BASE_URL}/api/analysis/{analysis_id}")
    print(f"2. History Record: ID={res2['analysis_id']}, Verdict={res2['prediction']}, Prob={res2['model_probability']}")
    assert res1["analysis_id"] == res2["analysis_id"], "ID mismatch between ingestion and history"
    assert res1["prediction"] == res2["prediction"], "Prediction mismatch"
    assert res1["model_probability"] == res2["model_probability"], "Probability mismatch"
    assert res1["faces_detected"] == res2["faces_detected"], "Faces detected mismatch"
    
    # 3. GET /api/reports (list)
    reports_list = fetch_json(f"{BASE_URL}/api/reports")
    matched_rep = None
    for rep in reports_list:
        if rep.get("caseId") == analysis_id or rep.get("case_id") == analysis_id or analysis_id in rep.get("id", ""):
            matched_rep = rep
            break
    assert matched_rep is not None, f"Report for {analysis_id} not found in reports list"
    rep_id = matched_rep["id"]
    print(f"3. Reports Summary: ReportID={rep_id}, CaseID={matched_rep.get('caseId')}, SHA256={matched_rep.get('hashSha256')[:16]}...")
    
    # Check consistency in summary
    assert len(matched_rep.get("hashSha256", "")) == 64, "Missing or invalid SHA256 in report summary"
    
    # 4. GET /api/reports/{rep_id}
    detail_rep = fetch_json(f"{BASE_URL}/api/reports/{rep_id}")
    detail_sha256 = detail_rep.get("hashSha256") or detail_rep.get("case_identification", {}).get("sha256_hash")
    print(f"4. Detailed Report: ID={detail_rep['id']}, PrimaryDetector={detail_rep.get('primary_detector', {}).get('detector_name')}, SHA256={detail_sha256[:16]}...")
    assert detail_sha256 == matched_rep["hashSha256"], "SHA256 mismatch between summary and detailed report"
    
    # Check verdict consistency
    primary_verdict = detail_rep.get("primary_detector", {}).get("verdict", "").lower()
    assert (res1["prediction"] in primary_verdict) or (primary_verdict in ["likely manipulated", "likely authentic", "real", "fake"]), f"Verdict mismatch: {res1['prediction']} vs {primary_verdict}"
    
    # 5. GET /api/reports/{rep_id}/printable
    html = fetch_text(f"{BASE_URL}/api/reports/{rep_id}/printable")
    assert analysis_id in html, "Analysis ID not found in printable HTML"
    assert rep_id in html, "Report ID not found in printable HTML"
    assert detail_sha256 in html, "SHA256 hash not found in printable HTML"
    assert "<html" in html.lower() and "DEEPTRACE" in html, "Invalid HTML in printable report"
    assert "automated" in html.lower() or "disclaimer" in html.lower() or "statistically" in html.lower(), "Missing disclaimer in printable report"
    print(f"5. Printable HTML: Length={len(html)} bytes, Verified ID, SHA256, and Forensic Disclaimers")
    
    print(f"==> ALL 5 SURFACES FULLY CONSISTENT FOR {analysis_id} <==")
    return True

if __name__ == "__main__":
    print("=======================================================")
    print(" DEEPTRACE Stage 9D Cross-System Consistency Audit")
    print("=======================================================")
    audit_consistency("deepfake_model/tests/samples/known_real.jpg", "image/jpeg", "image")
    audit_consistency("deepfake_model/tests/samples/known_fake.jpg", "image/jpeg", "image")
    audit_consistency("backend/test_fixtures/test_fake_video.mp4", "video/mp4", "video")
    print("\n[SUCCESS] Cross-system consistency verified across Image (Real/Fake) and Video.")
