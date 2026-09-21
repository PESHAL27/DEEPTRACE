import os
import sys
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

def log_test(name: str, passed: bool, details: str = ""):
    status_str = "PASS" if passed else "FAIL"
    print(f"[{status_str}] {name} {details}")
    if not passed:
        sys.exit(1)

def create_test_media():
    """Generates real test media for testing the backend."""
    test_dir = Path("backend/test_fixtures")
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample JPG
    jpg_path = test_dir / "test_image.jpg"
    img_array = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img_array, (320, 240), 100, (0, 180, 255), -1)
    cv2.imwrite(str(jpg_path), img_array)

    # 2. Sample PNG
    png_path = test_dir / "test_image.png"
    img_pil = Image.fromarray(img_array)
    img_pil.save(png_path, format="PNG")

    # 3. Sample MP4 video (30 frames, 1 second, 30fps)
    mp4_path = test_dir / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(mp4_path), fourcc, 30.0, (640, 480))
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    # 4. Invalid file
    invalid_path = test_dir / "test_script.txt"
    invalid_path.write_text("This is an unsupported text file.")

    return jpg_path, png_path, mp4_path, invalid_path

def upload_file_multipart(url: str, file_path: Path):
    """Performs multipart/form-data upload using standard library."""
    boundary = "----DEEPTRACEBoundary" + str(int(time.time()))
    filename = file_path.name
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("latin-1") + file_bytes + f"\r\n--{boundary}--\r\n".encode("latin-1")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            parsed = json.loads(error_body)
        except Exception:
            parsed = {"raw": error_body}
        return e.code, parsed

def run_tests():
    print("========================================")
    print("DEEPTRACE BACKEND AUTOMATED VERIFICATION")
    print("========================================")

    # 1. Health check
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/health") as res:
            health_data = json.loads(res.read().decode("utf-8"))
            log_test(
                "GET /api/health", 
                res.status == 200 and health_data.get("status") == "ok",
                f"-> status: {health_data.get('status')}, service: {health_data.get('service')}"
            )
    except Exception as e:
        log_test("GET /api/health", False, f"Connection failed: {e}")

    # Generate fixtures
    jpg_path, png_path, mp4_path, invalid_path = create_test_media()

    # 2. Upload JPG (no face fixture)
    status_code, jpg_res = upload_file_multipart(f"{BASE_URL}/api/analyze", jpg_path)
    is_valid_jpg = (
        status_code == 200 and
        jpg_res.get("media_type") == "image" and
        jpg_res.get("metadata", {}).get("width") == 640 and
        jpg_res.get("metadata", {}).get("height") == 480 and
        jpg_res.get("detection", {}).get("status") in ["no_face_detected", "model_not_loaded"] and
        jpg_res.get("prediction") is None
    )
    log_test("POST /api/analyze (JPG no face)", is_valid_jpg, f"-> ID: {jpg_res.get('analysis_id')}, 640x480, status: {jpg_res.get('status')}")

    # 3. Upload Face Image (if sample exists)
    face_sample = Path("public/assets/sample_frame.jpg")
    if face_sample.exists():
        status_code, face_res = upload_file_multipart(f"{BASE_URL}/api/analyze", face_sample)
        is_valid_face = (
            status_code == 200 and
            face_res.get("media_type") == "image" and
            face_res.get("detection", {}).get("status") == "completed" and
            (face_res.get("detection", {}).get("model") == "Xception" or face_res.get("detection", {}).get("model", {}).get("name") == "Xception") and
            face_res.get("prediction") in ["real", "fake"] and
            isinstance(face_res.get("probability"), float) and
            len(face_res.get("faces", [])) > 0
        )
        log_test(
            "POST /api/analyze (Face Image)", 
            is_valid_face, 
            f"-> ID: {face_res.get('analysis_id')}, Prediction: {face_res.get('prediction')}, Prob: {face_res.get('probability')}, Faces: {len(face_res.get('faces', []))}"
        )

    # 4. Upload PNG
    status_code, png_res = upload_file_multipart(f"{BASE_URL}/api/analyze", png_path)
    is_valid_png = (
        status_code == 200 and
        png_res.get("media_type") == "image" and
        png_res.get("metadata", {}).get("format") == "PNG"
    )
    log_test("POST /api/analyze (PNG)", is_valid_png, f"-> ID: {png_res.get('analysis_id')}, format: PNG")

    # 5. Upload MP4
    status_code, mp4_res = upload_file_multipart(f"{BASE_URL}/api/analyze", mp4_path)
    is_valid_mp4 = (
        status_code == 200 and
        mp4_res.get("media_type") == "video" and
        mp4_res.get("metadata", {}).get("frame_count") == 30 and
        mp4_res.get("metadata", {}).get("duration") == 1.0 and
        mp4_res.get("detection", {}).get("status") == "model_not_loaded" and
        mp4_res.get("prediction") is None
    )
    log_test(
        "POST /api/analyze (MP4)", 
        is_valid_mp4, 
        f"-> ID: {mp4_res.get('analysis_id')}, FPS: {mp4_res.get('metadata', {}).get('fps')}, Frames: {mp4_res.get('metadata', {}).get('frame_count')}"
    )

    # 5. Unsupported file rejection
    status_code, invalid_res = upload_file_multipart(f"{BASE_URL}/api/analyze", invalid_path)
    is_rejected = status_code == 415 or status_code == 400
    log_test("POST /api/analyze (Unsupported .txt rejection)", is_rejected, f"-> HTTP {status_code} properly rejected")

    # 6. Retrieve stored case
    case_id = jpg_res.get("analysis_id")
    if case_id:
        with urllib.request.urlopen(f"{BASE_URL}/api/analysis/{case_id}") as res:
            retrieved = json.loads(res.read().decode("utf-8"))
            log_test(
                f"GET /api/analysis/{case_id}", 
                retrieved.get("analysis_id") == case_id,
                f"-> Retrieved successfully for {retrieved.get('filename')}"
            )

    print("========================================")
    print("ALL BACKEND VERIFICATION TESTS PASSED!")
    print("========================================")

if __name__ == "__main__":
    run_tests()
