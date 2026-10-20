import os
import sys
import time
import json
import threading
import urllib.request
import urllib.parse
from pathlib import Path
import psutil

BACKEND_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BACKEND_DIR.parent
FIXTURE_KNOWN_REAL = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_real.jpg"
FIXTURE_FAKE_VIDEO = BACKEND_DIR / "test_fixtures" / "test_fake_video.mp4"
FIXTURE_REAL_VIDEO = BACKEND_DIR / "test_fixtures" / "test_real_video.mp4"
FIXTURE_ALL_IN_ONE = WORKSPACE_ROOT / "deepfake_model" / "DeepfakeBench" / "training" / "dataset" / "library" / "all_in_one.jpg"
API_URL = "http://127.0.0.1:8000/api"

# Find uvicorn process
def find_server_process():
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == 8000 and conn.status == 'LISTEN':
            if conn.pid:
                return psutil.Process(conn.pid)
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = " ".join(p.info['cmdline'] or [])
            if "uvicorn" in cmd and "8000" in cmd and p.info['pid'] != 1256:
                return psutil.Process(p.info['pid'])
        except Exception:
            pass
    return psutil.Process(25604)

server_proc = find_server_process()
if not server_proc:
    print("Warning: Could not find uvicorn process, using current process for memory.")
    server_proc = psutil.Process(os.getpid())

print(f"Monitoring server process PID: {server_proc.pid}")

class MemoryTracker:
    def __init__(self, proc, interval_sec=0.005):
        self.proc = proc
        self.interval_sec = interval_sec
        self.running = False
        self.peak_rss = 0
        self.thread = None

    def start(self):
        self.running = True
        try:
            self.peak_rss = self.proc.memory_info().rss
        except Exception:
            self.peak_rss = 0
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def _monitor(self):
        while self.running:
            try:
                rss = self.proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                break
            time.sleep(self.interval_sec)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        return self.peak_rss

def upload_and_analyze(file_path: Path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = file_path.name
    content_type = "video/mp4" if filename.endswith(".mp4") else "image/jpeg"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{API_URL}/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def get_request(url: str):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.read()

results = {}

print("\n--- Starting Post-Optimization Measurement ---")

# Warm up detector
print("Warming up...")
upload_and_analyze(FIXTURE_KNOWN_REAL)
time.sleep(0.5)

# Scenario A: Normal Image
print("\n[Scenario A] Normal Image (known_real.jpg)")
tracker = MemoryTracker(server_proc)
rss_before = server_proc.memory_info().rss
tracker.start()
t0 = time.perf_counter()
res_a = upload_and_analyze(FIXTURE_KNOWN_REAL)
lat_a = time.perf_counter() - t0
peak_a = tracker.stop()
rss_after_a = server_proc.memory_info().rss
results['A'] = {
    'latency': lat_a,
    'rss_before': rss_before / 1024**2,
    'peak_rss': peak_a / 1024**2,
    'rss_after': rss_after_a / 1024**2,
    'prediction': res_a['prediction'],
    'probability': res_a.get('probability'),
    'faces': len(res_a.get('faces', [])),
    'gradcam': res_a.get('gradcam_available'),
    'response_bytes': len(json.dumps(res_a))
}
print(f"Scenario A: {lat_a:.3f}s | Peak RSS: {peak_a/1024**2:.2f} MB | Pred: {res_a['prediction']} | Prob: {res_a['probability']}")

# Scenario B: Normal Video
print("\n[Scenario B] Normal Video (test_fake_video.mp4)")
tracker = MemoryTracker(server_proc)
rss_before = server_proc.memory_info().rss
tracker.start()
t0 = time.perf_counter()
res_b = upload_and_analyze(FIXTURE_FAKE_VIDEO)
lat_b = time.perf_counter() - t0
peak_b = tracker.stop()
rss_after_b = server_proc.memory_info().rss
results['B'] = {
    'latency': lat_b,
    'rss_before': rss_before / 1024**2,
    'peak_rss': peak_b / 1024**2,
    'rss_after': rss_after_b / 1024**2,
    'prediction': res_b['prediction'],
    'video_probability': res_b.get('video_probability'),
    'frames_analyzed': res_b.get('frames_analyzed'),
    'response_bytes': len(json.dumps(res_b))
}
print(f"Scenario B: {lat_b:.3f}s | Peak RSS: {peak_b/1024**2:.2f} MB | Pred: {res_b['prediction']} | Prob: {res_b['video_probability']}")

# Scenario C: Multiple-face Image
print("\n[Scenario C] Multi-Face Image (all_in_one.jpg)")
tracker = MemoryTracker(server_proc)
rss_before = server_proc.memory_info().rss
tracker.start()
t0 = time.perf_counter()
res_c = upload_and_analyze(FIXTURE_ALL_IN_ONE)
lat_c = time.perf_counter() - t0
peak_c = tracker.stop()
rss_after_c = server_proc.memory_info().rss
results['C'] = {
    'latency': lat_c,
    'rss_before': rss_before / 1024**2,
    'peak_rss': peak_c / 1024**2,
    'rss_after': rss_after_c / 1024**2,
    'prediction': res_c['prediction'],
    'probability': res_c.get('probability'),
    'faces': len(res_c.get('faces', [])),
    'response_bytes': len(json.dumps(res_c))
}
print(f"Scenario C: {lat_c:.3f}s | Peak RSS: {peak_c/1024**2:.2f} MB | Pred: {res_c['prediction']} | Prob: {res_c['probability']} | Faces: {len(res_c.get('faces', []))}")

# Scenario D: Multi-frame Video
print("\n[Scenario D] Multi-frame Video (test_real_video.mp4)")
tracker = MemoryTracker(server_proc)
rss_before = server_proc.memory_info().rss
tracker.start()
t0 = time.perf_counter()
res_d = upload_and_analyze(FIXTURE_REAL_VIDEO)
lat_d = time.perf_counter() - t0
peak_d = tracker.stop()
rss_after_d = server_proc.memory_info().rss
results['D'] = {
    'latency': lat_d,
    'rss_before': rss_before / 1024**2,
    'peak_rss': peak_d / 1024**2,
    'rss_after': rss_after_d / 1024**2,
    'prediction': res_d['prediction'],
    'video_probability': res_d.get('video_probability'),
    'frames_analyzed': res_d.get('frames_analyzed'),
    'response_bytes': len(json.dumps(res_d))
}
print(f"Scenario D: {lat_d:.3f}s | Peak RSS: {peak_d/1024**2:.2f} MB | Pred: {res_d['prediction']} | Prob: {res_d['video_probability']}")

# Scenario E: Report Generation (JSON)
print("\n[Scenario E] Report JSON")
case_id = res_a['analysis_id']
report_id = f"REP-{case_id}"
tracker = MemoryTracker(server_proc)
rss_before = server_proc.memory_info().rss
tracker.start()
t0 = time.perf_counter()
raw_e = get_request(f"{API_URL}/reports/{report_id}")
lat_e = time.perf_counter() - t0
peak_e = tracker.stop()
rss_after_e = server_proc.memory_info().rss
results['E'] = {
    'latency': lat_e,
    'rss_before': rss_before / 1024**2,
    'peak_rss': peak_e / 1024**2,
    'rss_after': rss_after_e / 1024**2,
    'response_bytes': len(raw_e)
}
print(f"Scenario E: {lat_e:.4f}s | Size: {len(raw_e)} B")

# Scenario F: Printable Report (HTML)
print("\n[Scenario F] Printable Report HTML")
tracker = MemoryTracker(server_proc)
rss_before = server_proc.memory_info().rss
tracker.start()
t0 = time.perf_counter()
raw_f = get_request(f"{API_URL}/reports/{report_id}/printable")
lat_f = time.perf_counter() - t0
peak_f = tracker.stop()
rss_after_f = server_proc.memory_info().rss
results['F'] = {
    'latency': lat_f,
    'rss_before': rss_before / 1024**2,
    'peak_rss': peak_f / 1024**2,
    'rss_after': rss_after_f / 1024**2,
    'response_bytes': len(raw_f)
}
print(f"Scenario F: {lat_f:.4f}s | Size: {len(raw_f)} B")

# Scenario G: Concurrent Image Requests (2 workers)
print("\n[Scenario G] Concurrent Images (2 workers)")
tracker = MemoryTracker(server_proc)
rss_before = server_proc.memory_info().rss
tracker.start()
thread_lats = []
def worker_image():
    w_t0 = time.perf_counter()
    upload_and_analyze(FIXTURE_KNOWN_REAL)
    thread_lats.append(time.perf_counter() - w_t0)

t1 = threading.Thread(target=worker_image)
t2 = threading.Thread(target=worker_image)
t0 = time.perf_counter()
t1.start(); t2.start()
t1.join(); t2.join()
lat_g = time.perf_counter() - t0
peak_g = tracker.stop()
rss_after_g = server_proc.memory_info().rss
results['G'] = {
    'latency': lat_g,
    'thread_lats': thread_lats,
    'rss_before': rss_before / 1024**2,
    'peak_rss': peak_g / 1024**2,
    'rss_after': rss_after_g / 1024**2
}
print(f"Scenario G: Total {lat_g:.3f}s (Threads: {[round(l, 3) for l in thread_lats]})")

# Scenario H: Concurrent Video Requests (2 workers)
print("\n[Scenario H] Concurrent Videos (2 workers)")
tracker = MemoryTracker(server_proc)
rss_before = server_proc.memory_info().rss
tracker.start()
v_thread_lats = []
def worker_video():
    w_t0 = time.perf_counter()
    upload_and_analyze(FIXTURE_FAKE_VIDEO)
    v_thread_lats.append(time.perf_counter() - w_t0)

t1 = threading.Thread(target=worker_video)
t2 = threading.Thread(target=worker_video)
t0 = time.perf_counter()
t1.start(); t2.start()
t1.join(); t2.join()
lat_h = time.perf_counter() - t0
peak_h = tracker.stop()
rss_after_h = server_proc.memory_info().rss
results['H'] = {
    'latency': lat_h,
    'thread_lats': v_thread_lats,
    'rss_before': rss_before / 1024**2,
    'peak_rss': peak_h / 1024**2,
    'rss_after': rss_after_h / 1024**2
}
print(f"Scenario H: Total {lat_h:.3f}s (Threads: {[round(l, 3) for l in v_thread_lats]})")

# Output JSON of results
with open(BACKEND_DIR / "evaluation" / "post_optimization_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n--- Completed. Saved to evaluation/post_optimization_results.json ---")
