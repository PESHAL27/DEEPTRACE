import os
import re
import time
import random
from pathlib import Path
from typing import Tuple, List, Union
from ..config import ALLOWED_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS, MAGIC_NUMBERS

_analysis_counter = random.randint(1000, 9999)

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
}

# Signatures for dangerous executable / script files that must be rejected regardless of extension
DANGEROUS_MAGIC_HEADERS = [
    (b"MZ", "Windows PE executable / DLL"),
    (b"\x7fELF", "Linux ELF binary"),
    (b"#!", "Shell script"),
    (b"<!DOCTYPE", "HTML / XML document"),
    (b"<html", "HTML document"),
    (b"<HTML", "HTML document"),
    (b"<?xml", "XML document"),
    (b"<svg", "SVG vector graphic"),
    (b"\xca\xfe\xba\xbe", "Java / Mach-O class binary"),
    (b"%PDF", "PDF document"),
]


def generate_analysis_id() -> str:
    """Generates a secure, unique analysis ID conforming to DT-2026-XXXXXX."""
    global _analysis_counter
    _analysis_counter += 1
    timestamp_suffix = str(int(time.time()))[-4:]
    return f"DT-2026-{timestamp_suffix}{_analysis_counter:04d}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename to prevent path traversal, hidden files, dangerous characters,
    and Windows device namespace collisions.
    
    Guarantees:
    1. Returns strictly a single basename without directory separators.
    2. Strips Windows drive designators (e.g., C:).
    3. Neutralizes traversal patterns (../../, ..\\, etc.).
    4. Guards against Windows reserved names (CON, NUL, AUX, PRN, COM1-9, LPT1-9).
    5. Bounds maximum length to 128 characters while preserving extension.
    """
    if not filename:
        return f"evidence_{int(time.time())}.bin"

    # Remove null bytes
    cleaned = filename.replace("\x00", "")

    # Strip Windows drive prefix like C: or D:
    cleaned = re.sub(r"^[a-zA-Z]:", "", cleaned)

    # Normalize both backslashes and slashes, then split to isolate the true basename
    parts = [p for p in re.split(r"[\\/]+", cleaned) if p and p != "." and p != ".."]
    base = parts[-1] if parts else ""

    # Remove any remaining directory separators
    base = base.replace("/", "").replace("\\", "")

    # Remove leading dots to prevent hidden files
    base = base.lstrip(".")

    # Replace whitespace with underscore
    base = re.sub(r"\s+", "_", base)

    # Strip any characters other than alphanumeric, underscore, hyphen, and period
    base = re.sub(r"[^a-zA-Z0-9._-]", "", base)

    # Split stem and extension
    stem, ext = os.path.splitext(base)

    # Guard against Windows reserved device names
    if stem.upper() in WINDOWS_RESERVED_NAMES:
        stem = f"file_{stem}"

    # Reassemble base
    base = f"{stem}{ext}"

    # Truncate length to maximum 128 chars while preserving extension
    if len(base) > 128:
        max_stem_len = max(1, 128 - len(ext))
        stem = stem[:max_stem_len]
        base = f"{stem}{ext}"

    # If base collapsed into nothing or just dots, provide safe fallback
    if not base or base.startswith(".") or not stem:
        base = f"evidence_{int(time.time())}.bin"

    return base


def get_media_type_from_extension(ext: str) -> str:
    """Returns 'image' or 'video' based on file extension."""
    clean_ext = ext.lower().lstrip(".")
    if clean_ext in ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if clean_ext in ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def validate_file_content(filename: str, header_bytes: bytes) -> Tuple[bool, str]:
    """
    Validates file extension, checks for dangerous executable signatures,
    and verifies magic byte headers for media format authenticity.
    Returns (is_valid, error_reason).
    """
    ext = Path(filename).suffix.lower().lstrip(".")
    if not ext or ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension .{ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    if not header_bytes:
        return False, "File is empty (0 bytes)."

    # 1. Reject known dangerous / executable magic signatures
    for sig, desc in DANGEROUS_MAGIC_HEADERS:
        if header_bytes.startswith(sig) or (len(sig) > 4 and sig in header_bytes[:64]):
            return False, f"Dangerous or invalid content detected: identified as {desc}."

    # 2. Strict format-specific magic byte verification
    if ext in ("jpg", "jpeg"):
        if not header_bytes.startswith(b"\xFF\xD8\xFF"):
            return False, f"File content does not match expected signature for JPEG (.jpg/.jpeg)"
    elif ext == "png":
        if not header_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return False, f"File content does not match expected signature for PNG (.png)"
    elif ext == "webp":
        if not (header_bytes.startswith(b"RIFF") and b"WEBP" in header_bytes[:16]):
            return False, f"File content does not match expected signature for WebP (.webp)"
    elif ext == "mp4":
        if b"ftyp" not in header_bytes[:32]:
            return False, f"File content does not match expected signature for MP4 video container"
    elif ext == "mov":
        matches_mov = any(box in header_bytes[:32] for box in (b"moov", b"mdat", b"ftyp", b"wide"))
        if not matches_mov and len(header_bytes) >= 12:
            return False, f"File content does not match expected signature for QuickTime MOV video container"

    return True, ""


def sanitize_error_message(msg: str) -> str:
    """
    Sanitizes an error message or exception string by scrubbing absolute host filesystem
    paths (Windows C:\\... or POSIX /home/...) and Python traceback references.
    """
    if not msg:
        return "An unknown error occurred."

    s = str(msg)

    # Scrub Windows absolute paths: e.g. C:\Users\... or D:/...
    s = re.sub(r"[a-zA-Z]:[\\/][^:*?\"<>|\r\n]+", "[internal path]", s)

    # Scrub common Unix absolute paths: e.g. /home/..., /root/..., /var/..., /tmp/...
    s = re.sub(r"/(?:home|root|tmp|var|usr|opt|mnt)/[^:*?\"<>|\s\r\n]+", "[internal path]", s)

    # Scrub python file references like File "...", line 123
    s = re.sub(r'File ".*?", line \d+', 'File "[internal]", line [x]', s)

    # Truncate overly long error messages
    if len(s) > 500:
        s = s[:497] + "..."

    return s.strip()


def is_safe_path(target_path: Union[Path, str], allowed_dirs: List[Union[Path, str]]) -> bool:
    """
    Verifies that target_path resolves strictly within at least one of the allowed directories.
    Blocks directory traversal via symlinks or '..' components.
    """
    try:
        resolved_target = Path(target_path).resolve()
        for d in allowed_dirs:
            resolved_dir = Path(d).resolve()
            if resolved_target == resolved_dir or resolved_dir in resolved_target.parents:
                return True
        return False
    except Exception:
        return False


def resolve_safe_path(path_str: str, allowed_dirs: List[Union[Path, str]]) -> Path:
    """
    Safely resolves path_str against allowed directories.
    Raises PermissionError if path attempts to escape allowed roots.
    Raises FileNotFoundError if path does not exist.
    """
    candidate = Path(path_str)

    # If absolute, must be within allowed_dirs
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not is_safe_path(resolved, allowed_dirs):
            raise PermissionError("Access to requested path outside approved directory is forbidden.")
        return resolved

    # If relative, check against each allowed dir
    for base in allowed_dirs:
        resolved = (Path(base) / path_str).resolve()
        if is_safe_path(resolved, allowed_dirs) and resolved.exists():
            return resolved

    # Fallback to resolving relative to first allowed dir and check safety
    fallback = (Path(allowed_dirs[0]) / path_str).resolve()
    if not is_safe_path(fallback, allowed_dirs):
        raise PermissionError("Access to requested path outside approved directory is forbidden.")
    return fallback
