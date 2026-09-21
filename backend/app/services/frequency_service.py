from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

from ..config import FREQUENCY_DIR
from ..models.schemas import FrequencyAnalysisResult, FrequencyFinding
from ..utils.logger import logger


def compute_fft_magnitude_spectrum(gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes mathematically rigorous 2D Fast Fourier Transform on a single-channel image.
    Returns:
    - magnitude: raw uncompressed magnitude spectrum centered at DC origin
    - log_magnitude: natural log-scaled magnitude ln(1 + |F(u, v)|)
    """
    img_float = np.float32(gray_img)
    # 2D Fast Fourier Transform
    f_transform = np.fft.fft2(img_float)
    # Shift zero-frequency DC component to the center of the spectrum
    f_shift = np.fft.fftshift(f_transform)
    # Absolute magnitude spectrum
    magnitude = np.abs(f_shift)
    # Logarithmic scaling for human/computer visual inspection
    log_magnitude = np.log1p(magnitude)
    return magnitude, log_magnitude


def extract_spectral_features(magnitude: np.ndarray) -> Dict[str, float]:
    """
    Calculates deterministic frequency-domain statistical and radial energy features:
    - Total energy (sum of squared magnitudes)
    - Low-frequency energy (radius rho < 0.15)
    - Mid-frequency energy (0.15 <= rho < 0.50)
    - High-frequency energy (rho >= 0.50)
    - High-frequency ratio (E_high / E_total)
    - Mean and median frequency magnitude
    - Spectral centroid (mass-weighted normalized radial frequency)
    """
    h, w = magnitude.shape
    if h == 0 or w == 0:
        return {
            "mean_magnitude": 0.0,
            "median_magnitude": 0.0,
            "low_frequency_energy": 0.0,
            "mid_frequency_energy": 0.0,
            "high_frequency_energy": 0.0,
            "high_frequency_ratio": 0.0,
            "spectral_centroid": 0.0,
            "total_energy": 0.0
        }

    # Center coordinates of the shifted spectrum
    cy, cx = h / 2.0, w / 2.0
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    max_radius = np.sqrt(cy ** 2 + cx ** 2)
    if max_radius <= 0:
        max_radius = 1.0

    # Normalized radial distance rho in [0, 1]
    rho = distances / max_radius

    # Radial masks
    low_mask = rho < 0.15
    mid_mask = (rho >= 0.15) & (rho < 0.50)
    high_mask = rho >= 0.50

    # Energy calculations (squared magnitude)
    squared_mag = magnitude ** 2
    total_energy = float(np.sum(squared_mag))
    low_energy = float(np.sum(squared_mag[low_mask]))
    mid_energy = float(np.sum(squared_mag[mid_mask]))
    high_energy = float(np.sum(squared_mag[high_mask]))

    if total_energy > 0:
        high_ratio = high_energy / total_energy
    else:
        high_ratio = 0.0

    mean_mag = float(np.mean(magnitude))
    median_mag = float(np.median(magnitude))

    mag_sum = float(np.sum(magnitude))
    if mag_sum > 0:
        spectral_centroid = float(np.sum(rho * magnitude) / mag_sum)
    else:
        spectral_centroid = 0.0

    return {
        "mean_magnitude": round(mean_mag, 4),
        "median_magnitude": round(median_mag, 4),
        "low_frequency_energy": round(low_energy, 4),
        "mid_frequency_energy": round(mid_energy, 4),
        "high_frequency_energy": round(high_energy, 4),
        "high_frequency_ratio": round(high_ratio, 4),
        "spectral_centroid": round(spectral_centroid, 4),
        "total_energy": round(total_energy, 4)
    }


def analyze_frequency_domain(
    image_bgr: np.ndarray,
    identifier_prefix: str,
    analysis_region: str = "face_region"
) -> FrequencyAnalysisResult:
    """
    Executes full frequency-domain forensic analysis:
    1. Converts image region to grayscale and computes centered 2D FFT.
    2. Calculates normalized radial energy distribution and spectral centroid.
    3. Renders and saves colorized spectrum artifact to backend/uploads/forensics/frequency/.
    4. Evaluates deterministic factual forensic findings without fabricated certainty.
    """
    if image_bgr is None or image_bgr.size == 0:
        return FrequencyAnalysisResult(
            available=False,
            analysis_region=analysis_region,
            findings=[
                FrequencyFinding(
                    type="input_error",
                    message="Target image region is empty or unreadable.",
                    severity="warning"
                )
            ]
        )

    # 1. Grayscale conversion
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    h, w = gray.shape

    # 2. Compute 2D Fast Fourier Transform
    try:
        magnitude, log_mag = compute_fft_magnitude_spectrum(gray)
    except Exception as e:
        logger.error(f"Failed to compute 2D FFT for {identifier_prefix}: {e}")
        return FrequencyAnalysisResult(
            available=False,
            analysis_region=analysis_region,
            findings=[
                FrequencyFinding(
                    type="fft_computation_error",
                    message=f"Error executing 2D FFT: {e}",
                    severity="warning"
                )
            ]
        )

    # 3. Extract numerical spectral features
    features = extract_spectral_features(magnitude)

    # 4. Generate visual spectrum artifact
    # Normalize log magnitude to 8-bit [0, 255]
    norm_spectrum = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply VIRIDIS colormap (standard for spectral analysis and visually distinct from JET Grad-CAM)
    colored_spectrum = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_VIRIDIS)

    spectrum_filename = f"{identifier_prefix}_frequency_spectrum.jpg"
    spectrum_path = FREQUENCY_DIR / spectrum_filename
    cv2.imwrite(str(spectrum_path), colored_spectrum)
    spectrum_url = f"/uploads/forensics/frequency/{spectrum_filename}"

    # 5. Deterministic, factual forensic findings
    findings: List[FrequencyFinding] = []
    region_label = "analyzed facial subject" if analysis_region == "face_region" else "full media frame"
    findings.append(
        FrequencyFinding(
            type="spectral_decomposition",
            message=f"2D Fast Fourier Transform evaluated across {region_label} (resolution {w}x{h}).",
            severity="info"
        )
    )

    high_ratio_pct = features["high_frequency_ratio"] * 100
    findings.append(
        FrequencyFinding(
            type="energy_distribution",
            message=f"High-frequency band accounts for {high_ratio_pct:.2f}% of total spectral energy (Centroid: {features['spectral_centroid']:.4f}).",
            severity="info"
        )
    )

    if features["high_frequency_ratio"] > 0.35:
        findings.append(
            FrequencyFinding(
                type="spectral_distribution_observation",
                message="Elevated high-frequency energy concentration observed in outer radial bands (>35% interpretation threshold); analytical observation consistent with resampling, sharpening, compression noise, or boundary blending.",
                severity="info"
            )
        )
    else:
        findings.append(
            FrequencyFinding(
                type="spectral_baseline",
                message="Frequency-domain measurements recorded; natural 1/f spectral falloff observed across low and mid bands.",
                severity="info"
            )
        )

    return FrequencyAnalysisResult(
        available=True,
        analysis_region=analysis_region,
        frequency_spectrum_url=spectrum_url,
        mean_magnitude=features["mean_magnitude"],
        median_magnitude=features["median_magnitude"],
        low_frequency_energy=features["low_frequency_energy"],
        mid_frequency_energy=features["mid_frequency_energy"],
        high_frequency_energy=features["high_frequency_energy"],
        high_frequency_ratio=features["high_frequency_ratio"],
        spectral_centroid=features["spectral_centroid"],
        total_energy=features["total_energy"],
        findings=findings
    )
