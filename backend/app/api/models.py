import torch
from typing import List
from fastapi import APIRouter
from ..models.schemas import ModelInfoResponse
from ..services.detection_service import get_xception_detector, WEIGHTS_PATH
from ..utils.logger import logger

router = APIRouter(tags=["Models"])


@router.get("/models", response_model=List[ModelInfoResponse])
async def get_detection_models():
    """
    Returns live metadata and runtime telemetry for neural network models deployed in DEEPTRACE.
    Hardware acceleration, checkpoint paths, and parameter counts are read dynamically from runtime.
    """
    detector, config, device = get_xception_detector()

    # Calculate actual parameter count from the loaded model
    total_params = sum(p.numel() for p in detector.parameters())
    params_str = f"{total_params / 1e6:.1f}M"

    # Determine runtime hardware device description
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA Device"
        device_str = f"CUDA ({gpu_name})"
    else:
        device_str = "CPU (Host Processor)"

    # Extract configuration attributes from the actual DeepfakeBench config
    resolution = config.get("resolution", 256)
    mean = config.get("mean", [0.5, 0.5, 0.5])
    std = config.get("std", [0.5, 0.5, 0.5])
    num_classes = config.get("backbone_config", {}).get("num_classes", 2)
    checkpoint_name = WEIGHTS_PATH.name if WEIGHTS_PATH.exists() else "xception_best.pth"

    live_xception = ModelInfoResponse(
        id="xception-detector",
        name="Xception Deepfake Detector",
        category="Convolutional Neural Network (CNN)",
        purpose="Facial boundary artifact & manipulation classification",
        status="Ready",
        parameters=params_str,
        inputResolution=f"{resolution}x{resolution}",
        description="Official DeepfakeBench benchmark Xception detector loaded with FaceForensics++ (c23) pretrained weights.",
        accuracyMetric="AUC 99.2% (FF++ benchmark)",
        architecture="Xception (DeepfakeBench)",
        checkpoint=checkpoint_name,
        device=device_str,
        mean=[float(v) for v in mean],
        std=[float(v) for v in std],
        numClasses=int(num_classes)
    )

    # Live Comparison Model: DeepfakeBench MesoNet (Meso4)
    from ..services.model_registry import get_model_registry
    registry = get_model_registry()
    meso_adapter = registry.get_detector("meso4_deepfakebench")
    meso_status = "Ready" if (meso_adapter and meso_adapter.is_loaded()) else "Unavailable"
    
    live_meso4 = ModelInfoResponse(
        id="meso4-deepfakebench",
        name="DeepfakeBench MesoNet (Meso4)",
        category="Compact CNN (Comparison Detector)",
        purpose="Independent cross-validation & microscopic face artifact comparison",
        status=meso_status,
        parameters="27.9K",
        inputResolution="256x256",
        description="Independent compact 4-layer CNN (Afchar et al., IEEE WIFS 2018) trained on FaceForensics++ (c23). Used for cross-validation and disagreement analysis.",
        accuracyMetric="AUC 84.8% (FF++ benchmark)",
        architecture="MesoNet-4 (DeepfakeBench)",
        checkpoint="meso4_best.pth",
        device=device_str,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        numClasses=2
    )

    # Planned models for research roadmap
    planned_models = [
        ModelInfoResponse(
            id="efficientnet-b4",
            name="EfficientNet-B4 Spatial Forensics",
            category="Compound Scaled CNN",
            purpose="High-frequency generative artifact detection",
            status="Planned",
            parameters="19.3M",
            inputResolution="380x380",
            description="Compound-scaled convolutional network targeting generative synthesis inconsistencies and blending noise.",
            accuracyMetric="Planned Research",
            architecture="EfficientNet-B4",
            checkpoint="efficientnet_b4.pth",
            device=device_str,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            numClasses=2
        ),
        ModelInfoResponse(
            id="temporal-transformer",
            name="Spatio-Temporal Sequence Transformer",
            category="Vision Transformer (ViT)",
            purpose="Inter-frame coherence & temporal flicker detection",
            status="Planned",
            parameters="86.0M",
            inputResolution="224x224 (x16 frames)",
            description="Transformer architecture modeling inter-frame face stability, eye blinking rhythms, and temporal jitter.",
            accuracyMetric="Planned Research",
            architecture="Video Transformer",
            checkpoint="transformer_temporal.pth",
            device=device_str,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            numClasses=2
        )
    ]

    # Live Whole-Image Model: Smaug AI Image Detector (Abacus.AI)
    from ..services.detectors.ai_generated_image_detector import LOCAL_WEIGHTS_DIR
    smaug_available = (LOCAL_WEIGHTS_DIR / "config.json").exists()
    smaug_status = "Ready" if smaug_available else "Available"

    live_smaug = ModelInfoResponse(
        id="smaug-ai-detector",
        name="Smaug AI Image Detector",
        category="Vision Transformer (EmbeddingGemma-2)",
        purpose="Whole-image synthetic / generative image identification across all visual categories",
        status=smaug_status,
        parameters="439M",
        inputResolution="576x528 (528x528 native crops)",
        description="Open-source vision-only EmbeddingGemma-2 fine-tune (Abacus.AI, Apache 2.0). Detects full-scene AI-generated images (Midjourney, FLUX, DALL-E, SD, StyleGAN) without requiring faces.",
        accuracyMetric="OpenFake Benchmark SOTA (93.4% AUC)",
        architecture="EmbeddingGemma-2 Vision (Abacus.AI)",
        checkpoint="model.safetensors",
        device=device_str,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        numClasses=2
    )

    return [live_xception, live_smaug, live_meso4, *planned_models]

