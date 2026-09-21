import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status

from ..models.benchmark_schemas import (
    BenchmarkRunRequest,
    BenchmarkRunSummary,
    BenchmarkRunDetail,
    EvaluationMetrics,
    ConfusionMatrix,
    ErrorAnalysisItem,
    DatasetManifest
)
from ..services.benchmark_service import (
    BenchmarkRunner,
    list_benchmark_runs,
    get_benchmark_run,
    get_sample_manifest_template,
    resolve_sample_path
)
from ..utils.logger import logger
from ..utils.file_utils import sanitize_error_message

router = APIRouter(prefix="/benchmark", tags=["Benchmark & Evaluation"])

BENCHMARK_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@router.post("/run", response_model=BenchmarkRunDetail, status_code=status.HTTP_200_OK)
async def execute_benchmark_run(request: BenchmarkRunRequest):
    """
    Executes a reproducible evaluation benchmark across a labeled dataset manifest.
    Accepts an inline manifest or path to a JSON manifest file.
    Does NOT alter production detector configuration.
    """
    manifest: Optional[DatasetManifest] = None

    if request.manifest_content:
        manifest = request.manifest_content
    elif request.manifest_path:
        try:
            target_path = resolve_sample_path(request.manifest_path)
        except PermissionError as pe:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to manifest path outside approved benchmark directory is forbidden."
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid manifest path."
            )

        if not target_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Manifest file not found: '{request.manifest_path}'."
            )
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                manifest = DatasetManifest(**data)
        except Exception as e:
            logger.error(f"Error reading manifest {target_path.name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid manifest format or corrupted JSON file."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'manifest_path' or 'manifest_content' must be provided."
        )

    runner = BenchmarkRunner(threshold=request.threshold)
    try:
        run_detail = runner.run_benchmark(manifest, max_samples=request.max_samples)
        return run_detail
    except ValueError as e:
        clean_err = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=clean_err
        )
    except Exception as e:
        clean_err = sanitize_error_message(str(e))
        logger.error(f"Benchmark execution failed: {clean_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution error: {clean_err}"
        )


@router.get("/runs", response_model=List[BenchmarkRunSummary])
async def get_all_benchmark_runs():
    """Retrieves summaries of all completed benchmark evaluations."""
    return list_benchmark_runs()


@router.get("/runs/{run_id}", response_model=BenchmarkRunDetail)
async def get_benchmark_run_detail(run_id: str):
    """Retrieves full evaluation report, metrics, confusion matrix, and errors for a specific run."""
    if not BENCHMARK_ID_REGEX.match(run_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid benchmark run ID format."
        )
    run = get_benchmark_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run '{run_id}' not found."
        )
    return run


@router.get("/runs/{run_id}/metrics")
async def get_benchmark_run_metrics(run_id: str):
    """Retrieves metrics and confusion matrix for a specific run."""
    if not BENCHMARK_ID_REGEX.match(run_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid benchmark run ID format."
        )
    run = get_benchmark_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run '{run_id}' not found."
        )
    return {
        "benchmark_run_id": run.benchmark_run_id,
        "dataset_name": run.dataset_name,
        "threshold": run.threshold,
        "sample_count": run.sample_count,
        "metrics": run.metrics,
        "confusion_matrix": run.confusion_matrix,
        "threshold_sweep": run.threshold_sweep
    }


@router.get("/runs/{run_id}/errors", response_model=List[ErrorAnalysisItem])
async def get_benchmark_run_errors(run_id: str):
    """Retrieves False Positives and False Negatives error analysis for a specific run."""
    if not BENCHMARK_ID_REGEX.match(run_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid benchmark run ID format."
        )
    run = get_benchmark_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run '{run_id}' not found."
        )
    return run.errors


@router.get("/template", response_model=Dict[str, Any])
async def get_benchmark_manifest_template():
    """Retrieves an official evaluation dataset manifest template with explanatory annotations."""
    return get_sample_manifest_template()
