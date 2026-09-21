import hashlib
import re
from typing import List, Union, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse
from ..models.schemas import AnalysisResponse
from ..models.report_schemas import DetailedForensicReport
from ..config import UPLOADS_DIR
from ..utils.logger import logger
from .analysis import analysis_store
from ..services.report_service import get_report_service, compute_file_sha256

router = APIRouter(tags=["Reports"])

REPORT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def build_forensic_report(record: AnalysisResponse) -> DetailedForensicReport:
    """Generates an official forensic report from a completed analysis record."""
    service = get_report_service()
    return service.generate_professional_report(record)


@router.get("/reports", response_model=List[DetailedForensicReport])
async def get_forensic_reports():
    """Retrieves official forensic audit reports for all analyzed media cases."""
    service = get_report_service()
    report_dict = {}

    # 1. Load any persisted reports from disk first (immutability)
    if service.reports_dir.exists():
        for p in sorted(service.reports_dir.glob("REP-*.json"), key=lambda x: x.stat().st_mtime):
            rep = service.load_persisted_report(p.stem)
            if rep:
                report_dict[rep.id] = rep

    # 2. Add or update with live active analysis records from memory
    for record in analysis_store.values():
        rep = service.generate_professional_report(record)
        report_dict[rep.id] = rep

    # Return in reverse chronological order (newest first)
    return list(reversed(list(report_dict.values())))


@router.get("/reports/{report_id}", response_model=DetailedForensicReport)
async def get_forensic_report(report_id: str):
    """Retrieves a specific forensic report by report ID or analysis case ID."""
    if not REPORT_ID_REGEX.match(report_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report ID format."
        )

    service = get_report_service()
    clean_id = report_id.removeprefix("REP-")

    # 1. Look up in active memory analysis_store
    if clean_id in analysis_store:
        return service.generate_professional_report(analysis_store[clean_id])

    for analysis_id, record in analysis_store.items():
        if f"REP-{analysis_id}" == report_id or analysis_id == report_id:
            return service.generate_professional_report(record)

    # 2. Look up in persistent disk storage
    persisted = service.load_persisted_report(report_id)
    if persisted:
        return persisted

    persisted_case = service.load_persisted_report(f"REP-{clean_id}")
    if persisted_case:
        return persisted_case

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Forensic report '{report_id}' not found."
    )


@router.get("/reports/{report_id}/printable", response_class=HTMLResponse)
async def get_printable_forensic_report(report_id: str):
    """Returns a standalone, high-fidelity print-optimized HTML forensic report."""
    if not REPORT_ID_REGEX.match(report_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report ID format."
        )
    report = await get_forensic_report(report_id)
    service = get_report_service()
    html_content = service.render_printable_report_html(report)
    return HTMLResponse(content=html_content, status_code=200)
