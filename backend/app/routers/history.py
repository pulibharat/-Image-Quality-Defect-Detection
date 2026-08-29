import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalysisResult
from app.schemas import AnalysisListOut, AnalysisOut
from app.services import storage
from app.services.pipeline import to_analysis_out_dict

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/analyses", response_model=AnalysisListOut)
def list_analyses(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    total = db.execute(select(func.count()).select_from(AnalysisResult)).scalar_one()
    rows = db.execute(
        select(AnalysisResult).order_by(AnalysisResult.created_at.desc()).limit(limit).offset(offset)
    ).scalars().all()

    items = [
        {
            "id": r.id,
            "created_at": r.created_at,
            "original_filename": r.original_filename,
            "quality_score": r.quality_score,
            "quality_label": r.quality_label,
            "issue_count": len(json.loads(r.issues_json)),
            "image_url": f"/api/analyses/{r.id}/image",
        }
        for r in rows
    ]
    return AnalysisListOut(total=total, limit=limit, offset=offset, items=items)


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    row = db.get(AnalysisResult, analysis_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    return to_analysis_out_dict(row)


@router.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    row = db.get(AnalysisResult, analysis_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    db.delete(row)
    db.commit()
    return None


@router.get("/analyses/{analysis_id}/image")
def get_analysis_image(analysis_id: str, db: Session = Depends(get_db)):
    row = db.get(AnalysisResult, analysis_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    path = storage.resolve_path(row.stored_image_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Stored image file is missing.")
    return FileResponse(path)


@router.get("/analyses/{analysis_id}/heatmap")
def get_analysis_heatmap(analysis_id: str, db: Session = Depends(get_db)):
    row = db.get(AnalysisResult, analysis_id)
    if row is None or not row.heatmap_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Heatmap not available for this analysis.")
    path = storage.resolve_path(row.heatmap_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Stored heatmap file is missing.")
    return FileResponse(path)
