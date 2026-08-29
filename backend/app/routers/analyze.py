from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import AnalysisOut
from app.services.image_validation import InvalidImageError
from app.services.pipeline import run_analysis, to_analysis_out_dict

router = APIRouter(prefix="/api", tags=["analyze"])

MAX_BATCH_FILES = 20


@router.post("/analyze", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
async def analyze_image(file: UploadFile, db: Session = Depends(get_db)):
    if not file or not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No file was uploaded.")

    raw_bytes = await file.read()
    try:
        row = run_analysis(raw_bytes, file.filename, file.content_type, db)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return to_analysis_out_dict(row)


@router.post("/analyze/batch")
async def analyze_batch(files: list[UploadFile], db: Session = Depends(get_db)):
    """Bonus: batch analysis. Each file is processed and persisted
    independently; a single bad file returns its own error entry instead of
    failing the whole batch."""
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No files were uploaded.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                             detail=f"Batch limited to {MAX_BATCH_FILES} files per request.")

    results = []
    for f in files:
        raw_bytes = await f.read()
        try:
            row = run_analysis(raw_bytes, f.filename or "upload", f.content_type, db)
            results.append(to_analysis_out_dict(row))
        except InvalidImageError as exc:
            results.append({"error": str(exc), "original_filename": f.filename})
    return results
