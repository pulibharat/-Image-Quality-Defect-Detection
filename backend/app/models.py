import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    original_filename: Mapped[str] = mapped_column(String(255))
    stored_image_path: Mapped[str] = mapped_column(String(500))
    heatmap_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    file_size_bytes: Mapped[int] = mapped_column(Integer)

    quality_score: Mapped[float] = mapped_column(Float)
    quality_label: Mapped[str] = mapped_column(String(20))
    issues_json: Mapped[str] = mapped_column(Text)
    features_json: Mapped[str] = mapped_column(Text)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    processing_time_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
