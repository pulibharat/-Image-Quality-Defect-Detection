from datetime import datetime

from pydantic import BaseModel, ConfigDict


class IssueOut(BaseModel):
    type: str
    severity: str
    confidence: float
    confidence_source: str
    explanation: str


class AnalysisOut(BaseModel):
    id: str
    created_at: datetime
    original_filename: str
    width: int
    height: int
    file_size_bytes: int

    quality_score: float
    quality_label: str
    issues: list[IssueOut]
    features: dict[str, float]
    anomaly_score: float | None
    anomaly_threshold: float | None

    processing_time_ms: float
    model_version: str

    image_url: str
    heatmap_url: str | None = None

    model_config = ConfigDict(from_attributes=True)


class AnalysisSummary(BaseModel):
    id: str
    created_at: datetime
    original_filename: str
    quality_score: float
    quality_label: str
    issue_count: int
    image_url: str

    model_config = ConfigDict(from_attributes=True)


class AnalysisListOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AnalysisSummary]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    app_version: str
    model_version: str


class ErrorOut(BaseModel):
    detail: str
