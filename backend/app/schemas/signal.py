from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class SignalWaveformRequest(BaseModel):
    dataset_id: str
    sample_indices: List[int] = Field(default_factory=lambda: [0])
    max_points: int = Field(default=2000)  # Downsample if signal is very long


class PipelineStep(BaseModel):
    name: str  # e.g. "to_fixed_length", "normalize_zscore", "peak_asymmetric_window"
    params: Dict[str, Any] = Field(default_factory=dict)


class SignalProcessRequest(BaseModel):
    dataset_id: str
    pipeline: List[PipelineStep]
    save_as_new_dataset: bool = False
    new_dataset_name: Optional[str] = None


class SignalExportRequest(BaseModel):
    dataset_id: str
    target_length: Optional[int] = None
    output_format: str = "csv"  # csv, txt, npy, npz
    separate_labels: bool = False
    pipeline: Optional[List[PipelineStep]] = None
