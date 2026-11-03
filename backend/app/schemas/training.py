from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    model_id: str
    dataset_id: str
    epochs: int = Field(default=20, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=2048)
    learning_rate: float = Field(default=0.001, gt=0.0)
    optimizer: str = Field(default="adam")  # adam, sgd, rmsprop, adamw
    loss: Optional[str] = None  # auto-detected if None
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"])
    val_split: float = Field(default=0.2, ge=0.0, le=0.5)
    shuffle: bool = Field(default=True)
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=5, ge=1)
    seed: Optional[int] = Field(default=42)
    resume: bool = Field(default=False)
    frozen_layers: Optional[List[str]] = None


class TrainingJobStatus(BaseModel):
    job_id: str
    model_id: str
    dataset_id: str
    status: str  # pending, running, completed, stopped, failed
    current_epoch: int = 0
    total_epochs: int = 0
    history: Dict[str, List[float]] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
