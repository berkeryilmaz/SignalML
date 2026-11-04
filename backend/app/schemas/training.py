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
    idec_mode: bool = Field(default=False, description="Force or enable IDEC clustering training")
    n_clusters: int = Field(default=5, ge=2, le=100, description="Number of clusters for IDEC")
    gamma: float = Field(default=0.1, ge=0.001, le=10.0, description="Clustering KLD loss weight")
    pretrain_epochs: int = Field(default=5, ge=0, le=500, description="Autoencoder pre-training epochs before IDEC clustering")
    update_interval: int = Field(default=1, ge=1, le=50, description="Target distribution P update interval in epochs")


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
