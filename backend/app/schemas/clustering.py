from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FeatureExtractionRequest(BaseModel):
    model_id: str
    dataset_id: str
    layer_name: str
    sample_limit: Optional[int] = None  # None = full dataset
    batch_size: int = Field(default=32, ge=1)


class ClusteringRequest(BaseModel):
    model_config = {"extra": "allow"}

    source_type: str  # "dataset", "features", "processed"
    source_id: str
    algorithm: str = Field(default="kmeans")  # "kmeans", "dbscan", "hdbscan", "optics", "idec"
    
    # KMeans / General Clustering parameters
    n_clusters: int = Field(default=3, ge=2)
    init: str = Field(default="k-means++")
    n_init: int = Field(default=10, ge=1)
    
    # DBSCAN / Density parameters
    eps: float = Field(default=0.5, gt=0.0)
    min_samples: Optional[int] = Field(default=5, ge=1)
    metric: str = Field(default="euclidean")
    
    # HDBSCAN parameters
    min_cluster_size: int = Field(default=5, ge=2)
    cluster_selection_epsilon: float = Field(default=0.0, ge=0.0)
    cluster_selection_method: str = Field(default="eom")
    
    # OPTICS parameters
    max_eps: Optional[float] = None
    xi: float = Field(default=0.05, gt=0.0, lt=1.0)

    # IDEC parameters
    model_id: Optional[str] = None
    epochs: int = Field(default=15, ge=0)
    pretrain_epochs: int = Field(default=5, ge=0)
    gamma: float = Field(default=0.1, ge=0.0)
    update_interval: int = Field(default=1, ge=1)
    
    # Preprocessing
    standardize: bool = Field(default=True)
    random_state: int = Field(default=42)
    
    # PCA
    compute_pca: bool = Field(default=True)
