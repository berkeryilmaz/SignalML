from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FeatureExtractionRequest(BaseModel):
    model_id: str
    dataset_id: str
    layer_name: str
    sample_limit: Optional[int] = None  # None = full dataset
    batch_size: int = Field(default=32, ge=1)


class ClusteringRequest(BaseModel):
    source_type: str  # "dataset", "features", "processed"
    source_id: str
    algorithm: str = Field(default="kmeans")  # "kmeans", "dbscan"
    
    # KMeans parameters
    n_clusters: int = Field(default=3, ge=2)
    init: str = Field(default="k-means++")
    n_init: int = Field(default=10, ge=1)
    
    # DBSCAN parameters
    eps: float = Field(default=0.5, gt=0.0)
    min_samples: int = Field(default=5, ge=1)
    metric: str = Field(default="euclidean")
    
    # Preprocessing
    standardize: bool = Field(default=True)
    random_state: int = Field(default=42)
    
    # PCA
    compute_pca: bool = Field(default=True)
