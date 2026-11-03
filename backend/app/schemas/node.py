from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class PortType(str, Enum):
    SIGNAL = "Signal"
    NUMERIC_ARRAY = "Numeric Array"
    DATASET = "Dataset"
    TENSOR = "Tensor"
    MODEL = "Model"
    FEATURE_MATRIX = "Feature Matrix"
    LABEL_ARRAY = "Label Array"
    CLUSTER_RESULTS = "Cluster Results"
    PCA_COORDINATES = "PCA Coordinates"
    ANY = "Any"


class NodePort(BaseModel):
    name: str
    type: str = "Any"
    label: Optional[str] = None


class NodeDefinition(BaseModel):
    id: Union[str, int]
    type: str
    title: Optional[str] = None
    pos: Optional[List[float]] = None
    size: Optional[List[float]] = None
    inputs: List[NodePort] = Field(default_factory=list)
    outputs: List[NodePort] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    widgets_values: Optional[List[Any]] = None


class GraphLink(BaseModel):
    id: Optional[Union[str, int]] = None
    origin_id: Union[str, int]
    origin_slot: int
    target_id: Union[str, int]
    target_slot: int
    type: Optional[str] = None


class NodeGraph(BaseModel):
    nodes: List[NodeDefinition] = Field(default_factory=list)
    links: List[Union[GraphLink, List[Any]]] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    version: float = 1.0


class GraphValidationResult(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    execution_order: List[Union[str, int]] = Field(default_factory=list)
    layers_summary: Optional[List[Dict[str, Any]]] = None
    total_params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
