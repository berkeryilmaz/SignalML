from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(default="")


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    active_dataset_id: Optional[str] = None
    active_model_id: Optional[str] = None
    active_workflow_id: Optional[str] = None
    metadata_json: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    active_dataset_id: Optional[str] = None
    active_model_id: Optional[str] = None
    active_workflow_id: Optional[str] = None
    metadata_json: Optional[str] = "{}"
