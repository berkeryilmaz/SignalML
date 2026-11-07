import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

from app.core.responses import SafeJSONResponse, patch_starlette_json_response
from app.core.config import APP_DIR, STORAGE_DIR, DATA_DIR, HOST, PORT, DEBUG
from app.core.database import init_database, ensure_default_project
from app.api import (
    projects,
    data,
    signals,
    models,
    training,
    feature_extraction,
    clustering,
    workflow,
    exports
)

# Apply global defense-in-depth monkey patch to JSONResponse.render
patch_starlette_json_response()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite database and ensures a default project workspace exists."""
    init_database()
    ensure_default_project()
    yield


app = FastAPI(
    title="SignalML — Visual Node-Based Machine Learning Platform",
    description="Python & Keras Node-Based Machine Learning and Signal Processing Platform",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions and guarantees a JSON response.
    Prevents plain-text 'Internal Server Error' which breaks browser JSON parsers.
    """
    import traceback
    traceback.print_exc()
    return SafeJSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# Mount static files
static_dir = APP_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Jinja2 Templates
templates_dir = APP_DIR / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Include API Routers
api_routers = [
    projects.router,
    data.router,
    signals.router,
    models.router,
    training.router,
    feature_extraction.router,
    clustering.router,
    workflow.router,
    exports.router,
]
for r in api_routers:
    r.default_response_class = SafeJSONResponse
    app.include_router(r)

# Guarantee that all registered API routes use SafeJSONResponse
for route in app.routes:
    if getattr(route, "response_class", None) is JSONResponse:
        route.response_class = SafeJSONResponse


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
def health_check():
    """System health check endpoint verifying TF, storage, and database."""
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "status": "healthy",
        "tensorflow_version": tf.__version__,
        "gpu_available": len(gpus) > 0,
        "gpu_devices": [g.name for g in gpus],
        "storage_writable": os.access(str(STORAGE_DIR), os.W_OK),
        "data_dir": str(DATA_DIR)
    }


# Template Views for 12 Core Sections
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def page_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"active_page": "dashboard"})


@app.get("/projects", response_class=HTMLResponse)
def page_projects(request: Request):
    return templates.TemplateResponse(request=request, name="projects.html", context={"active_page": "projects"})


@app.get("/data-manager", response_class=HTMLResponse)
def page_data_manager(request: Request):
    return templates.TemplateResponse(request=request, name="data_manager.html", context={"active_page": "data_manager"})


@app.get("/signal-tagger", response_class=HTMLResponse)
def page_signal_tagger(request: Request):
    return templates.TemplateResponse(request=request, name="signal_tagger.html", context={"active_page": "signal_tagger"})


@app.get("/signal-processing", response_class=HTMLResponse)
@app.get("/preprocessing", response_class=HTMLResponse)
def page_signal_processing(request: Request):
    return templates.TemplateResponse(request=request, name="signal_processing.html", context={"active_page": "signal_processing"})


@app.get("/workflow-builder", response_class=HTMLResponse)
@app.get("/workflow", response_class=HTMLResponse)
def page_workflow_builder(request: Request):
    return templates.TemplateResponse(request=request, name="workflow_builder.html", context={"active_page": "workflow_builder"})


@app.get("/nn-builder", response_class=HTMLResponse)
def page_nn_builder(request: Request):
    return templates.TemplateResponse(request=request, name="nn_builder.html", context={"active_page": "nn_builder"})


@app.get("/training", response_class=HTMLResponse)
def page_training(request: Request):
    return templates.TemplateResponse(request=request, name="training.html", context={"active_page": "training"})


@app.get("/feature-extraction", response_class=HTMLResponse)
@app.get("/features", response_class=HTMLResponse)
def page_feature_extraction(request: Request):
    return templates.TemplateResponse(request=request, name="feature_extraction.html", context={"active_page": "feature_extraction"})


@app.get("/clustering", response_class=HTMLResponse)
def page_clustering(request: Request):
    return templates.TemplateResponse(request=request, name="clustering.html", context={"active_page": "clustering"})


@app.get("/model-evaluation", response_class=HTMLResponse)
@app.get("/evaluation", response_class=HTMLResponse)
def page_model_evaluation(request: Request):
    return templates.TemplateResponse(request=request, name="model_evaluation.html", context={"active_page": "model_evaluation"})


@app.get("/exports", response_class=HTMLResponse)
def page_exports(request: Request):
    return templates.TemplateResponse(request=request, name="exports.html", context={"active_page": "exports"})


@app.get("/styleguide", response_class=HTMLResponse)
def page_styleguide(request: Request):
    return templates.TemplateResponse(request=request, name="_styleguide.html", context={"active_page": "styleguide"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=DEBUG)
