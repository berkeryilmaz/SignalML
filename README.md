# SignalML — Visual Node-Based Machine Learning & Signal Processing Platform

SignalML is a browser-based, drag-and-drop **node-based** (Blender Geometry Nodes / Unreal Blueprint style) machine learning and 1D time-series signal processing platform. It compiles visual node graphs into genuine **TensorFlow / Keras Functional API** models, supports multi-channel signal workflows, asynchronous training, intermediate layer feature extraction, and deep/classical clustering with interactive 3D PCA visualization.

---

## Key Architectural Highlights

- **100% Python Backend**: FastAPI, Uvicorn, TensorFlow / Keras, NumPy, Pandas, SciPy, scikit-learn, HDBSCAN/OPTICS, and SQLite.
- **Zero Production Node.js Runtime Dependency**: No `npm`, `vite`, or `webpack` build step required to run the platform.
- **100% Offline-Ready Frontend**: LiteGraph, Highcharts, and PapaParse are bundled locally under `backend/app/static/vendor/` with local Lucide SVG sprite icons and a dual Light/Dark technical theme.
- **Genuine Keras Functional API Compilation**: Visual node graphs are validated with real-time tensor shape inference and compiled into `.keras` models, with support for multi-input/multi-output branches, dynamic `Concatenate` ports, custom Python `keras.layers.Layer` classes, and executable Python code export.
- **Signal-First Data & Preprocessing Architecture**:
  - **Data Manager (`/data-manager`)**: Focused strictly on raw signal dataset ingestion, column role assignment (`ID`, `Input`, `Output`, `Ignore`), missing-value handling, train/val/test split configuration, and multi-prefix 1D-CNN channel tensor grouping `(N, T, C)` while preserving raw signal amplitudes (no One-Hot/Label encoding or scaling clutter in Data Manager).
  - **Signal Processing (`/signal-processing`)**: Dedicated sequential signal pipeline for scaling/normalization (Z-Score, Min-Max, Min/Max Scalers), baseline subtraction (Median, Mode, Gaussian Peak), alignment, peak windowing, padding/truncation, and exact-length export.
  - **Feature Extraction (`/feature-extraction`)**: Extracts intermediate neural network layer activations for any selected dataset, automatically synchronizes with the latest trained dataset and row counts, exports activation matrices with sample IDs to CSV, and allows saving extracted features directly as new project datasets.
- **NaN / Inf Safe Serialization**: Custom `SafeJSONResponse` sanitizes floating-point `NaN` and `±Inf` values across all REST endpoints.
- **Security & Isolation**: Built-in Zip Slip (path traversal) and Zip Bomb protection, sandboxed path resolution, and project-scoped workspaces.

---

## Quickstart with Docker

### 1. Default (CPU Mode)

```bash
cd SignalML
docker compose up --build
```

- **Web UI**: [http://localhost:8080](http://localhost:8080)
- **Swagger / OpenAPI Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)

### 2. NVIDIA GPU Mode (Optional)

If NVIDIA drivers and NVIDIA Container Toolkit are installed on the host:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

### 3. Mounting Local Host Signal Data (`/data`)

Place signal files directly in `./data` or mount any host directory in `compose.yaml`:

```yaml
volumes:
  - ./storage:/app/storage
  - /path/to/local/signals:/data:ro
```

Access mounted files from **Data Manager &rarr; Mounted Host Directory (`/data`)**.

### 4. Local Development (Without Docker)

```bash
./run.sh
# or manually:
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Platform Modules (12 Core Screens)

### 1. Dashboard (`/dashboard`)
System health overview, active project telemetry, dataset/model counters, CPU/GPU runtime status, and recent training run summaries.

### 2. Projects (`/projects`)
Multi-workspace management: create, switch, duplicate, rename, and delete isolated project workspaces backed by SQLite and persistent storage directories.

### 3. Data Manager (`/data-manager`)
- **Raw Signal Ingestion**: Upload CSV, TXT, TSV, JSON, NPY, NPZ, or ZIP archives (with Zip Slip & Zip Bomb protection), or import from mounted `/data`.
- **Raw Amplitude Preservation**: Preserves raw signal values without applying premature scaling or categorical one-hot encoding in Data Manager; scaling and filtering are handled in the dedicated **Signal Processing** module.
- **Interactive Schema & Preview**: Assign column roles (`ID / Key`, `Input (X)`, `Target / Label (y)`, `Ignore / Drop`), configure missing-value strategies (`zero`, `mean`, `ffill`, `drop`), configure Train/Val/Test split ratios, transpose single-column signals, and auto-group multi-prefix sequence columns (`s*`, `a_*`, `dvdt_*`) into 3D `(N, T, C)` tensors for 1D-CNN models.
- **Consistent Row & Column Tracking**: Synchronizes `row_count`, `num_rows`, and `total_rows` across registration, updates, previews, and downstream modules.

### 4. Signal Tagger (`/signal-tagger`)
- High-resolution interactive waveform inspection (Highcharts).
- Sample navigation, multi-signal overlay comparison, zoom, and pan.

### 5. Signal Processing (`/signal-processing`)
Dedicated screen for signal scaling, filtering, and windowing with live before/after waveform comparison and group envelope statistics:
- **Length Standardization**: `to_fixed_length` (pad/truncate), `first_n`, `last_n`, `slice`.
- **Threshold Alignment & Windowing**: `align_starts`, `align_ends`, asymmetric/symmetric `peak_window`.
- **Baseline & Polarity Filters**: Median subtraction (`x - median`), Mode subtraction, Histogram Gaussian Peak baseline subtraction, and signal reversal (`-x`).
- **Scaling & Normalization**: Z-Score (`zscore`), Min-Max (`minmax`), Min Scaler (`min_scaler`), Max Scaler (`max_scaler`) with per-waveform (row-wise) or global scope.
- **Save & Export**: Save processed waveforms as a new project dataset (preserving sample IDs and labels) or export exact-$N$ sample matrices (CSV, TXT, NPY, NPZ).

### 6. Visual Workflow Builder (`/workflow-builder`)
General-purpose DAG pipeline editor (`Data Source` &rarr; `CSV Parser` &rarr; `Preprocessing` &rarr; `Feature Extractor` &rarr; `Clustering` &rarr; `PCA` &rarr; `Export`) with type-checked ports, cycle detection, topological execution, and step timing metrics.

### 7. Neural Network Builder (`/nn-builder`)
Visual Keras Functional API architecture studio:
- **Supported Layers**:
  - *Input / Output*: Configurable shape, per-node dataset binding, derived signal channels (`raw`, `derivative_1`, `derivative_2`, `region_mask`, `envelope_rms`, `rolling_mean`, `fft_magnitude`), and target source (`auto`, `reconstruction`, `region_labels`, `label_class`, `regression`, `clustering`).
  - *Core*: `Dense`, `Activation`, `Flatten`, `Reshape`, `Dropout`, `BatchNormalization`, `LayerNormalization`, `UnitNormalization`.
  - *1D & 2D Convolution*: `Conv1D`, `Conv1DTranspose`, `UpSampling1D`, `SeparableConv1D`, `Conv2D`, `MaxPooling1D/2D`, `AveragePooling1D/2D`, `GlobalMaxPooling1D`, `GlobalAveragePooling1D`.
  - *Recurrent & Attention*: `LSTM`, `GRU`, `SimpleRNN`, `Bidirectional`, `MultiHeadAttention`, `PositionalEncoding`.
  - *Merge & Clustering*: Dynamic multi-port `Concatenate`, `Add` (residual), `Multiply`, and IDEC `ClusteringLayer`.
- **Custom Keras Layers**: Write, validate, and persist custom Python `keras.layers.Layer` subclasses per project.
- **4-Way Canvas Layout & Grid Snapping**: Supports `LR`, `RL`, `TB`, and `BT` flow orientations with 20px grid snapping and one-click Python Functional API code export.

### 8. Training (`/training`)
- Asynchronous background training jobs with real-time loss/accuracy curves.
- Optimizers: `Adam`, `SGD`, `RMSprop`, `AdamW`.
- **Resume & Transfer Learning**: Resume training from checkpoints and selectively freeze/unfreeze layers (`POST /api/models/{model_id}/freeze`).
- **IDEC Deep Clustering Mode**: Jointly optimizes KL-divergence clustering loss and autoencoder reconstruction MSE loss.

### 9. Feature Extraction (`/feature-extraction`)
- Inspect all intermediate layers and output tensor shapes of any trained Keras model.
- Automatically defaults to the dataset on which the selected model was most recently trained, displaying live dataset row/column counts and clearing stale results on dataset/model switch.
- Extracts intermediate activations via `keras.Model(inputs=model.inputs, outputs=target_layer.output)` on any chosen dataset (always honoring the newly selected dataset's row count).
- **Save as Dataset & CSV Export**: Register extracted feature matrices directly as a new project dataset (`POST /api/features/{feature_id}/save_dataset`) or download as CSV with original sample IDs.

### 10. Clustering & PCA (`/clustering`)
- **Algorithms**: `K-Means`, `DBSCAN`, `HDBSCAN`, `OPTICS`, and deep `IDEC` clustering.
- **Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, cluster confidence, assignment entropy, and per-cluster waveform mean/std envelopes.
- **Interactive 3D PCA Canvas**: Hardware-accelerated 3D/2D scatter plot with orbit/zoom controls, outlier highlighting, sample tooltips, and customizable CSV export.

### 11. Model Evaluation (`/model-evaluation`)
- Confusion matrix, precision/recall/F1 classification report, regression metrics ($R^2$, MSE, MAE), historical run comparison, and batch prediction CSV export (`GET /api/training/predict_csv/{model_id}/{dataset_id}`).

### 12. Exports (`/exports`)
Centralized download and model import hub for `.keras` weights, architecture JSON, Python scripts, preprocessed signals, and extracted feature sets.

---

## Testing

The repository includes **98 Pytest unit/integration tests** and **18 Node.js visual editor tests**:

```bash
# Run full Python backend test suite
PYTHONPATH=backend pytest backend/tests -v

# Run NN Builder graph flow & snapping tests
node backend/tests/test_nn_flow_and_snap.js
```

### Test Modules (`backend/tests/`)
- `test_data_manager.py`: CSV/TXT/NPY/NPZ loading, Zip Slip/Bomb security, row-count (`row_count`/`num_rows`/`total_rows`) consistency, missing-value drop recalculation, raw signal preservation without unintended scaling, and multi-channel 3D tensor grouping.
- `test_signal_processing.py`: Fixed-length padding/truncation, threshold alignment, asymmetric peak windowing, baseline subtraction, scaling, and exact-$N$ export.
- `test_feature_extraction.py`: Submodel layer extraction, dataset switching row-count verification, saving extracted features as datasets, and full REST API lifecycle.
- `test_keras_builder.py`: Keras Functional API compilation, multi-branch shape inference, auto-flattening, custom layers, and Python code generation.
- `test_training.py`: Asynchronous training, NaN safety, multi-input hybrid training, layer freezing, checkpoint resumption, derived signal channels, and batch prediction CSV.
- `test_clustering_pca.py`: K-Means, DBSCAN, HDBSCAN, OPTICS, IDEC soft probabilities, and 2D/3D PCA projections.
- `test_node_graph.py` & `test_workflow_engine.py`: DAG topological sorting, port validation, cycle detection, and pipeline execution.
- `test_safe_json_response.py` & `test_api_integration.py`: Strict JSON `NaN`/`Inf` sanitization and end-to-end route/template rendering.

---

## Directory Structure

```text
SignalML/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI REST routers (data, signals, models, training, features, clustering, workflow, projects, exports)
│   │   ├── core/             # Config, SQLite database schema, path security, SafeJSONResponse
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── services/         # DataManager, SignalProcessing, KerasModelBuilder, TrainingService, FeatureExtractor, ClusteringService, WorkflowEngine
│   │   ├── templates/        # 12 Jinja2 HTML application views
│   │   ├── static/
│   │   │   ├── css/          # Design system & node editor stylesheets
│   │   │   ├── js/           # Frontend controllers (nn_editor.js, pca_visualizer.js, workflow_editor.js, etc.)
│   │   │   └── vendor/       # Bundled offline libraries (LiteGraph, Highcharts, PapaParse)
│   │   └── main.py           # FastAPI application entrypoint
│   ├── tests/                # Pytest & JS test suites
│   └── requirements.txt      # Python dependencies
├── data/                     # Mounted host dataset directory (/data)
├── docs/                     # Design system & UI architecture docs
├── storage/                  # Persistent SQLite database, datasets, models, features, clusters, and exports
├── Dockerfile                # Single-container Python 3.12 image
├── compose.yaml              # Docker Compose configuration
├── compose.gpu.yaml          # NVIDIA GPU override configuration
├── run.sh                    # Local startup script
└── README.md
```
