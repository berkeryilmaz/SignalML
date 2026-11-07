// General Purpose Workflow Visual Node Editor (LiteGraph integration for full DAG workflows)

class WorkflowEditor {
    constructor(canvasId) {
        this.canvasId = canvasId;
        this.graph = new LGraph();
        this.canvas = null;
        this.selectedNode = null;
        this.init();
    }

    init() {
        const canvasElement = document.getElementById(this.canvasId);
        if (!canvasElement) return;

        canvasElement.width = canvasElement.parentElement.clientWidth;
        canvasElement.height = canvasElement.parentElement.clientHeight;

        this.canvas = new LGraphCanvas(canvasElement, this.graph);
        if (window.SignalML && typeof window.SignalML.applyGraphTheme === "function") {
            window.SignalML.applyGraphTheme(this.canvas);
        }

        this.registerWorkflowNodes();
        this.setupDefaultGraph();
        if (window.SignalML && typeof window.SignalML.applyGraphTheme === "function") {
            window.SignalML.applyGraphTheme(this.canvas);
        }
        window.addEventListener("signalml:themechange", () => {
            if (window.SignalML && typeof window.SignalML.applyGraphTheme === "function") {
                window.SignalML.applyGraphTheme(this.canvas);
            }
        });
        this.setupEventHandlers();

        this.graph.start();
    }

    registerWorkflowNodes() {
        const registerWfNode = (type, title, inputs, outputs, properties) => {
            function WfNode() {
                for (let inp of inputs) {
                    this.addInput(inp.name, inp.type || "Any");
                }
                for (let out of outputs) {
                    this.addOutput(out.name, out.type || "Any");
                }
                this.properties = { ...properties };
                this.title = title;
                this.size = [200, 60 + Object.keys(properties).length * 20];
            }
            WfNode.title = title;
            LiteGraph.registerNodeType(type, WfNode);
        };

        // 1. Data Source
        registerWfNode("data_source", "Data Source", [], [{ name: "dataset", type: "Dataset" }], { dataset_id: "" });

        // 2. CSV / TXT Parser
        registerWfNode("csv_parser", "CSV / TXT Parser", [{ name: "raw_data", type: "Dataset" }], [{ name: "signals", type: "Signal" }, { name: "labels", type: "Label Array" }], { delimiter: ",", has_header: false });

        // 3. Preprocessing
        registerWfNode("fixed_length", "To Fixed Length", [{ name: "signals", type: "Signal" }], [{ name: "signals", type: "Signal" }], { target_length: 512, pad_mode: "constant", truncate_pos: "end" });
        registerWfNode("signal_truncate", "Signal Truncate", [{ name: "signals", type: "Signal" }], [{ name: "signals", type: "Signal" }], { target_length: 512, position: "end" });
        registerWfNode("signal_pad", "Signal Pad", [{ name: "signals", type: "Signal" }], [{ name: "signals", type: "Signal" }], { target_length: 1024, mode: "constant", constant_value: 0.0 });
        registerWfNode("peak_window", "Peak Window", [{ name: "signals", type: "Signal" }], [{ name: "signals", type: "Signal" }], { left_m: 100, right_k: 200, mode: "max_abs" });
        registerWfNode("normalize", "Normalize", [{ name: "signals", type: "Signal" }], [{ name: "signals", type: "Signal" }], { method: "zscore", per_signal: true });

        // 4. Feature Extractor
        registerWfNode("feature_extractor", "Feature Extractor", [{ name: "model", type: "Model" }, { name: "signals", type: "Signal" }], [{ name: "features", type: "Feature Matrix" }], { layer_name: "", batch_size: 32 });

        // 5. Clustering & PCA
        registerWfNode("clustering_kmeans", "K-Means Clustering", [{ name: "features", type: "Feature Matrix" }], [{ name: "cluster_results", type: "Cluster Results" }], { n_clusters: 3 });
        registerWfNode("clustering_dbscan", "DBSCAN Clustering", [{ name: "features", type: "Feature Matrix" }], [{ name: "cluster_results", type: "Cluster Results" }], { eps: 0.5, min_samples: 5 });
        registerWfNode("clustering_idec", "IDEC Deep Clustering", [{ name: "features", type: "Feature Matrix" }], [{ name: "cluster_results", type: "Cluster Results" }], { n_clusters: 5, epochs: 15, pretrain_epochs: 5, gamma: 0.1 });
        registerWfNode("pca_analysis", "PCA Analysis", [{ name: "features", type: "Feature Matrix" }], [{ name: "pca_coords", type: "PCA Coordinates" }], { n_components: 3 });

        // 6. Signal Export
        registerWfNode("signal_export", "Signal Export", [{ name: "signals", type: "Signal" }, { name: "labels", type: "Label Array" }], [], { target_length: 512, output_format: "csv", filename: "exported_signals" });
    }

    setupDefaultGraph() {
        const nDataSource = LiteGraph.createNode("data_source");
        nDataSource.pos = [60, 160];
        this.graph.add(nDataSource);

        const nParser = LiteGraph.createNode("csv_parser");
        nParser.pos = [280, 160];
        this.graph.add(nParser);

        const nFixedLen = LiteGraph.createNode("fixed_length");
        nFixedLen.pos = [520, 160];
        nFixedLen.properties.target_length = 512;
        this.graph.add(nFixedLen);

        const nCluster = LiteGraph.createNode("clustering_kmeans");
        nCluster.pos = [760, 160];
        nCluster.properties.n_clusters = 3;
        this.graph.add(nCluster);

        const nPCA = LiteGraph.createNode("pca_analysis");
        nPCA.pos = [1000, 160];
        this.graph.add(nPCA);

        // Connections
        nDataSource.connect(0, nParser, 0);
        nParser.connect(0, nFixedLen, 0);
        nFixedLen.connect(0, nCluster, 0);
        nFixedLen.connect(0, nPCA, 0);
    }

    setupEventHandlers() {
        this.canvas.onNodeSelected = (node) => {
            this.selectedNode = node;
            this.updateInspector(node);
        };

        this.canvas.onNodeDeselected = () => {
            this.selectedNode = null;
            this.clearInspector();
        };

        window.addEventListener("resize", () => {
            const canvasEl = document.getElementById(this.canvasId);
            if (canvasEl && canvasEl.parentElement) {
                this.canvas.resize(canvasEl.parentElement.clientWidth, canvasEl.parentElement.clientHeight);
                this.canvas.draw(true, true);
            }
        });
    }

    updateInspector(node) {
        const body = document.getElementById("wf-inspector-body");
        if (!body) return;

        let html = `
            <div class="mb-3">
                <div class="text-subhead">${node.title}</div>
                <div class="text-xs text-muted mono">Type: ${node.type} | ID: ${node.id}</div>
            </div>
            <hr class="divider mb-3">
        `;

        if (node.properties && Object.keys(node.properties).length > 0) {
            html += `<div class="section-title">Node Configuration</div>`;
            
            if (node.type === "normalize") {
                const curMethod = String(node.properties.method || "zscore").toLowerCase();
                const curPerSig = String(node.properties.per_signal) !== "false";
                html += `
                    <div class="form-group">
                        <label class="form-label">Normalization Method</label>
                        <select class="form-control wf-prop-input" data-prop="method">
                            <option value="zscore" ${curMethod === 'zscore' ? 'selected' : ''}>Z-Score Standardization ((x - mu) / sigma)</option>
                            <option value="minmax" ${curMethod === 'minmax' ? 'selected' : ''}>Min-Max Scaling ([0, 1])</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Normalization scope</label>
                        <select class="form-control wf-prop-input" data-prop="per_signal">
                            <option value="true" ${curPerSig ? 'selected' : ''}>Per-waveform (scale each signal independently)</option>
                            <option value="false" ${!curPerSig ? 'selected' : ''}>Global dataset (scale across all samples)</option>
                        </select>
                    </div>
                    <div class="callout callout-accent mb-3 text-sm">
                        Per-waveform normalization scales each signal by its own mean and range to remove amplitude offsets.
                    </div>
                `;
            } else {
                for (let [k, v] of Object.entries(node.properties)) {
                    if (typeof v === "boolean" || v === "true" || v === "false") {
                        const isTrue = String(v) === "true";
                        html += `
                            <div class="form-group">
                                <label class="form-label">${k}</label>
                                <select class="form-control wf-prop-input" data-prop="${k}">
                                    <option value="true" ${isTrue ? 'selected' : ''}>True (Enabled)</option>
                                    <option value="false" ${!isTrue ? 'selected' : ''}>False (Disabled)</option>
                                </select>
                            </div>
                        `;
                    } else {
                        html += `
                            <div class="form-group">
                                <label class="form-label">${k}</label>
                                <input class="form-control wf-prop-input" data-prop="${k}" type="${typeof v === 'number' ? 'number' : 'text'}" value="${v}">
                            </div>
                        `;
                    }
                }
            }
        } else {
            html += `<div class="text-muted text-sm">No configurable parameters for this node.</div>`;
        }

        body.innerHTML = html;

        body.querySelectorAll(".wf-prop-input").forEach(input => {
            const updateHandler = (e) => {
                const prop = e.target.dataset.prop;
                let val = e.target.value;
                if (e.target.type === "number") val = Number(val);
                if (val === "true") val = true;
                if (val === "false") val = false;
                node.properties[prop] = val;
            };
            input.addEventListener("input", updateHandler);
            input.addEventListener("change", updateHandler);
        });
    }

    clearInspector() {
        const body = document.getElementById("wf-inspector-body");
        if (body) {
            const iconHtml = (window.SignalML && typeof window.SignalML.icon === "function") ? window.SignalML.icon("sliders-horizontal") : "";
            body.innerHTML = `
                <div class="empty-inspector">
                    ${iconHtml}
                    <p>Select any node to view configuration parameters and live execution status.</p>
                </div>
            `;
        }
    }

    serializeGraph() {
        return this.graph.serialize();
    }

    async validateWorkflow() {
        const payload = { graph: this.serializeGraph() };
        try {
            const res = await fetch("/api/workflow/validate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await SignalML.safeJson(res);
            if (data.valid) {
                SignalML.toast("Workflow DAG is valid and acyclic!", "success");
            } else {
                const errMsg = Array.isArray(data.errors) ? data.errors.join("; ") : (data.detail || "Validation Error");
                SignalML.toast(`Workflow Validation Error: ${errMsg}`, "error");
            }
        } catch (e) {
            SignalML.toast(`Network Error: ${e.message}`, "error");
        }
    }

    async executeWorkflow() {
        const graphData = this.serializeGraph();
        const activeProjectId = (window.SignalML && typeof window.SignalML.getProjectId === "function" ? window.SignalML.getProjectId() : (window.SignalML && window.SignalML.currentProjectId))
            || (typeof SignalML !== "undefined" && (typeof SignalML.getProjectId === "function" ? SignalML.getProjectId() : SignalML.currentProjectId))
            || localStorage.getItem("signalml_active_project")
            || "default-project";
        const payload = {
            project_id: activeProjectId,
            name: "Automated Signal Workflow",
            graph: graphData
        };

        SignalML.toast("Executing workflow DAG on backend...", "info");

        try {
            const res = await fetch("/api/workflow/execute", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await SignalML.safeJson(res);
                SignalML.toast(`Workflow executed in ${data.total_duration_seconds}s!`, "success");
                this.renderExecutionResults(data);
            } else {
                const err = await SignalML.safeJson(res);
                SignalML.toast(`Workflow Failed: ${err.detail || "Execution Error"}`, "error");
            }
        } catch (e) {
            SignalML.toast(`Network Error: ${e.message}`, "error");
        }
    }

    renderExecutionResults(data) {
        const container = document.getElementById("wf-results-container");
        if (!container) return;

        let html = `
            <div class="tile mt-3">
                <div class="row-between mb-2">
                    <strong class="text-sm">DAG Execution Log</strong>
                    <span class="status-badge success">Success (${data.total_duration_seconds}s)</span>
                </div>
                <div class="max-h-sm scroll-y">
        `;

        for (let [nid, res] of Object.entries(data.node_results)) {
            const badgeClass = res.status === "success" ? "success" : "error";
            html += `
                <div class="row-between text-sm py-1 border-b">
                    <div>
                        <strong>${res.node_type}</strong> (ID: <span class="mono">${res.node_id}</span>)
                        <div class="text-muted text-xs">${res.preview || ""}</div>
                    </div>
                    <div class="row">
                        <span class="mono text-xs">${res.duration_ms}ms</span>
                        <span class="status-badge ${badgeClass}">${res.status}</span>
                    </div>
                </div>
            `;
        }
        html += `</div></div>`;
        container.innerHTML = html;
    }
}

window.WorkflowEditor = WorkflowEditor;
