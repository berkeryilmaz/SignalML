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
        this.canvas.background_color = "#10141d";
        this.canvas.grid_color = "#1a2233";

        this.registerWorkflowNodes();
        this.setupDefaultGraph();
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
                canvasEl.width = canvasEl.parentElement.clientWidth;
                canvasEl.height = canvasEl.parentElement.clientHeight;
                this.canvas.resize();
            }
        });
    }

    updateInspector(node) {
        const body = document.getElementById("wf-inspector-body");
        if (!body) return;

        let html = `
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 16px; font-weight: 700; color: #fff;">${node.title}</div>
                <div style="font-size: 12px; color: var(--text-muted); font-family: var(--font-mono);">Type: ${node.type} | ID: ${node.id}</div>
            </div>
            <hr style="border: none; border-top: 1px solid var(--border-color); margin-bottom: 1rem;">
        `;

        if (node.properties && Object.keys(node.properties).length > 0) {
            html += `<div style="font-weight: 600; font-size: 13px; margin-bottom: 0.5rem;">Node Configuration</div>`;
            
            if (node.type === "normalize") {
                const curMethod = String(node.properties.method || "zscore").toLowerCase();
                const curPerSig = String(node.properties.per_signal) !== "false";
                html += `
                    <div class="form-group">
                        <label class="form-label">Normalization Method</label>
                        <select class="form-control wf-prop-input" data-prop="method">
                            <option value="zscore" ${curMethod === 'zscore' ? 'selected' : ''}>Z-Score Standardization ((x - μ) / σ)</option>
                            <option value="minmax" ${curMethod === 'minmax' ? 'selected' : ''}>Min-Max Scaling ([0, 1])</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Scope / Normalizasyon Kapsamı</label>
                        <select class="form-control wf-prop-input" data-prop="per_signal">
                            <option value="true" ${curPerSig ? 'selected' : ''}>Per-Waveform / Kendi İçinde (Her sinyal bağımsız ölçeklenir)</option>
                            <option value="false" ${!curPerSig ? 'selected' : ''}>Global Dataset (Tüm veri setine göre)</option>
                        </select>
                    </div>
                    <div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid var(--accent-cyan); padding: 8px 10px; font-size: 11px; color: var(--text-secondary); border-radius: 4px; margin-bottom: 12px;">
                        💡 <strong>Kendi İçinde Normalizasyon (per_signal):</strong> Her dalga formu bağımsız olarak kendi ortalaması/aralığına göre dönüştürülür. Genlik farklılıklarını gidermek için önerilir.
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
                                    <option value="true" ${isTrue ? 'selected' : ''}>True (Aktif)</option>
                                    <option value="false" ${!isTrue ? 'selected' : ''}>False (Devre Dışı)</option>
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
            html += `<div style="color: var(--text-muted); font-size: 13px;">No configurable parameters for this node.</div>`;
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
            body.innerHTML = `
                <div class="empty-inspector">
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
            <div style="margin-top: 1rem; padding: 1rem; background: var(--bg-secondary); border-radius: 8px; border: 1px solid var(--border-color);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <strong>DAG Execution Log</strong>
                    <span class="status-badge success">Success (${data.total_duration_seconds}s)</span>
                </div>
                <div style="max-height: 250px; overflow-y: auto;">
        `;

        for (let [nid, res] of Object.entries(data.node_results)) {
            const badgeClass = res.status === "success" ? "success" : "error";
            html += `
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <div>
                        <strong>${res.node_type}</strong> (ID: ${res.node_id})
                        <div style="color: var(--text-muted); font-size: 11px;">${res.preview || ""}</div>
                    </div>
                    <div>
                        <span style="font-family: var(--font-mono); font-size: 11px; margin-right: 8px;">${res.duration_ms}ms</span>
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
