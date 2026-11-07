// Neural Network Visual Node Editor (LiteGraph integration for Keras Functional API)

class NNEditor {
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

        // Resize canvas to container
        canvasElement.width = canvasElement.parentElement.clientWidth;
        canvasElement.height = canvasElement.parentElement.clientHeight;

        this.canvas = new LGraphCanvas(canvasElement, this.graph);
        if (window.SignalML && typeof window.SignalML.applyGraphTheme === "function") {
            window.SignalML.applyGraphTheme(this.canvas);
        }

        this.registerKerasNodes();
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

    toast(message, type = "info") {
        const sml = (typeof window !== "undefined" && window.SignalML) || (typeof SignalML !== "undefined" ? SignalML : null);
        if (sml && typeof sml.toast === "function") {
            sml.toast(message, type);
        } else {
            console.log(`[${type}] ${message}`);
        }
    }

    getActiveProjectId() {
        const sml = (typeof window !== "undefined" && window.SignalML) || (typeof SignalML !== "undefined" ? SignalML : null);
        if (sml) {
            if (typeof sml.getProjectId === "function") return sml.getProjectId();
            if (sml.currentProjectId) return sml.currentProjectId;
        }
        if (typeof localStorage !== "undefined") {
            const saved = localStorage.getItem("signalml_active_project");
            if (saved) return saved;
        }
        return "default-project";
    }

    registerKerasNodes() {
        const registerLayer = (type, title, inputs, outputs, properties, aliases = []) => {
            function KerasNode() {
                for (let inp of inputs) {
                    this.addInput(inp.name, inp.type || "Tensor");
                }
                for (let out of outputs) {
                    this.addOutput(out.name, out.type || "Tensor");
                }
                this.properties = { ...properties };
                this.title = title;
                this.size = [180, 50 + Object.keys(properties).length * 20];
            }
            KerasNode.title = title;
            KerasNode.prototype.onPropertyChanged = function() {
                if (window.nnEditorInstance) {
                    window.nnEditorInstance.updateInspector(this);
                }
            };
            LiteGraph.registerNodeType(`keras/${type}`, KerasNode);
            LiteGraph.registerNodeType(type, KerasNode);
            for (let alias of aliases) {
                LiteGraph.registerNodeType(`keras/${alias}`, KerasNode);
                LiteGraph.registerNodeType(alias, KerasNode);
            }
        };

        // 1. Input & Output
        registerLayer("input", "Input", [], [{ name: "tensor", type: "Tensor" }], { shape: "512, 1" }, ["inputlayer"]);
        registerLayer("output", "Model Output", [{ name: "in", type: "Tensor" }], [], { activation: "linear" }, ["modeloutput"]);

        // 2. Dense
        registerLayer("dense", "Dense", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, activation: "relu" });

        // 3. CNN (1D & Transposed Deconvolution)
        registerLayer("conv1d", "Conv1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 5, strides: 1, padding: "same", activation: "relu" });
        registerLayer("conv1dtranspose", "Conv1DTranspose", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 5, strides: 2, padding: "same", activation: "relu" }, ["deconv1d", "conv1d_transpose"]);
        registerLayer("deconv1d", "Deconv1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 5, strides: 2, padding: "same", activation: "relu" }, ["conv1dtranspose"]);
        registerLayer("upsampling1d", "UpSampling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { size: 2 });
        registerLayer("separable_conv1d", "SeparableConv1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 5, padding: "same", activation: "relu" });
        registerLayer("conv2d", "Conv2D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 3, padding: "same", activation: "relu" });
        registerLayer("maxpooling1d", "MaxPooling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { pool_size: 2, padding: "same" }, ["max_pooling1d", "maxpool1d"]);
        registerLayer("averagepooling1d", "AveragePooling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { pool_size: 2, padding: "same" }, ["avgpooling1d", "avg_pooling1d", "avgpool1d"]);
        registerLayer("global_maxpooling1d", "GlobalMaxPooling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["globalmaxpooling1d"]);
        registerLayer("global_averagepooling1d", "GlobalAveragePooling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["globalaveragepooling1d", "globalavgpooling1d"]);

        // 4. Recurrent & TimeDistributed
        registerLayer("lstm", "LSTM", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, return_sequences: false, dropout: 0.0 });
        registerLayer("gru", "GRU", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, return_sequences: false });
        registerLayer("simplernn", "SimpleRNN", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, return_sequences: false, activation: "tanh" });
        registerLayer("bidirectional", "Bidirectional", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, recurrent_type: "lstm", return_sequences: false });
        registerLayer("timedistributed", "TimeDistributed", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { layer_type: "Dense", units: 32 }, ["time_distributed"]);

        // 5. Attention & Transformer
        registerLayer("multihead_attention", "MultiHeadAttention", [{ name: "query", type: "Tensor" }, { name: "value", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { num_heads: 4, key_dim: 32, dropout: 0.1 }, ["multiheadattention"]);
        registerLayer("positional_encoding", "PositionalEncoding", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { d_model: 64, max_len: 2048 }, ["positionalencoding"]);

        // 6. Reshape & Regularization
        registerLayer("flatten", "Flatten", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {});
        registerLayer("reshape", "Reshape", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { target_shape: "-1, 1" });
        registerLayer("dropout", "Dropout", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { rate: 0.2 });
        registerLayer("batchnorm", "BatchNormalization", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["batchnormalization", "bn"]);
        registerLayer("layernorm", "LayerNormalization", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["layernormalization", "ln"]);
        registerLayer("unitnorm", "UnitNormalization", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["unitnormalization"]);

        // 7. Merging
        registerLayer("concatenate", "Concatenate", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { axis: -1 });
        registerLayer("add", "Add", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {});
        registerLayer("multiply", "Multiply", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {});
        registerLayer("average", "Average", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {});
        registerLayer("dot", "Dot", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { axes: -1, normalize: false });
        registerLayer("activation", "Activation", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { activation: "relu" });

        // 8. IDEC Clustering & Custom Layers
        registerLayer("clustering", "ClusteringLayer (IDEC)", [{ name: "in", type: "Tensor" }], [{ name: "prob", type: "Tensor" }], { n_clusters: 8, alpha: 1.0 }, ["clusteringlayer"]);
        registerLayer("custom_layer", "Custom Layer", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { class_name: "CustomSignalLayer", units: 32, activation: "relu", code: "" }, ["customlayer", "customsignallayer"]);
    }

    setupDefaultGraph() {
        // Create an example deep learning pipeline: Input -> Conv1D -> LSTM -> Dense
        const nodeInput = LiteGraph.createNode("keras/input");
        nodeInput.pos = [100, 180];
        nodeInput.properties.shape = "512, 1";
        this.graph.add(nodeInput);

        const nodeConv = LiteGraph.createNode("keras/conv1d");
        nodeConv.pos = [340, 180];
        nodeConv.properties.filters = 32;
        nodeConv.properties.kernel_size = 5;
        this.graph.add(nodeConv);

        const nodeLSTM = LiteGraph.createNode("keras/lstm");
        nodeLSTM.pos = [580, 180];
        nodeLSTM.properties.units = 64;
        nodeLSTM.properties.return_sequences = false;
        this.graph.add(nodeLSTM);

        const nodeDense = LiteGraph.createNode("keras/dense");
        nodeDense.pos = [820, 180];
        nodeDense.properties.units = 6;
        nodeDense.properties.activation = "softmax";
        this.graph.add(nodeDense);

        // Connect
        nodeInput.connect(0, nodeConv, 0);
        nodeConv.connect(0, nodeLSTM, 0);
        nodeLSTM.connect(0, nodeDense, 0);
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

        // Window resize
        window.addEventListener("resize", () => {
            const canvasEl = document.getElementById(this.canvasId);
            if (canvasEl && canvasEl.parentElement) {
                this.canvas.resize(canvasEl.parentElement.clientWidth, canvasEl.parentElement.clientHeight);
                this.canvas.draw(true, true);
            }
        });

        // Add Model Wizard to Canvas Right-Click Context Menu
        const origGetMenuOptions = this.canvas.getCanvasMenuOptions;
        this.canvas.getCanvasMenuOptions = () => {
            const opts = origGetMenuOptions ? origGetMenuOptions.call(this.canvas) : [];
            const wizardItem = {
                content: "Architecture wizard...",
                callback: () => {
                    if (typeof window.openWizardModal === "function") {
                        window.openWizardModal();
                    }
                }
            };
            return [wizardItem, null, ...(opts || [])];
        };
    }

    updateInspector(node) {
        const body = document.getElementById("nn-inspector-body");
        if (!body) return;

        const currentCustomName = (node.properties && node.properties.custom_name) !== undefined ? node.properties.custom_name : (node.title || "");

        const sml = (typeof window !== "undefined" && window.SignalML) || (typeof SignalML !== "undefined" ? SignalML : null);
        const safeCustomName = (sml && typeof sml.escapeHtml === "function") ? sml.escapeHtml(currentCustomName) : currentCustomName;
        const iconFn = (sml && typeof sml.icon === "function") ? sml.icon.bind(sml) : (() => "");

        let shapeBadgeHtml = "";
        if (node._inferredShape !== undefined || node._paramCount !== undefined) {
            const shapeStr = Array.isArray(node._inferredShape)
                ? "(" + node._inferredShape.map(d => (d === null || d === -1) ? "?" : d).join(", ") + ")"
                : (node._inferredShape !== undefined ? JSON.stringify(node._inferredShape) : "N/A");
            const paramStr = node._paramCount !== undefined ? Number(node._paramCount).toLocaleString() : "0";
            shapeBadgeHtml = `
                <div class="tile mb-3">
                    <div class="row text-micro text-success mb-2">${iconFn("layers", "icon-xs")} <span>Output shape &amp; parameters</span></div>
                    <div class="row-between text-sm mb-1">
                        <span class="text-muted">Output Shape:</span>
                        <span class="shape-badge">${shapeStr}</span>
                    </div>
                    <div class="row-between text-sm">
                        <span class="text-muted">Parameters:</span>
                        <span class="badge badge-mono badge-success">${paramStr}</span>
                    </div>
                </div>
            `;
        }

        let html = `
            <div class="mb-3">
                <div class="text-subhead">${node.title}</div>
                <div class="text-xs text-muted mono">Type: ${node.type} | ID: ${node.id}</div>
            </div>
            <div class="form-group tile mb-3">
                <label class="form-label">Layer name</label>
                <input class="form-control mono" id="inspector-custom-name" type="text" value="${safeCustomName}" placeholder="e.g. encoder_bottleneck">
            </div>
            ${shapeBadgeHtml}
            <hr class="divider mb-3">
        `;

        if (node.properties && Object.keys(node.properties).length > 0) {
            html += `<div class="section-title">Parameters</div>`;
            for (let [k, v] of Object.entries(node.properties)) {
                if (k === "custom_name") continue;
                if (k === "code") {
                    const safeCode = (sml && typeof sml.escapeHtml === "function") ? sml.escapeHtml(v || "") : (v || "");
                    html += `
                        <div class="form-group">
                            <div class="row-between mb-1">
                                <label class="form-label m-0">Layer code</label>
                                <button class="btn btn-secondary btn-xs" onclick="openCustomLayerModal()">Open Studio</button>
                            </div>
                            <textarea class="form-control prop-input mono text-xs" data-prop="code" rows="5">${safeCode}</textarea>
                        </div>
                    `;
                } else if (typeof v === "boolean") {
                    html += `
                        <div class="form-group">
                            <label class="form-label">${k}</label>
                            <select class="form-control prop-input-bool" data-prop="${k}">
                                <option value="true" ${v === true ? "selected" : ""}>true</option>
                                <option value="false" ${v === false ? "selected" : ""}>false</option>
                            </select>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="form-group">
                            <label class="form-label">${k}</label>
                            <input class="form-control prop-input" data-prop="${k}" type="${typeof v === 'number' ? 'number' : 'text'}" value="${v !== undefined && v !== null ? v : ''}">
                        </div>
                    `;
                }
            }
        } else {
            html += `<div class="text-muted text-sm">No configurable parameters for this layer.</div>`;
        }

        body.innerHTML = html;

        // Bind custom name input event
        const nameInput = document.getElementById("inspector-custom-name");
        if (nameInput) {
            nameInput.addEventListener("input", (e) => {
                const val = e.target.value;
                node.properties.custom_name = val.trim();
                node.title = val.trim() || node.type;
                if (this.canvas) {
                    this.canvas.setDirty(true, true);
                }
            });
        }

        // Bind boolean select change events
        body.querySelectorAll(".prop-input-bool").forEach(sel => {
            sel.addEventListener("change", (e) => {
                const prop = e.target.dataset.prop;
                node.properties[prop] = (e.target.value === "true");
                if (this.canvas) {
                    this.canvas.setDirty(true, true);
                }
            });
        });

        // Bind generic input events
        body.querySelectorAll(".prop-input").forEach(input => {
            input.addEventListener("input", (e) => {
                const prop = e.target.dataset.prop;
                let val = e.target.value;
                if (e.target.type === "number") val = val === "" ? 0 : Number(val);
                node.properties[prop] = val;
                if (this.canvas) {
                    this.canvas.setDirty(true, true);
                }
            });
        });
    }

    clearInspector() {
        const body = document.getElementById("nn-inspector-body");
        if (body) {
            const iconHtml = (window.SignalML && typeof window.SignalML.icon === "function") ? window.SignalML.icon("layers") : "";
            body.innerHTML = `
                <div class="empty-inspector">
                    ${iconHtml}
                    <p>Select any layer node on the canvas to configure units, activation, and hyper-parameters.</p>
                </div>
            `;
        }
    }

    autoLayout() {
        if (!this.graph || !this.graph._nodes || this.graph._nodes.length === 0) {
            this.toast("No nodes to align", "warning");
            return;
        }

        const nodes = this.graph._nodes;
        const links = this.graph.links || {};

        // Build adjacency list and in-degree for topological longest-path depth computation
        const adj = {};
        const inDegree = {};
        nodes.forEach(n => {
            adj[n.id] = [];
            inDegree[n.id] = 0;
        });

        // Map node id to node object
        const nodeMap = {};
        nodes.forEach(n => {
            nodeMap[n.id] = n;
        });

        // Collect links from graph.links or node outputs
        const allLinks = [];
        if (Array.isArray(links)) {
            links.forEach(l => {
                if (l) allLinks.push({ origin_id: l[1], target_id: l[3] });
            });
        } else if (typeof links === "object" && links !== null) {
            Object.values(links).forEach(l => {
                if (l) allLinks.push({ origin_id: l.origin_id || l[1], target_id: l.target_id || l[3] });
            });
        }

        // Also check node.outputs/inputs if graph.links is sparse
        nodes.forEach(n => {
            if (n.outputs) {
                n.outputs.forEach((out, outIdx) => {
                    if (out.links) {
                        out.links.forEach(linkId => {
                            const l = (this.graph.links && this.graph.links[linkId]) || (Array.isArray(links) && links.find(x => x && x[0] === linkId));
                            if (l) {
                                allLinks.push({ origin_id: l[1], target_id: l[3] });
                            }
                        });
                    }
                });
            }
        });

        // Deduplicate links
        const uniqueLinks = {};
        allLinks.forEach(l => {
            if (l.origin_id && l.target_id && nodeMap[l.origin_id] && nodeMap[l.target_id]) {
                const key = `${l.origin_id}->${l.target_id}`;
                if (!uniqueLinks[key]) {
                    uniqueLinks[key] = true;
                    adj[l.origin_id].push(l.target_id);
                    inDegree[l.target_id] = (inDegree[l.target_id] || 0) + 1;
                }
            }
        });

        // Compute longest path depth using Kahn's algorithm / BFS topological sort
        const depth = {};
        nodes.forEach(n => { depth[n.id] = 0; });

        const queue = [];
        nodes.forEach(n => {
            if ((inDegree[n.id] || 0) === 0) {
                queue.push(n.id);
                depth[n.id] = 0;
            }
        });

        // If no 0 in-degree nodes found, fallback to node order
        if (queue.length === 0 && nodes.length > 0) {
            queue.push(nodes[0].id);
            depth[nodes[0].id] = 0;
        }

        while (queue.length > 0) {
            const u = queue.shift();
            const currentDepth = depth[u];
            (adj[u] || []).forEach(v => {
                if (depth[v] < currentDepth + 1) {
                    depth[v] = currentDepth + 1;
                }
                inDegree[v]--;
                if (inDegree[v] === 0) {
                    queue.push(v);
                }
            });
        }

        // Group nodes by depth
        const depthGroups = {};
        nodes.forEach(n => {
            const d = depth[n.id] !== undefined ? depth[n.id] : 0;
            if (!depthGroups[d]) depthGroups[d] = [];
            depthGroups[d].push(n);
        });

        // Position nodes into clean horizontal columns and vertically centered rows
        const colWidth = 260;
        const rowHeight = 160;
        const startX = 80;
        const startY = 140;

        Object.keys(depthGroups).sort((a, b) => Number(a) - Number(b)).forEach(dStr => {
            const d = Number(dStr);
            const groupNodes = depthGroups[d];
            const x = startX + d * colWidth;
            const totalGroupHeight = groupNodes.length * rowHeight;
            
            groupNodes.forEach((node, idx) => {
                const y = startY + idx * rowHeight;
                node.pos = [x, y];
            });
        });

        if (this.canvas) {
            this.canvas.setDirty(true, true);
        }

        this.toast("Nodes aligned", "success");
    }

    convertKerasConfigToLiteGraph(kerasConfig) {
        const layersList = kerasConfig.config?.layers || kerasConfig.layers || [];
        const nodes = [];
        const links = [];
        const nameToNodeId = {};

        layersList.forEach((layer, idx) => {
            nameToNodeId[layer.name] = idx + 1;
        });

        layersList.forEach((layer, idx) => {
            const nodeId = idx + 1;
            const cname = (layer.class_name || "").toLowerCase();
            let mappedType = cname;
            if (cname === "inputlayer") mappedType = "input";
            else if (cname === "timedistributed" || cname === "time_distributed") mappedType = "timedistributed";
            else if (cname === "conv1dtranspose" || cname === "deconv1d") mappedType = "conv1dtranspose";
            else if (cname === "average") mappedType = "average";
            else if (cname === "dot") mappedType = "dot";
            else if (cname === "batchnormalization") mappedType = "batchnorm";
            else if (cname === "layernormalization") mappedType = "layernorm";
            else if (cname === "unitnormalization") mappedType = "unitnorm";
            else if (cname === "maxpooling1d" || cname === "max_pooling1d") mappedType = "maxpooling1d";
            else if (cname === "averagepooling1d" || cname === "avgpooling1d") mappedType = "averagepooling1d";
            else if (cname === "globalmaxpooling1d" || cname === "global_maxpooling1d") mappedType = "global_maxpooling1d";
            else if (cname === "globalaveragepooling1d" || cname === "global_averagepooling1d") mappedType = "global_averagepooling1d";
            else if (cname === "upsampling1d") mappedType = "upsampling1d";
            else if (cname === "multiheadattention") mappedType = "multihead_attention";
            else if (cname === "positionalencoding") mappedType = "positional_encoding";
            else if (cname === "clusteringlayer") mappedType = "clustering";
            else if (cname === "customsignallayer" || cname === "customlayer") mappedType = "custom_layer";

            const props = {};
            const cfg = layer.config || {};
            if (cfg.filters !== undefined) props.filters = Number(cfg.filters);
            if (cfg.kernel_size !== undefined) props.kernel_size = Array.isArray(cfg.kernel_size) ? cfg.kernel_size[0] : Number(cfg.kernel_size);
            if (cfg.strides !== undefined) props.strides = Array.isArray(cfg.strides) ? cfg.strides[0] : Number(cfg.strides);
            if (cfg.padding !== undefined) props.padding = String(cfg.padding);
            if (cfg.units !== undefined) props.units = Number(cfg.units);
            if (cfg.activation !== undefined) props.activation = typeof cfg.activation === "string" ? cfg.activation : (cfg.activation?.name || "relu");
            if (cfg.pool_size !== undefined) props.pool_size = Array.isArray(cfg.pool_size) ? cfg.pool_size[0] : Number(cfg.pool_size);
            if (cfg.size !== undefined) props.size = Array.isArray(cfg.size) ? cfg.size[0] : Number(cfg.size);
            if (cfg.rate !== undefined) props.rate = Number(cfg.rate);
            if (cfg.return_sequences !== undefined) props.return_sequences = Boolean(cfg.return_sequences);
            if (cfg.n_clusters !== undefined) props.n_clusters = Number(cfg.n_clusters);
            if (cfg.alpha !== undefined) props.alpha = Number(cfg.alpha);
            if (mappedType === "input") {
                const shape = cfg.batch_shape || cfg.input_shape || cfg.shape;
                if (Array.isArray(shape)) {
                    const nonBatch = shape[0] === null ? shape.slice(1) : shape;
                    props.shape = nonBatch.join(", ");
                }
            }

            const col = idx % 4;
            const row = Math.floor(idx / 4);
            const pos = [120 + col * 260, 100 + row * 180];

            const isMultiInput = ["concatenate", "add", "multiply", "average", "dot", "multihead_attention"].includes(mappedType);
            let nodeInputs = [];
            if (mappedType === "input") {
                nodeInputs = [];
            } else if (mappedType === "multihead_attention") {
                nodeInputs = [{ name: "query", type: "Tensor" }, { name: "value", type: "Tensor" }];
            } else if (isMultiInput) {
                nodeInputs = [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }];
            } else {
                nodeInputs = [{ name: "in", type: "Tensor" }];
            }

            const nodeOutputs = (mappedType === "output" ? [] : [{ name: "out", type: "Tensor", links: [] }]);

            nodes.push({
                id: nodeId,
                type: `keras/${mappedType}`,
                title: layer.name || `Layer ${nodeId}`,
                pos: pos,
                size: [200, 90],
                properties: props,
                inputs: nodeInputs,
                outputs: nodeOutputs
            });
        });

        // Reconstruct links from inbound_nodes or fallback to sequential
        let linkIdCounter = 1;
        let hasInboundLinks = false;

        function findInboundLayerNames(obj, list = []) {
            if (!obj) return list;
            if (typeof obj === "string") {
                if (nameToNodeId[obj] !== undefined) list.push(obj);
                return list;
            }
            if (Array.isArray(obj)) {
                if (obj.length >= 1 && typeof obj[0] === "string" && nameToNodeId[obj[0]] !== undefined) {
                    list.push(obj[0]);
                } else {
                    for (let item of obj) {
                        findInboundLayerNames(item, list);
                    }
                }
            } else if (typeof obj === "object") {
                if (obj.keras_history && Array.isArray(obj.keras_history) && typeof obj.keras_history[0] === "string" && nameToNodeId[obj.keras_history[0]] !== undefined) {
                    list.push(obj.keras_history[0]);
                } else if (obj.history && Array.isArray(obj.history) && typeof obj.history[0] === "string" && nameToNodeId[obj.history[0]] !== undefined) {
                    list.push(obj.history[0]);
                } else {
                    for (let v of Object.values(obj)) {
                        findInboundLayerNames(v, list);
                    }
                }
            }
            return list;
        }

        layersList.forEach((layer, idx) => {
            const targetId = idx + 1;
            if (layer.inbound_nodes && Array.isArray(layer.inbound_nodes) && layer.inbound_nodes.length > 0) {
                const inbounds = findInboundLayerNames(layer.inbound_nodes);
                if (inbounds.length > 0) {
                    inbounds.forEach((originName, slotIdx) => {
                        const originId = nameToNodeId[originName];
                        if (originId && originId !== targetId) {
                            links.push([linkIdCounter++, originId, 0, targetId, slotIdx, "Tensor"]);
                            hasInboundLinks = true;
                        }
                    });
                }
            }
        });

        if (!hasInboundLinks && layersList.length > 1) {
            for (let i = 1; i < layersList.length; i++) {
                links.push([linkIdCounter++, i, 0, i + 1, 0, "Tensor"]);
            }
        }

        // Connect node input and output slots to link IDs
        nodes.forEach(node => {
            const nodeInLinks = links.filter(l => l[3] === node.id);
            const nodeOutLinks = links.filter(l => l[1] === node.id);

            while (node.inputs.length < nodeInLinks.length) {
                node.inputs.push({ name: `in${node.inputs.length + 1}`, type: "Tensor" });
            }
            nodeInLinks.forEach(l => {
                const slot = l[4];
                if (node.inputs[slot]) {
                    node.inputs[slot].link = l[0];
                }
            });

            if (node.outputs && node.outputs.length > 0) {
                node.outputs[0].links = nodeOutLinks.map(l => l[0]);
            }
        });

        return {
            last_node_id: nodes.length,
            last_link_id: links.length,
            nodes: nodes,
            links: links
        };
    }

    loadGraph(graphData) {
        if (!graphData) return;
        if (typeof graphData === "string") {
            try {
                graphData = JSON.parse(graphData);
            } catch (e) {
                console.error("Failed to parse graphData string", e);
                this.toast("Invalid JSON format: " + e.message, "error");
                return;
            }
        }

        // Unwrap wrappers if nested
        if (graphData.graph && (graphData.graph.nodes || typeof graphData.graph === "object")) {
            graphData = graphData.graph;
        } else if (graphData.data && (graphData.data.nodes || graphData.data.graph)) {
            graphData = graphData.data.graph || graphData.data;
        }

        // Check if Keras native config JSON
        if (!graphData.nodes && (graphData.class_name || graphData.config?.layers || graphData.layers)) {
            graphData = this.convertKerasConfigToLiteGraph(graphData);
        }

        if (graphData.nodes && Array.isArray(graphData.nodes)) {
            for (let n of graphData.nodes) {
                if (!n.type) continue;
                let t = String(n.type).toLowerCase().trim();
                if (!t.startsWith("keras/")) {
                    t = "keras/" + t;
                }
                // Handle aliases
                if (t === "keras/inputlayer") t = "keras/input";
                else if (t === "keras/modeloutput") t = "keras/output";
                else if (t === "keras/deconv1d") t = "keras/conv1dtranspose";
                else if (t === "keras/time_distributed") t = "keras/timedistributed";
                else if (t === "keras/batchnormalization" || t === "keras/bn") t = "keras/batchnorm";
                else if (t === "keras/layernormalization" || t === "keras/ln") t = "keras/layernorm";
                else if (t === "keras/unitnormalization") t = "keras/unitnorm";
                else if (t === "keras/max_pooling1d" || t === "keras/maxpool1d") t = "keras/maxpooling1d";
                else if (t === "keras/avgpooling1d" || t === "keras/avg_pooling1d" || t === "keras/avgpool1d") t = "keras/averagepooling1d";
                else if (t === "keras/globalmaxpooling1d") t = "keras/global_maxpooling1d";
                else if (t === "keras/globalaveragepooling1d" || t === "keras/globalavgpooling1d") t = "keras/global_averagepooling1d";
                else if (t === "keras/multiheadattention") t = "keras/multihead_attention";
                else if (t === "keras/positionalencoding") t = "keras/positional_encoding";
                else if (t === "keras/clusteringlayer") t = "keras/clustering";
                else if (t === "keras/customsignallayer" || t === "keras/customlayer") t = "keras/custom_layer";
                n.type = t;

                if (!n.pos || !Array.isArray(n.pos)) {
                    n.pos = [100 + Math.random() * 200, 100 + Math.random() * 200];
                }
                if (!n.properties || typeof n.properties !== "object") {
                    n.properties = {};
                }
            }
        }

        this.graph.clear();
        this.graph.configure(graphData);
        if (this.canvas) {
            this.canvas.setDirty(true, true);
        }
        this.validateGraph();
    }

    serializeGraph() {
        if (!this.graph) return { nodes: [], links: [] };
        const serialized = this.graph.serialize();
        if (serialized && serialized.nodes && this.graph._nodes) {
            serialized.nodes.forEach(n => {
                const actualNode = this.graph.getNodeById(n.id);
                if (actualNode && actualNode.title) {
                    n.title = actualNode.title;
                }
            });
        }
        return serialized;
    }

    async validateGraph() {
        const payload = { graph: this.serializeGraph() };
        const SML = (typeof window !== "undefined" && window.SignalML) || (typeof SignalML !== "undefined" ? SignalML : {
            toast: (m, t) => this.toast(m, t),
            safeJson: async (r) => { try { return await r.json(); } catch (_) { return {}; } }
        });
        try {
            const res = await fetch("/api/models/validate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await SML.safeJson(res);
            if (data.valid) {
                this.toast(`Valid Keras Architecture! Total params: ${data.summary.total_params.toLocaleString()}`, "success");
                
                if (data.summary && Array.isArray(data.summary.layers) && this.graph) {
                    data.summary.layers.forEach(l => {
                        const node = this.graph.getNodeById(Number(l.node_id)) || this.graph.getNodeById(l.node_id);
                        if (node) {
                            node._inferredShape = l.output_shape;
                            node._paramCount = l.params;
                            const shapeStr = Array.isArray(l.output_shape)
                                ? "(" + l.output_shape.map(d => (d === null || d === -1) ? "?" : d).join(", ") + ")"
                                : "";
                            if (node.outputs && node.outputs.length > 0 && shapeStr) {
                                node.outputs[0].label = shapeStr;
                            }
                        }
                    });
                    if (this.canvas) this.canvas.setDirty(true, true);
                }

                if (this.selectedNode) {
                    this.updateInspector(this.selectedNode);
                }

                this.renderShapesSummary(data.summary);
            } else {
                SML.toast(`Validation Error: ${data.error || data.detail || "Validation Error"}`, "error");
            }
        } catch (e) {
            SML.toast(`Network Error: ${e.message}`, "error");
        }
    }

    renderShapesSummary(summary) {
        const container = document.getElementById("shapes-summary-container");
        if (!container) return;

        let html = `
            <div class="tile mt-3">
                <div class="row-between mb-2">
                    <strong class="text-sm">Inferred Layer Shapes</strong>
                    <span class="shape-badge">${summary.total_params.toLocaleString()} Params</span>
                </div>
                <div class="max-h-xs scroll-y">
        `;
        for (let l of summary.layers) {
            html += `
                <div class="row-between text-xs py-1 border-b">
                    <span>${l.layer_type} (<span class="mono">${l.node_id}</span>)</span>
                    <span class="shape-badge">${JSON.stringify(l.output_shape)}</span>
                </div>
            `;
        }
        html += `</div></div>`;
        container.innerHTML = html;
    }

    async buildAndSave(name = "Keras Signal Model", taskType = "classification", optimizer = "adam") {
        const graphData = this.serializeGraph();
        const activeProjectId = this.getActiveProjectId();
        const SML = (typeof window !== "undefined" && window.SignalML) || (typeof SignalML !== "undefined" ? SignalML : {
            currentProjectId: activeProjectId,
            toast: (m, t) => this.toast(m, t),
            safeJson: async (r) => { try { return await r.json(); } catch (_) { return {}; } }
        });
        const payload = {
            project_id: activeProjectId,
            name: name,
            task_type: taskType,
            optimizer: optimizer,
            graph: graphData
        };

        try {
            const res = await fetch("/api/models/build", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                const data = await SML.safeJson(res);
                this.toast(`Model '${name}' compiled and saved successfully!`, "success");
                SML.activeModel = data.model_id;
                if (typeof window.populateSavedModelsDropdown === "function") {
                    await window.populateSavedModelsDropdown();
                    const sel = document.getElementById("load-saved-model-select");
                    if (sel && data.model_id) {
                        sel.value = data.model_id;
                        const updateBtn = document.getElementById("update-saved-model-btn");
                        const deleteBtn = document.getElementById("delete-saved-model-btn");
                        if (updateBtn) { updateBtn.hidden = false; updateBtn.style.display = "inline-flex"; }
                        if (deleteBtn) { deleteBtn.hidden = false; deleteBtn.style.display = "inline-flex"; }
                    }
                }
                return data;
            } else {
                const err = await SML.safeJson(res);
                this.toast(`Build Failed: ${err.detail || "Validation Error"}`, "error");
            }
        } catch (e) {
            this.toast(`Network Error: ${e.message}`, "error");
        }
    }

    generateArchitecture(type, params = {}) {
        this.graph.clear();
        const seqLen = params.seqLen || 512;
        const baseFilters = params.baseFilters || 32;
        const latentDim = params.latentDim || 16;
        const numClasses = params.numClasses || 4;
        const act = params.activation || "relu";

        if (type === "conv_ae") {
            // 1D Convolutional Autoencoder:
            // Input(seq, 1) -> Conv1D(base) -> MaxPool(2) -> Conv1D(base*2) -> MaxPool(2) -> Bottleneck(latent)
            //               -> UpSampling1D(2) -> Conv1D(base*2) -> UpSampling1D(2) -> Conv1D(base) -> Conv1D(1, linear) -> Output
            const nInput = LiteGraph.createNode("keras/input");
            nInput.pos = [80, 160];
            nInput.properties.shape = `${seqLen}, 1`;
            this.graph.add(nInput);

            const nEnc1 = LiteGraph.createNode("keras/conv1d");
            nEnc1.pos = [300, 160];
            nEnc1.title = "Encoder Conv1";
            nEnc1.properties.filters = baseFilters;
            nEnc1.properties.kernel_size = 7;
            nEnc1.properties.activation = act;
            this.graph.add(nEnc1);

            const nPool1 = LiteGraph.createNode("keras/maxpooling1d");
            nPool1.pos = [520, 160];
            nPool1.properties.pool_size = 2;
            this.graph.add(nPool1);

            const nEnc2 = LiteGraph.createNode("keras/conv1d");
            nEnc2.pos = [740, 160];
            nEnc2.title = "Encoder Conv2";
            nEnc2.properties.filters = baseFilters * 2;
            nEnc2.properties.kernel_size = 5;
            nEnc2.properties.activation = act;
            this.graph.add(nEnc2);

            const nPool2 = LiteGraph.createNode("keras/maxpooling1d");
            nPool2.pos = [960, 160];
            nPool2.properties.pool_size = 2;
            this.graph.add(nPool2);

            // Bottleneck (Latent Space)
            const nBottle = LiteGraph.createNode("keras/conv1d");
            nBottle.pos = [1180, 260];
            nBottle.title = "Bottleneck (Latent)";
            nBottle.properties.filters = latentDim;
            nBottle.properties.kernel_size = 3;
            nBottle.properties.activation = act;
            this.graph.add(nBottle);

            // Decoder
            const nUp1 = LiteGraph.createNode("keras/upsampling1d");
            nUp1.pos = [960, 420];
            nUp1.properties.size = 2;
            this.graph.add(nUp1);

            const nDec1 = LiteGraph.createNode("keras/conv1d");
            nDec1.pos = [740, 420];
            nDec1.title = "Decoder Conv1";
            nDec1.properties.filters = baseFilters * 2;
            nDec1.properties.kernel_size = 5;
            nDec1.properties.activation = act;
            this.graph.add(nDec1);

            const nUp2 = LiteGraph.createNode("keras/upsampling1d");
            nUp2.pos = [520, 420];
            nUp2.properties.size = 2;
            this.graph.add(nUp2);

            const nDec2 = LiteGraph.createNode("keras/conv1d");
            nDec2.pos = [300, 420];
            nDec2.title = "Decoder Conv2";
            nDec2.properties.filters = baseFilters;
            nDec2.properties.kernel_size = 7;
            nDec2.properties.activation = act;
            this.graph.add(nDec2);

            const nFinal = LiteGraph.createNode("keras/conv1d");
            nFinal.pos = [80, 420];
            nFinal.title = "Reconstructed Waveform";
            nFinal.properties.filters = 1;
            nFinal.properties.kernel_size = 3;
            nFinal.properties.activation = "linear";
            this.graph.add(nFinal);

            const nOutput = LiteGraph.createNode("keras/output");
            nOutput.pos = [80, (seqLen % 4 === 0) ? 560 : 740];
            nOutput.title = "AE Output";
            nOutput.properties.activation = "linear";
            this.graph.add(nOutput);

            // Connections
            nInput.connect(0, nEnc1, 0);
            nEnc1.connect(0, nPool1, 0);
            nPool1.connect(0, nEnc2, 0);
            nEnc2.connect(0, nPool2, 0);
            nPool2.connect(0, nBottle, 0);
            nBottle.connect(0, nUp1, 0);
            nUp1.connect(0, nDec1, 0);
            nDec1.connect(0, nUp2, 0);
            nUp2.connect(0, nDec2, 0);

            if (seqLen % 4 !== 0) {
                const nAdjFlat = LiteGraph.createNode("keras/flatten");
                nAdjFlat.pos = [80, 500];
                nAdjFlat.title = "Shape Adjust Flatten";
                this.graph.add(nAdjFlat);

                const nAdjDense = LiteGraph.createNode("keras/dense");
                nAdjDense.pos = [80, 580];
                nAdjDense.title = "Target Dimension Dense";
                nAdjDense.properties.units = seqLen;
                nAdjDense.properties.activation = "linear";
                this.graph.add(nAdjDense);

                const nAdjReshape = LiteGraph.createNode("keras/reshape");
                nAdjReshape.pos = [80, 660];
                nAdjReshape.title = "Target Shape Reshape";
                nAdjReshape.properties.target_shape = `${seqLen}, 1`;
                this.graph.add(nAdjReshape);

                nDec2.connect(0, nFinal, 0);
                nFinal.connect(0, nAdjFlat, 0);
                nAdjFlat.connect(0, nAdjDense, 0);
                nAdjDense.connect(0, nAdjReshape, 0);
                nAdjReshape.connect(0, nOutput, 0);
            } else {
                nDec2.connect(0, nFinal, 0);
                nFinal.connect(0, nOutput, 0);
            }

        } else if (type === "unet_1d") {
            // 1D U-Net with Skip Connections
            const nInput = LiteGraph.createNode("keras/input");
            nInput.pos = [80, 140];
            nInput.properties.shape = `${seqLen}, 1`;
            this.graph.add(nInput);

            const nEnc1 = LiteGraph.createNode("keras/conv1d");
            nEnc1.pos = [320, 140];
            nEnc1.title = "Encoder L1 (Skip Source)";
            nEnc1.properties.filters = baseFilters;
            nEnc1.properties.kernel_size = 5;
            this.graph.add(nEnc1);

            const nDown1 = LiteGraph.createNode("keras/maxpooling1d");
            nDown1.pos = [540, 140];
            nDown1.properties.pool_size = 2;
            this.graph.add(nDown1);

            const nEnc2 = LiteGraph.createNode("keras/conv1d");
            nEnc2.pos = [760, 240];
            nEnc2.title = "Encoder L2 (Skip Source)";
            nEnc2.properties.filters = baseFilters * 2;
            nEnc2.properties.kernel_size = 5;
            this.graph.add(nEnc2);

            const nDown2 = LiteGraph.createNode("keras/maxpooling1d");
            nDown2.pos = [980, 240];
            nDown2.properties.pool_size = 2;
            this.graph.add(nDown2);

            const nBottle = LiteGraph.createNode("keras/conv1d");
            nBottle.pos = [1200, 360];
            nBottle.title = "U-Net Bottleneck";
            nBottle.properties.filters = baseFilters * 4;
            nBottle.properties.kernel_size = 3;
            this.graph.add(nBottle);

            // Level 2 Decoder with skip from Enc2
            const nUp2 = LiteGraph.createNode("keras/upsampling1d");
            nUp2.pos = [980, 480];
            nUp2.properties.size = 2;
            this.graph.add(nUp2);

            const nConcat2 = LiteGraph.createNode("keras/concatenate");
            nConcat2.pos = [760, 480];
            nConcat2.title = "Skip Concat L2";
            nConcat2.properties.axis = -1;
            this.graph.add(nConcat2);

            const nDec2 = LiteGraph.createNode("keras/conv1d");
            nDec2.pos = [540, 480];
            nDec2.title = "Decoder L2";
            nDec2.properties.filters = baseFilters * 2;
            this.graph.add(nDec2);

            // Level 1 Decoder with skip from Enc1
            const nUp1 = LiteGraph.createNode("keras/upsampling1d");
            nUp1.pos = [320, 480];
            nUp1.properties.size = 2;
            this.graph.add(nUp1);

            const nConcat1 = LiteGraph.createNode("keras/concatenate");
            nConcat1.pos = [100, 480];
            nConcat1.title = "Skip Concat L1";
            nConcat1.properties.axis = -1;
            this.graph.add(nConcat1);

            const nDec1 = LiteGraph.createNode("keras/conv1d");
            nDec1.pos = [100, 620];
            nDec1.title = "Decoder L1";
            nDec1.properties.filters = baseFilters;
            this.graph.add(nDec1);

            const nFinal = LiteGraph.createNode("keras/conv1d");
            nFinal.pos = [360, 620];
            nFinal.title = "U-Net Segment/Reconstruct";
            nFinal.properties.filters = 1;
            nFinal.properties.kernel_size = 1;
            nFinal.properties.activation = "linear";
            this.graph.add(nFinal);

            const nOutput = LiteGraph.createNode("keras/output");
            nOutput.pos = [620, 620];
            nOutput.title = "Model Output";
            nOutput.properties.activation = "linear";
            this.graph.add(nOutput);

            // Connections
            nInput.connect(0, nEnc1, 0);
            nEnc1.connect(0, nDown1, 0);
            nDown1.connect(0, nEnc2, 0);
            nEnc2.connect(0, nDown2, 0);
            nDown2.connect(0, nBottle, 0);
            nBottle.connect(0, nUp2, 0);

            // Skip connection 2: up2 -> concat2[0], enc2 -> concat2[1]
            nUp2.connect(0, nConcat2, 0);
            nEnc2.connect(0, nConcat2, 1);
            nConcat2.connect(0, nDec2, 0);

            nDec2.connect(0, nUp1, 0);
            // Skip connection 1: up1 -> concat1[0], enc1 -> concat1[1]
            nUp1.connect(0, nConcat1, 0);
            nEnc1.connect(0, nConcat1, 1);
            nConcat1.connect(0, nDec1, 0);

            nDec1.connect(0, nFinal, 0);
            nFinal.connect(0, nOutput, 0);

        } else if (type === "dense_ae") {
            // Dense / MLP Autoencoder
            const nInput = LiteGraph.createNode("keras/input");
            nInput.pos = [80, 180];
            nInput.properties.shape = `${seqLen}, 1`;
            this.graph.add(nInput);

            const nFlat = LiteGraph.createNode("keras/flatten");
            nFlat.pos = [300, 180];
            this.graph.add(nFlat);

            const nEnc = LiteGraph.createNode("keras/dense");
            nEnc.pos = [520, 180];
            nEnc.title = "Dense Encoder";
            nEnc.properties.units = 128;
            this.graph.add(nEnc);

            const nBottle = LiteGraph.createNode("keras/dense");
            nBottle.pos = [740, 180];
            nBottle.title = "Latent Code";
            nBottle.properties.units = latentDim;
            this.graph.add(nBottle);

            const nDec = LiteGraph.createNode("keras/dense");
            nDec.pos = [960, 180];
            nDec.title = "Dense Decoder";
            nDec.properties.units = 128;
            this.graph.add(nDec);

            const nOutDense = LiteGraph.createNode("keras/dense");
            nOutDense.pos = [1180, 180];
            nOutDense.title = "Reconstruct Dense";
            nOutDense.properties.units = seqLen;
            nOutDense.properties.activation = "linear";
            this.graph.add(nOutDense);

            const nReshape = LiteGraph.createNode("keras/reshape");
            nReshape.pos = [1400, 180];
            nReshape.properties.target_shape = `${seqLen}, 1`;
            this.graph.add(nReshape);

            const nOutput = LiteGraph.createNode("keras/output");
            nOutput.pos = [1620, 180];
            nOutput.properties.activation = "linear";
            this.graph.add(nOutput);

            nInput.connect(0, nFlat, 0);
            nFlat.connect(0, nEnc, 0);
            nEnc.connect(0, nBottle, 0);
            nBottle.connect(0, nDec, 0);
            nDec.connect(0, nOutDense, 0);
            nOutDense.connect(0, nReshape, 0);
            nReshape.connect(0, nOutput, 0);

        } else if (type === "cnn_classifier") {
            // 1D CNN Classifier
            const nInput = LiteGraph.createNode("keras/input");
            nInput.pos = [80, 200];
            nInput.properties.shape = `${seqLen}, 1`;
            this.graph.add(nInput);

            const nConv1 = LiteGraph.createNode("keras/conv1d");
            nConv1.pos = [300, 200];
            nConv1.properties.filters = baseFilters;
            this.graph.add(nConv1);

            const nPool1 = LiteGraph.createNode("keras/maxpooling1d");
            nPool1.pos = [520, 200];
            this.graph.add(nPool1);

            const nConv2 = LiteGraph.createNode("keras/conv1d");
            nConv2.pos = [740, 200];
            nConv2.properties.filters = baseFilters * 2;
            this.graph.add(nConv2);

            const nPool2 = LiteGraph.createNode("keras/global_maxpooling1d");
            nPool2.pos = [960, 200];
            this.graph.add(nPool2);

            const nDense1 = LiteGraph.createNode("keras/dense");
            nDense1.pos = [1180, 200];
            nDense1.properties.units = 64;
            this.graph.add(nDense1);

            const nDrop = LiteGraph.createNode("keras/dropout");
            nDrop.pos = [1380, 200];
            nDrop.properties.rate = 0.3;
            this.graph.add(nDrop);

            const nDenseOut = LiteGraph.createNode("keras/dense");
            nDenseOut.pos = [1580, 200];
            nDenseOut.properties.units = numClasses;
            nDenseOut.properties.activation = "softmax";
            this.graph.add(nDenseOut);

            const nOutput = LiteGraph.createNode("keras/output");
            nOutput.pos = [1780, 200];
            nOutput.properties.activation = "softmax";
            this.graph.add(nOutput);

            nInput.connect(0, nConv1, 0);
            nConv1.connect(0, nPool1, 0);
            nPool1.connect(0, nConv2, 0);
            nConv2.connect(0, nPool2, 0);
            nPool2.connect(0, nDense1, 0);
            nDense1.connect(0, nDrop, 0);
            nDrop.connect(0, nDenseOut, 0);
            nDenseOut.connect(0, nOutput, 0);

        } else if (type === "bilstm") {
            // BiLSTM Time Series Classifier
            const nInput = LiteGraph.createNode("keras/input");
            nInput.pos = [80, 200];
            nInput.properties.shape = `${seqLen}, 1`;
            this.graph.add(nInput);

            const nBi = LiteGraph.createNode("keras/bidirectional");
            nBi.pos = [320, 200];
            nBi.properties.units = 64;
            nBi.properties.return_sequences = true;
            this.graph.add(nBi);

            const nLstm = LiteGraph.createNode("keras/lstm");
            nLstm.pos = [580, 200];
            nLstm.properties.units = 32;
            nLstm.properties.return_sequences = false;
            this.graph.add(nLstm);

            const nDense = LiteGraph.createNode("keras/dense");
            nDense.pos = [820, 200];
            nDense.properties.units = 32;
            this.graph.add(nDense);

            const nOut = LiteGraph.createNode("keras/dense");
            nOut.pos = [1040, 200];
            nOut.properties.units = numClasses;
            nOut.properties.activation = "softmax";
            this.graph.add(nOut);

            const nOutput = LiteGraph.createNode("keras/output");
            nOutput.pos = [1260, 200];
            nOutput.properties.activation = "softmax";
            this.graph.add(nOutput);

            nInput.connect(0, nBi, 0);
            nBi.connect(0, nLstm, 0);
            nLstm.connect(0, nDense, 0);
            nDense.connect(0, nOut, 0);
            nOut.connect(0, nOutput, 0);

        } else if (type === "transformer_1d") {
            // 1D Transformer Attention
            const nInput = LiteGraph.createNode("keras/input");
            nInput.pos = [80, 200];
            nInput.properties.shape = `${seqLen}, 1`;
            this.graph.add(nInput);

            const nConv = LiteGraph.createNode("keras/conv1d");
            nConv.pos = [300, 200];
            nConv.title = "Linear Projection";
            nConv.properties.filters = 64;
            this.graph.add(nConv);

            const nPos = LiteGraph.createNode("keras/positional_encoding");
            nPos.pos = [520, 200];
            nPos.properties.d_model = 64;
            this.graph.add(nPos);

            const nMha = LiteGraph.createNode("keras/multihead_attention");
            nMha.pos = [760, 200];
            nMha.properties.num_heads = 4;
            nMha.properties.key_dim = 32;
            this.graph.add(nMha);

            const nAdd = LiteGraph.createNode("keras/add");
            nAdd.pos = [1020, 200];
            this.graph.add(nAdd);

            const nNorm = LiteGraph.createNode("keras/layernorm");
            nNorm.pos = [1240, 200];
            nNorm.properties.axis = -1;
            this.graph.add(nNorm);

            const nPool = LiteGraph.createNode("keras/global_averagepooling1d");
            nPool.pos = [1460, 200];
            this.graph.add(nPool);

            const nOut = LiteGraph.createNode("keras/dense");
            nOut.pos = [1680, 200];
            nOut.properties.units = numClasses;
            nOut.properties.activation = "softmax";
            this.graph.add(nOut);

            const nOutput = LiteGraph.createNode("keras/output");
            nOutput.pos = [1900, 200];
            nOutput.properties.activation = "softmax";
            this.graph.add(nOutput);

            nInput.connect(0, nConv, 0);
            nConv.connect(0, nPos, 0);
            nPos.connect(0, nMha, 0);
            nMha.connect(0, nAdd, 0);
            nPos.connect(0, nAdd, 1);
            nAdd.connect(0, nNorm, 0);
            nNorm.connect(0, nPool, 0);
            nPool.connect(0, nOut, 0);
            nOut.connect(0, nOutput, 0);

        } else if (type === "idec_cnn") {
            // 1D-CNN IDEC with editable Encoder, Decoder, and ClusteringLayer branch
            const nInput = LiteGraph.createNode("keras/input");
            nInput.pos = [80, 220];
            nInput.title = "Signal Input";
            nInput.properties.shape = `${seqLen}, 1`;
            this.graph.add(nInput);

            // Encoder
            const nEnc1 = LiteGraph.createNode("keras/conv1d");
            nEnc1.pos = [300, 220];
            nEnc1.title = "Encoder Conv1";
            nEnc1.properties.filters = baseFilters;
            nEnc1.properties.kernel_size = 7;
            nEnc1.properties.activation = act;
            this.graph.add(nEnc1);

            const nPool1 = LiteGraph.createNode("keras/maxpooling1d");
            nPool1.pos = [520, 220];
            nPool1.properties.pool_size = 2;
            this.graph.add(nPool1);

            const nEnc2 = LiteGraph.createNode("keras/conv1d");
            nEnc2.pos = [740, 220];
            nEnc2.title = "Encoder Conv2";
            nEnc2.properties.filters = baseFilters * 2;
            nEnc2.properties.kernel_size = 5;
            nEnc2.properties.activation = act;
            this.graph.add(nEnc2);

            const nPool2 = LiteGraph.createNode("keras/maxpooling1d");
            nPool2.pos = [960, 220];
            nPool2.properties.pool_size = 2;
            this.graph.add(nPool2);

            const nFlat = LiteGraph.createNode("keras/flatten");
            nFlat.pos = [1180, 220];
            this.graph.add(nFlat);

            const nBottle = LiteGraph.createNode("keras/dense");
            nBottle.pos = [1400, 220];
            nBottle.title = "Bottleneck Latent (Z)";
            nBottle.properties.units = latentDim;
            nBottle.properties.activation = "linear";
            this.graph.add(nBottle);

            // Branch A: ClusteringLayer
            const nClust = LiteGraph.createNode("keras/clustering");
            nClust.pos = [1640, 100];
            nClust.title = "IDEC ClusteringLayer";
            nClust.properties.n_clusters = numClasses || 8;
            nClust.properties.alpha = 1.0;
            this.graph.add(nClust);

            const nClustOut = LiteGraph.createNode("keras/output");
            nClustOut.pos = [1880, 100];
            nClustOut.title = "Cluster Probabilities (Q)";
            nClustOut.properties.activation = "linear";
            this.graph.add(nClustOut);

            // Branch B: Decoder (Signal Reconstruction)
            const nDecDense = LiteGraph.createNode("keras/dense");
            nDecDense.pos = [1640, 340];
            nDecDense.title = "Decoder Expand";
            nDecDense.properties.units = Math.floor(seqLen / 4) * baseFilters;
            nDecDense.properties.activation = act;
            this.graph.add(nDecDense);

            const nReshape = LiteGraph.createNode("keras/reshape");
            nReshape.pos = [1860, 340];
            nReshape.properties.target_shape = `${Math.floor(seqLen / 4)}, ${baseFilters}`;
            this.graph.add(nReshape);

            const nUp1 = LiteGraph.createNode("keras/upsampling1d");
            nUp1.pos = [2080, 340];
            nUp1.properties.size = 2;
            this.graph.add(nUp1);

            const nDec1 = LiteGraph.createNode("keras/conv1d");
            nDec1.pos = [2300, 340];
            nDec1.title = "Decoder Conv1";
            nDec1.properties.filters = baseFilters;
            nDec1.properties.kernel_size = 5;
            nDec1.properties.activation = act;
            this.graph.add(nDec1);

            const nUp2 = LiteGraph.createNode("keras/upsampling1d");
            nUp2.pos = [2520, 340];
            nUp2.properties.size = 2;
            this.graph.add(nUp2);

            const nDec2 = LiteGraph.createNode("keras/conv1d");
            nDec2.pos = [2740, 340];
            nDec2.title = "Reconstructed Waveform";
            nDec2.properties.filters = 1;
            nDec2.properties.kernel_size = 7;
            nDec2.properties.activation = "linear";
            this.graph.add(nDec2);

            const nReconOut = LiteGraph.createNode("keras/output");
            nReconOut.pos = [(seqLen % 4 === 0) ? 2960 : 3620, 340];
            nReconOut.title = "Reconstruction Output";
            nReconOut.properties.activation = "linear";
            this.graph.add(nReconOut);

            // Connections
            nInput.connect(0, nEnc1, 0);
            nEnc1.connect(0, nPool1, 0);
            nPool1.connect(0, nEnc2, 0);
            nEnc2.connect(0, nPool2, 0);
            nPool2.connect(0, nFlat, 0);
            nFlat.connect(0, nBottle, 0);

            // Latent to Clustering branch
            nBottle.connect(0, nClust, 0);
            nClust.connect(0, nClustOut, 0);

            // Latent to Decoder branch
            nBottle.connect(0, nDecDense, 0);
            nDecDense.connect(0, nReshape, 0);
            nReshape.connect(0, nUp1, 0);
            nUp1.connect(0, nDec1, 0);
            nDec1.connect(0, nUp2, 0);
            nUp2.connect(0, nDec2, 0);

            if (seqLen % 4 !== 0) {
                const nAdjFlat = LiteGraph.createNode("keras/flatten");
                nAdjFlat.pos = [2960, 340];
                nAdjFlat.title = "Shape Adjust Flatten";
                this.graph.add(nAdjFlat);

                const nAdjDense = LiteGraph.createNode("keras/dense");
                nAdjDense.pos = [3180, 340];
                nAdjDense.title = "Target Dimension Dense";
                nAdjDense.properties.units = seqLen;
                nAdjDense.properties.activation = "linear";
                this.graph.add(nAdjDense);

                const nAdjReshape = LiteGraph.createNode("keras/reshape");
                nAdjReshape.pos = [3400, 340];
                nAdjReshape.title = "Target Shape Reshape";
                nAdjReshape.properties.target_shape = `${seqLen}, 1`;
                this.graph.add(nAdjReshape);

                nDec2.connect(0, nAdjFlat, 0);
                nAdjFlat.connect(0, nAdjDense, 0);
                nAdjDense.connect(0, nAdjReshape, 0);
                nAdjReshape.connect(0, nReconOut, 0);
            } else {
                nDec2.connect(0, nReconOut, 0);
            }

        } else if (type === "idec_dense") {
            // Dense IDEC with editable Encoder, Decoder, and ClusteringLayer branch
            const nInput = LiteGraph.createNode("keras/input");
            nInput.pos = [80, 200];
            nInput.properties.shape = `${seqLen}, 1`;
            this.graph.add(nInput);

            const nFlat = LiteGraph.createNode("keras/flatten");
            nFlat.pos = [300, 200];
            this.graph.add(nFlat);

            const nEnc1 = LiteGraph.createNode("keras/dense");
            nEnc1.pos = [520, 200];
            nEnc1.title = "Dense Encoder 1";
            nEnc1.properties.units = 128;
            nEnc1.properties.activation = act;
            this.graph.add(nEnc1);

            const nEnc2 = LiteGraph.createNode("keras/dense");
            nEnc2.pos = [740, 200];
            nEnc2.title = "Dense Encoder 2";
            nEnc2.properties.units = 64;
            nEnc2.properties.activation = act;
            this.graph.add(nEnc2);

            const nBottle = LiteGraph.createNode("keras/dense");
            nBottle.pos = [960, 200];
            nBottle.title = "Bottleneck Latent (Z)";
            nBottle.properties.units = latentDim;
            nBottle.properties.activation = "linear";
            this.graph.add(nBottle);

            // Branch A: Clustering branch
            const nClust = LiteGraph.createNode("keras/clustering");
            nClust.pos = [1200, 100];
            nClust.title = "IDEC ClusteringLayer";
            nClust.properties.n_clusters = numClasses || 8;
            nClust.properties.alpha = 1.0;
            this.graph.add(nClust);

            const nClustOut = LiteGraph.createNode("keras/output");
            nClustOut.pos = [1440, 100];
            nClustOut.title = "Cluster Probabilities (Q)";
            nClustOut.properties.activation = "linear";
            this.graph.add(nClustOut);

            // Branch B: Decoder branch
            const nDec1 = LiteGraph.createNode("keras/dense");
            nDec1.pos = [1200, 320];
            nDec1.title = "Dense Decoder 1";
            nDec1.properties.units = 64;
            nDec1.properties.activation = act;
            this.graph.add(nDec1);

            const nDec2 = LiteGraph.createNode("keras/dense");
            nDec2.pos = [1420, 320];
            nDec2.title = "Dense Decoder 2";
            nDec2.properties.units = 128;
            nDec2.properties.activation = act;
            this.graph.add(nDec2);

            const nDecOut = LiteGraph.createNode("keras/dense");
            nDecOut.pos = [1640, 320];
            nDecOut.title = "Reconstructed Dense";
            nDecOut.properties.units = seqLen;
            nDecOut.properties.activation = "linear";
            this.graph.add(nDecOut);

            const nReshape = LiteGraph.createNode("keras/reshape");
            nReshape.pos = [1860, 320];
            nReshape.properties.target_shape = `${seqLen}, 1`;
            this.graph.add(nReshape);

            const nReconOut = LiteGraph.createNode("keras/output");
            nReconOut.pos = [2080, 320];
            nReconOut.title = "Reconstruction Output";
            nReconOut.properties.activation = "linear";
            this.graph.add(nReconOut);

            // Connections
            nInput.connect(0, nFlat, 0);
            nFlat.connect(0, nEnc1, 0);
            nEnc1.connect(0, nEnc2, 0);
            nEnc2.connect(0, nBottle, 0);

            nBottle.connect(0, nClust, 0);
            nClust.connect(0, nClustOut, 0);

            nBottle.connect(0, nDec1, 0);
            nDec1.connect(0, nDec2, 0);
            nDec2.connect(0, nDecOut, 0);
            nDecOut.connect(0, nReshape, 0);
            nReshape.connect(0, nReconOut, 0);

        } else if (type === "hybrid_cnn_tabular") {
            // Multi-Branch 1D-CNN + Tabular Input merging via Concatenate
            const nSigIn = LiteGraph.createNode("keras/input");
            nSigIn.pos = [80, 140];
            nSigIn.title = "Signal Waveform Input";
            nSigIn.properties.shape = `${seqLen}, 1`;
            this.graph.add(nSigIn);

            const nConv1 = LiteGraph.createNode("keras/conv1d");
            nConv1.pos = [300, 140];
            nConv1.title = "Signal Conv1D";
            nConv1.properties.filters = baseFilters;
            nConv1.properties.kernel_size = 5;
            nConv1.properties.activation = act;
            this.graph.add(nConv1);

            const nPool1 = LiteGraph.createNode("keras/maxpooling1d");
            nPool1.pos = [520, 140];
            nPool1.properties.pool_size = 2;
            this.graph.add(nPool1);

            const nConv2 = LiteGraph.createNode("keras/conv1d");
            nConv2.pos = [740, 140];
            nConv2.title = "Signal Conv1D 2";
            nConv2.properties.filters = baseFilters * 2;
            nConv2.properties.kernel_size = 3;
            nConv2.properties.activation = act;
            this.graph.add(nConv2);

            const nPool2 = LiteGraph.createNode("keras/global_averagepooling1d");
            nPool2.pos = [960, 140];
            this.graph.add(nPool2);

            const nSigDense = LiteGraph.createNode("keras/dense");
            nSigDense.pos = [1180, 140];
            nSigDense.title = "Signal Embeddings";
            nSigDense.properties.units = 32;
            nSigDense.properties.activation = act;
            this.graph.add(nSigDense);

            // Tabular Stream
            const nTabIn = LiteGraph.createNode("keras/input");
            nTabIn.pos = [80, 360];
            nTabIn.title = "Tabular Features Input";
            nTabIn.properties.shape = "16";
            this.graph.add(nTabIn);

            const nTabDense1 = LiteGraph.createNode("keras/dense");
            nTabDense1.pos = [320, 360];
            nTabDense1.title = "Tabular Dense 1";
            nTabDense1.properties.units = 32;
            nTabDense1.properties.activation = act;
            this.graph.add(nTabDense1);

            const nTabDrop = LiteGraph.createNode("keras/dropout");
            nTabDrop.pos = [540, 360];
            nTabDrop.properties.rate = 0.2;
            this.graph.add(nTabDrop);

            const nTabDense2 = LiteGraph.createNode("keras/dense");
            nTabDense2.pos = [760, 360];
            nTabDense2.title = "Tabular Embeddings";
            nTabDense2.properties.units = 16;
            nTabDense2.properties.activation = act;
            this.graph.add(nTabDense2);

            // Merge Stream
            const nConcat = LiteGraph.createNode("keras/concatenate");
            nConcat.pos = [1420, 240];
            nConcat.title = "Merge (Signal + Tabular)";
            nConcat.properties.axis = -1;
            this.graph.add(nConcat);

            const nFusion = LiteGraph.createNode("keras/dense");
            nFusion.pos = [1640, 240];
            nFusion.title = "Fusion Dense";
            nFusion.properties.units = 64;
            nFusion.properties.activation = act;
            this.graph.add(nFusion);

            const nDropFinal = LiteGraph.createNode("keras/dropout");
            nDropFinal.pos = [1860, 240];
            nDropFinal.properties.rate = 0.3;
            this.graph.add(nDropFinal);

            const nFinal = LiteGraph.createNode("keras/dense");
            nFinal.pos = [2080, 240];
            nFinal.title = "Classifier Head";
            nFinal.properties.units = numClasses || 4;
            nFinal.properties.activation = "softmax";
            this.graph.add(nFinal);

            const nOutput = LiteGraph.createNode("keras/output");
            nOutput.pos = [2300, 240];
            nOutput.title = "Classification Output";
            nOutput.properties.activation = "softmax";
            this.graph.add(nOutput);

            // Connect Signal Stream
            nSigIn.connect(0, nConv1, 0);
            nConv1.connect(0, nPool1, 0);
            nPool1.connect(0, nConv2, 0);
            nConv2.connect(0, nPool2, 0);
            nPool2.connect(0, nSigDense, 0);

            // Connect Tabular Stream
            nTabIn.connect(0, nTabDense1, 0);
            nTabDense1.connect(0, nTabDrop, 0);
            nTabDrop.connect(0, nTabDense2, 0);

            // Connect into Merge Concatenate
            nSigDense.connect(0, nConcat, 0);
            nTabDense2.connect(0, nConcat, 1);

            // Connect Head
            nConcat.connect(0, nFusion, 0);
            nFusion.connect(0, nDropFinal, 0);
            nDropFinal.connect(0, nFinal, 0);
            nFinal.connect(0, nOutput, 0);
        }

        setTimeout(() => {
            this.validateGraph();
        }, 200);
    }
}

window.NNEditor = NNEditor;
