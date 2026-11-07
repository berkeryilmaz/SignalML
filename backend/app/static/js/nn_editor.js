// Neural Network Visual Node Editor (LiteGraph integration for Keras Functional API)

// ==========================================================================
// LiteGraph Extensions: 4-Way Node Flow (LR, RL, TB, BT) & Real-time Grid Snapping
// ==========================================================================
(function() {
    if (typeof LiteGraph === "undefined") return;

    // Synchronize LiteGraph canvas grid size to 20px (matching background dot pattern)
    LiteGraph.CANVAS_GRID_SIZE = 20;

    // 1. Enhanced Connection / Port Positioning supporting LR, RL, TB, BT
    LGraphNode.prototype.getConnectionPos = function(is_input, slot_number, out) {
        out = out || new Float32Array(2);
        const flow = this.flowMode || (this.graph && this.graph.flowMode) || "lr";
        const numInputs = (this.inputs && this.inputs.length) || 1;
        const numOutputs = (this.outputs && this.outputs.length) || 1;
        const slotH = LiteGraph.NODE_SLOT_HEIGHT || 20;
        const startY = this.constructor.slot_start_y || 0;
        const f = 0.5 * slotH; // 10px offset for side ports
        const titleH = LiteGraph.NODE_TITLE_HEIGHT || 30;

        if (slot_number === -1) {
            out[0] = this.pos[0] + 0.5 * titleH;
            out[1] = this.pos[1] + 0.5 * titleH;
            return out;
        }

        if (is_input && this.inputs && this.inputs[slot_number] && this.inputs[slot_number].pos) {
            out[0] = this.pos[0] + this.inputs[slot_number].pos[0];
            out[1] = this.pos[1] + this.inputs[slot_number].pos[1];
            return out;
        }
        if (!is_input && this.outputs && this.outputs[slot_number] && this.outputs[slot_number].pos) {
            out[0] = this.pos[0] + this.outputs[slot_number].pos[0];
            out[1] = this.pos[1] + this.outputs[slot_number].pos[1];
            return out;
        }

        if (this.flags && this.flags.collapsed) {
            const w = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH || 80;
            if (flow === "tb" || flow === "bt") {
                out[0] = this.pos[0] + 0.5 * w;
                out[1] = (flow === "tb")
                    ? (is_input ? this.pos[1] - titleH : this.pos[1])
                    : (is_input ? this.pos[1] : this.pos[1] - titleH);
            } else if (flow === "rl") {
                out[0] = is_input ? this.pos[0] + w : this.pos[0];
                out[1] = this.pos[1] - 0.5 * titleH;
            } else {
                out[0] = is_input ? this.pos[0] : this.pos[0] + w;
                out[1] = this.pos[1] - 0.5 * titleH;
            }
            return out;
        }

        if (flow === "rl") {
            // Right-to-Left: Inputs on Right edge, Outputs on Left edge
            if (is_input) {
                out[0] = this.pos[0] + this.size[0] + 1 - f;
                out[1] = this.pos[1] + (slot_number + 0.7) * slotH + startY;
            } else {
                out[0] = this.pos[0] + f;
                out[1] = this.pos[1] + (slot_number + 0.7) * slotH + startY;
            }
        } else if (flow === "tb") {
            // Top-to-Bottom: Inputs on Top edge (below title bar), Outputs on Bottom edge
            if (is_input) {
                out[0] = this.pos[0] + (this.size[0] / (numInputs + 1)) * (slot_number + 1);
                out[1] = this.pos[1];
            } else {
                out[0] = this.pos[0] + (this.size[0] / (numOutputs + 1)) * (slot_number + 1);
                out[1] = this.pos[1] + this.size[1];
            }
        } else if (flow === "bt") {
            // Bottom-to-Top: Inputs on Bottom edge, Outputs on Top edge
            if (is_input) {
                out[0] = this.pos[0] + (this.size[0] / (numInputs + 1)) * (slot_number + 1);
                out[1] = this.pos[1] + this.size[1];
            } else {
                out[0] = this.pos[0] + (this.size[0] / (numOutputs + 1)) * (slot_number + 1);
                out[1] = this.pos[1];
            }
        } else {
            // Left-to-Right (Default): Inputs on Left edge, Outputs on Right edge
            if (is_input) {
                out[0] = this.pos[0] + f;
                out[1] = this.pos[1] + (slot_number + 0.7) * slotH + startY;
            } else {
                out[0] = this.pos[0] + this.size[0] + 1 - f;
                out[1] = this.pos[1] + (slot_number + 0.7) * slotH + startY;
            }
        }
        return out;
    };

    // 2. Symmetric Radial Hit-Testing for Input Connection Dropping
    if (typeof LGraphCanvas !== "undefined") {
        LGraphCanvas.prototype.isOverNodeInput = function(node, canvasX, canvasY, out) {
            if (node.inputs) {
                for (let i = 0; i < node.inputs.length; ++i) {
                    const pos = node.getConnectionPos(true, i);
                    const dx = canvasX - pos[0];
                    const dy = canvasY - pos[1];
                    // 20px hit-radius around slot center (symmetrical across all 4 flow directions)
                    if (dx * dx + dy * dy <= 400) {
                        if (out) { out[0] = pos[0]; out[1] = pos[1]; }
                        return i;
                    }
                }
            }
            return -1;
        };
    }

    // 3. Symmetric Radial Hit-Testing for Slot Interaction
    LGraphNode.prototype.getSlotInPosition = function(canvasX, canvasY) {
        const pos = new Float32Array(2);
        if (this.inputs) {
            for (let i = 0; i < this.inputs.length; ++i) {
                this.getConnectionPos(true, i, pos);
                const dx = canvasX - pos[0];
                const dy = canvasY - pos[1];
                if (dx * dx + dy * dy <= 400) {
                    return { input: this.inputs[i], slot: i, link_pos: pos };
                }
            }
        }
        if (this.outputs) {
            for (let i = 0; i < this.outputs.length; ++i) {
                this.getConnectionPos(false, i, pos);
                const dx = canvasX - pos[0];
                const dy = canvasY - pos[1];
                if (dx * dx + dy * dy <= 400) {
                    return { output: this.outputs[i], slot: i, link_pos: pos };
                }
            }
        }
        return null;
    };

    // 4. Direction-Aware Node & Slot Label Drawing
    if (typeof LGraphCanvas !== "undefined") {
        LGraphCanvas.prototype.drawNode = function(node, ctx) {
            this.current_node = node;
            const color = node.color || node.constructor.color || LiteGraph.NODE_DEFAULT_COLOR;
            let bgcolor = node.bgcolor || node.constructor.bgcolor || LiteGraph.NODE_DEFAULT_BGCOLOR;
            const lowScale = this.ds.scale < 0.6;

            if (this.live_mode) {
                if (!node.flags.collapsed && node.onDrawForeground) {
                    ctx.shadowColor = "transparent";
                    node.onDrawForeground(ctx, this, this.canvas);
                }
                return;
            }

            const alpha = this.editor_alpha;
            ctx.globalAlpha = alpha;
            if (this.render_shadows && !lowScale) {
                ctx.shadowColor = LiteGraph.DEFAULT_SHADOW_COLOR;
                ctx.shadowOffsetX = 2 * this.ds.scale;
                ctx.shadowOffsetY = 2 * this.ds.scale;
                ctx.shadowBlur = 3 * this.ds.scale;
            } else {
                ctx.shadowColor = "transparent";
            }

            if (node.flags.collapsed && node.onDrawCollapsed && node.onDrawCollapsed(ctx, this)) {
                return;
            }

            const shape = node._shape || LiteGraph.BOX_SHAPE;
            const size = [node.size[0], node.size[1]];
            const flow = node.flowMode || (node.graph && node.graph.flowMode) || "lr";

            if (node.flags.collapsed) {
                ctx.font = this.inner_text_font;
                const title = node.getTitle ? node.getTitle() : node.title;
                if (title != null) {
                    node._collapsed_width = Math.min(node.size[0], ctx.measureText(title).width + 2 * (LiteGraph.NODE_TITLE_HEIGHT || 30));
                    size[0] = node._collapsed_width;
                    size[1] = 0;
                }
            }

            if (node.clip_area) {
                ctx.save();
                ctx.beginPath();
                if (shape === LiteGraph.BOX_SHAPE) ctx.rect(0, 0, size[0], size[1]);
                else if (shape === LiteGraph.ROUND_SHAPE) ctx.roundRect ? ctx.roundRect(0, 0, size[0], size[1], 10) : ctx.rect(0, 0, size[0], size[1]);
                else if (shape === LiteGraph.CIRCLE_SHAPE) ctx.arc(0.5 * size[0], 0.5 * size[1], 0.5 * size[0], 0, 2 * Math.PI);
                ctx.clip();
            }

            if (node.has_errors) bgcolor = "red";
            this.drawNodeShape(node, ctx, size, color, bgcolor, node.is_selected, node.mouseOver);
            ctx.shadowColor = "transparent";

            if (node.onDrawForeground) {
                node.onDrawForeground(ctx, this, this.canvas);
            }

            ctx.font = this.inner_text_font;
            const showLabels = !lowScale;
            const connectingOutput = this.connecting_output;
            ctx.lineWidth = 1;
            const posHelper = new Float32Array(2);

            if (!node.flags.collapsed) {
                // Draw Inputs
                if (node.inputs) {
                    for (let i = 0; i < node.inputs.length; i++) {
                        const slot = node.inputs[i];
                        ctx.globalAlpha = alpha;
                        if (this.connecting_node && !LiteGraph.isValidConnection(slot.type, connectingOutput && connectingOutput.type)) {
                            ctx.globalAlpha = 0.4 * alpha;
                        }
                        ctx.fillStyle = (slot.link != null)
                            ? (slot.color_on || this.default_connection_color.input_on)
                            : (slot.color_off || this.default_connection_color.input_off);

                        const slotPos = node.getConnectionPos(true, i, posHelper);
                        const localX = slotPos[0] - node.pos[0];
                        const localY = slotPos[1] - node.pos[1];

                        ctx.beginPath();
                        if (lowScale) {
                            ctx.rect(localX - 4, localY - 4, 8, 8);
                        } else {
                            ctx.arc(localX, localY, 4.5, 0, 2 * Math.PI);
                        }
                        ctx.fill();

                        if (showLabels) {
                            const label = slot.label != null ? slot.label : slot.name;
                            if (label) {
                                ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
                                if (flow === "rl") {
                                    ctx.textAlign = "right";
                                    ctx.fillText(label, localX - 10, localY + 4);
                                } else if (flow === "tb") {
                                    ctx.textAlign = "center";
                                    ctx.fillText(label, localX, localY + 16);
                                } else if (flow === "bt") {
                                    ctx.textAlign = "center";
                                    ctx.fillText(label, localX, localY - 8);
                                } else { // "lr"
                                    ctx.textAlign = "left";
                                    ctx.fillText(label, localX + 10, localY + 4);
                                }
                            }
                        }
                    }
                }

                // Draw Outputs
                if (this.connecting_node) {
                    ctx.globalAlpha = 0.4 * alpha;
                }
                ctx.strokeStyle = "black";
                if (node.outputs) {
                    for (let i = 0; i < node.outputs.length; i++) {
                        const slot = node.outputs[i];
                        ctx.fillStyle = (slot.links && slot.links.length)
                            ? (slot.color_on || this.default_connection_color.output_on)
                            : (slot.color_off || this.default_connection_color.output_off);

                        const slotPos = node.getConnectionPos(false, i, posHelper);
                        const localX = slotPos[0] - node.pos[0];
                        const localY = slotPos[1] - node.pos[1];

                        ctx.beginPath();
                        if (lowScale) {
                            ctx.rect(localX - 4, localY - 4, 8, 8);
                        } else {
                            ctx.arc(localX, localY, 4.5, 0, 2 * Math.PI);
                        }
                        ctx.fill();
                        if (!lowScale) ctx.stroke();

                        if (showLabels) {
                            const label = slot.label != null ? slot.label : slot.name;
                            if (label) {
                                ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
                                if (flow === "rl") {
                                    ctx.textAlign = "left";
                                    ctx.fillText(label, localX + 10, localY + 4);
                                } else if (flow === "tb") {
                                    ctx.textAlign = "center";
                                    ctx.fillText(label, localX, localY - 8);
                                } else if (flow === "bt") {
                                    ctx.textAlign = "center";
                                    ctx.fillText(label, localX, localY + 16);
                                } else { // "lr"
                                    ctx.textAlign = "right";
                                    ctx.fillText(label, localX - 10, localY + 4);
                                }
                            }
                        }
                    }
                }

                ctx.textAlign = "left";
                ctx.globalAlpha = 1;
                if (node.widgets) {
                    let startY = 2;
                    if (node.widgets_start_y != null) startY = node.widgets_start_y;
                    this.drawNodeWidgets(node, startY, ctx, this.node_widget && this.node_widget[0] === node ? this.node_widget[1] : null);
                }
            } else if (this.render_collapsed_slots) {
                const w = node._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH || 80;
                const titleH = LiteGraph.NODE_TITLE_HEIGHT || 30;
                let hasConnectedInput = false;
                let hasConnectedOutput = false;
                if (node.inputs) {
                    for (let d = 0; d < node.inputs.length; d++) {
                        if (node.inputs[d].link != null) { hasConnectedInput = true; break; }
                    }
                }
                if (node.outputs) {
                    for (let d = 0; d < node.outputs.length; d++) {
                        if (node.outputs[d].links && node.outputs[d].links.length) { hasConnectedOutput = true; break; }
                    }
                }

                ctx.fillStyle = (this.default_connection_color && this.default_connection_color.input_on) || "#0090ff";
                if (hasConnectedInput) {
                    let ix, iy;
                    if (flow === "tb") { ix = 0.5 * w; iy = -titleH; }
                    else if (flow === "bt") { ix = 0.5 * w; iy = 0; }
                    else if (flow === "rl") { ix = w; iy = -0.5 * titleH; }
                    else { ix = 0; iy = -0.5 * titleH; }

                    ctx.beginPath();
                    ctx.arc(ix, iy, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }

                ctx.fillStyle = (this.default_connection_color && this.default_connection_color.output_on) || "#0090ff";
                if (hasConnectedOutput) {
                    let ox, oy;
                    if (flow === "tb") { ox = 0.5 * w; oy = 0; }
                    else if (flow === "bt") { ox = 0.5 * w; oy = -titleH; }
                    else if (flow === "rl") { ox = 0; oy = -0.5 * titleH; }
                    else { ox = w; oy = -0.5 * titleH; }

                    ctx.beginPath();
                    ctx.arc(ox, oy, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }

            if (node.clip_area) ctx.restore();
            ctx.globalAlpha = 1;
        };

        // 5. Real-Time Grid Snapping on Node Drag
        const origProcessMouseDown = LGraphCanvas.prototype.processMouseDown;
        LGraphCanvas.prototype.processMouseDown = function(e) {
            const res = origProcessMouseDown ? origProcessMouseDown.apply(this, arguments) : undefined;
            if (this.node_dragged) {
                this._dragStartMouse = [e.canvasX, e.canvasY];
                this._dragStartPosMap = new Map();
                if (this.selected_nodes) {
                    for (const id in this.selected_nodes) {
                        const n = this.selected_nodes[id];
                        if (n && n.pos) {
                            this._dragStartPosMap.set(n.id, [n.pos[0], n.pos[1]]);
                        }
                    }
                }
                if (this.node_dragged && !this._dragStartPosMap.has(this.node_dragged.id)) {
                    this._dragStartPosMap.set(this.node_dragged.id, [this.node_dragged.pos[0], this.node_dragged.pos[1]]);
                }
            }
            return res;
        };

        const origProcessMouseMove = LGraphCanvas.prototype.processMouseMove;
        LGraphCanvas.prototype.processMouseMove = function(e) {
            const res = origProcessMouseMove ? origProcessMouseMove.apply(this, arguments) : undefined;
            if (this.snap_to_grid && this.node_dragged && this._dragStartMouse && this._dragStartPosMap) {
                const grid = this.snap_grid_size || LiteGraph.CANVAS_GRID_SIZE || 20;
                const totalDx = e.canvasX - this._dragStartMouse[0];
                const totalDy = e.canvasY - this._dragStartMouse[1];
                const snappedDx = Math.round(totalDx / grid) * grid;
                const snappedDy = Math.round(totalDy / grid) * grid;

                this._dragStartPosMap.forEach((origPos, nId) => {
                    const n = (this.graph && this.graph._nodes_by_id) ? this.graph._nodes_by_id[nId] : null;
                    if (n && n.pos) {
                        n.pos[0] = Math.round((origPos[0] + snappedDx) / grid) * grid;
                        n.pos[1] = Math.round((origPos[1] + snappedDy) / grid) * grid;
                    }
                });
                this.dirty_bgcanvas = this.dirty_canvas = true;
            }
            return res;
        };

        const origProcessMouseUp = LGraphCanvas.prototype.processMouseUp;
        LGraphCanvas.prototype.processMouseUp = function(e) {
            if (this.snap_to_grid && this._dragStartPosMap) {
                const grid = this.snap_grid_size || LiteGraph.CANVAS_GRID_SIZE || 20;
                this._dragStartPosMap.forEach((_, nId) => {
                    const n = (this.graph && this.graph._nodes_by_id) ? this.graph._nodes_by_id[nId] : null;
                    if (n && n.pos) {
                        n.pos[0] = Math.round(n.pos[0] / grid) * grid;
                        n.pos[1] = Math.round(n.pos[1] / grid) * grid;
                    }
                });
            }
            this._dragStartMouse = null;
            this._dragStartPosMap = null;
            return origProcessMouseUp ? origProcessMouseUp.apply(this, arguments) : undefined;
        };
    }
})();

class NNEditor {
    constructor(canvasId) {
        this.canvasId = canvasId;
        this.graph = new LGraph();
        this.canvas = null;
        this.selectedNode = null;
        this.flowMode = "lr"; // Default flow mode: Left-to-Right
        this.snapToGrid = true; // Default: Snap to Grid enabled
        this.snapGridSize = 20; // 20px grid matches canvas background dot pattern
        this.init();
        window.nnEditorInstance = this;
    }

    init() {
        const canvasElement = document.getElementById(this.canvasId);
        if (!canvasElement) return;

        // Resize canvas to container
        canvasElement.width = canvasElement.parentElement.clientWidth;
        canvasElement.height = canvasElement.parentElement.clientHeight;

        this.canvas = new LGraphCanvas(canvasElement, this.graph);
        this.canvas.snap_to_grid = this.snapToGrid;
        this.canvas.align_to_grid = this.snapToGrid;
        this.canvas.snap_grid_size = this.snapGridSize;

        if (window.SignalML && typeof window.SignalML.applyGraphTheme === "function") {
            window.SignalML.applyGraphTheme(this.canvas);
        }

        // Snap newly added nodes to 20px grid and inherit active flow
        this.graph.onNodeAdded = (node) => {
            if (node && node.pos) {
                const grid = this.snapGridSize || 20;
                node.pos[0] = Math.round(node.pos[0] / grid) * grid;
                node.pos[1] = Math.round(node.pos[1] / grid) * grid;
            }
            this.applyNodeSlotDirections(node, node.flowMode || this.flowMode);
        };

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

        this.updateFlowToolbarUI(this.flowMode);
        this.updateSnapToolbarUI();

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
        if (typeof LiteGraph !== "undefined" && typeof LiteGraph.clearRegisteredTypes === "function") {
            LiteGraph.clearRegisteredTypes();
        }

        const createLayerConstructor = (title, inputs, outputs, properties) => {
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
            return KerasNode;
        };

        const registerLayer = (type, title, inputs, outputs, properties, aliases = [], category = "Core & IO") => {
            const PrimaryNode = createLayerConstructor(title, inputs, outputs, properties);
            LiteGraph.registerNodeType(`keras/${type}`, PrimaryNode);
            PrimaryNode.type = `keras/${type}`;
            PrimaryNode.category = category;
            PrimaryNode.skip_list = false;

            const allAliases = [type, ...aliases, ...aliases.map(a => `keras/${a}`)];
            for (const alias of allAliases) {
                if (alias === `keras/${type}`) continue;
                const AliasNode = createLayerConstructor(title, inputs, outputs, properties);
                LiteGraph.registerNodeType(alias, AliasNode);
                AliasNode.type = `keras/${type}`;
                AliasNode.category = null;
                AliasNode.skip_list = true;
            }
        };

        // 1. Core & IO
        registerLayer("input", "Input", [], [{ name: "tensor", type: "Tensor" }], { shape: "512, 1" }, ["inputlayer"], "Core & IO");
        registerLayer("output", "Model Output", [{ name: "in", type: "Tensor" }], [], { activation: "linear" }, ["modeloutput"], "Core & IO");
        registerLayer("dense", "Dense", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, activation: "relu" }, [], "Core & IO");
        registerLayer("activation", "Activation", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { activation: "relu" }, [], "Core & IO");

        // 2. 1D Convolution
        registerLayer("conv1d", "Conv1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 5, strides: 1, padding: "same", activation: "relu" }, [], "1D Convolution");
        registerLayer("conv1dtranspose", "Conv1DTranspose", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 5, strides: 2, padding: "same", activation: "relu" }, ["deconv1d", "conv1d_transpose"], "1D Convolution");
        registerLayer("upsampling1d", "UpSampling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { size: 2 }, [], "1D Convolution");
        registerLayer("separable_conv1d", "SeparableConv1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 5, padding: "same", activation: "relu" }, [], "1D Convolution");
        registerLayer("conv2d", "Conv2D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { filters: 32, kernel_size: 3, padding: "same", activation: "relu" }, [], "1D Convolution");
        registerLayer("maxpooling1d", "MaxPooling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { pool_size: 2, padding: "same" }, ["max_pooling1d", "maxpool1d"], "1D Convolution");
        registerLayer("averagepooling1d", "AveragePooling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { pool_size: 2, padding: "same" }, ["avgpooling1d", "avg_pooling1d", "avgpool1d"], "1D Convolution");
        registerLayer("global_maxpooling1d", "GlobalMaxPooling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["globalmaxpooling1d"], "1D Convolution");
        registerLayer("global_averagepooling1d", "GlobalAveragePooling1D", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["globalaveragepooling1d", "globalavgpooling1d"], "1D Convolution");

        // 3. Recurrent & TimeDistributed
        registerLayer("lstm", "LSTM", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, return_sequences: false, dropout: 0.0 }, [], "Recurrent");
        registerLayer("gru", "GRU", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, return_sequences: false }, [], "Recurrent");
        registerLayer("simplernn", "SimpleRNN", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, return_sequences: false, activation: "tanh" }, [], "Recurrent");
        registerLayer("bidirectional", "Bidirectional", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { units: 64, recurrent_type: "lstm", return_sequences: false }, [], "Recurrent");
        registerLayer("timedistributed", "TimeDistributed", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { layer_type: "Dense", units: 32 }, ["time_distributed"], "Recurrent");

        // 4. Attention & Transformer
        registerLayer("multihead_attention", "MultiHeadAttention", [{ name: "query", type: "Tensor" }, { name: "value", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { num_heads: 4, key_dim: 32, dropout: 0.1 }, ["multiheadattention"], "Attention & Transformer");
        registerLayer("positional_encoding", "PositionalEncoding", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { d_model: 64, max_len: 2048 }, ["positionalencoding"], "Attention & Transformer");

        // 5. Reshape & Regularization
        registerLayer("flatten", "Flatten", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, [], "Reshape & Regularization");
        registerLayer("reshape", "Reshape", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { target_shape: "-1, 1" }, [], "Reshape & Regularization");
        registerLayer("dropout", "Dropout", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { rate: 0.2 }, [], "Reshape & Regularization");
        registerLayer("batchnorm", "BatchNormalization", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["batchnormalization", "bn"], "Reshape & Regularization");
        registerLayer("layernorm", "LayerNormalization", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["layernormalization", "ln"], "Reshape & Regularization");
        registerLayer("unitnorm", "UnitNormalization", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, ["unitnormalization"], "Reshape & Regularization");

        // 6. Merge & Arithmetic
        registerLayer("concatenate", "Concatenate", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { axis: -1 }, [], "Merge & Arithmetic");
        registerLayer("add", "Add", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, [], "Merge & Arithmetic");
        registerLayer("multiply", "Multiply", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, [], "Merge & Arithmetic");
        registerLayer("average", "Average", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], {}, [], "Merge & Arithmetic");
        registerLayer("dot", "Dot", [{ name: "in1", type: "Tensor" }, { name: "in2", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { axes: -1, normalize: false }, [], "Merge & Arithmetic");

        // 7. Clustering & Custom Layers
        registerLayer("clustering", "ClusteringLayer (IDEC)", [{ name: "in", type: "Tensor" }], [{ name: "prob", type: "Tensor" }], { n_clusters: 8, alpha: 1.0 }, ["clusteringlayer"], "Clustering & Custom");
        registerLayer("custom_layer", "Custom Layer", [{ name: "in", type: "Tensor" }], [{ name: "out", type: "Tensor" }], { class_name: "CustomSignalLayer", units: 32, activation: "relu", code: "" }, ["customlayer", "customsignallayer"], "Clustering & Custom");
    }

    applyNodeSlotDirections(node, flow) {
        if (!node) return;
        const mode = flow || node.flowMode || (node.graph && node.graph.flowMode) || this.flowMode || "lr";
        if (mode === "rl") {
            if (node.inputs) node.inputs.forEach(s => { s.dir = LiteGraph.RIGHT; });
            if (node.outputs) node.outputs.forEach(s => { s.dir = LiteGraph.LEFT; });
        } else if (mode === "tb") {
            if (node.inputs) node.inputs.forEach(s => { s.dir = LiteGraph.UP; });
            if (node.outputs) node.outputs.forEach(s => { s.dir = LiteGraph.DOWN; });
        } else if (mode === "bt") {
            if (node.inputs) node.inputs.forEach(s => { s.dir = LiteGraph.DOWN; });
            if (node.outputs) node.outputs.forEach(s => { s.dir = LiteGraph.UP; });
        } else { // "lr"
            if (node.inputs) node.inputs.forEach(s => { s.dir = LiteGraph.LEFT; });
            if (node.outputs) node.outputs.forEach(s => { s.dir = LiteGraph.RIGHT; });
        }

        // Auto-expand node width for multi-port layers in vertical flows so port labels don't collide
        if (mode === "tb" || mode === "bt") {
            const numInputs = (node.inputs && node.inputs.length) || 1;
            const numOutputs = (node.outputs && node.outputs.length) || 1;
            const maxPorts = Math.max(numInputs, numOutputs);
            const minW = Math.max(180, (maxPorts + 1) * 45);
            if (node.size && node.size[0] < minW) {
                node.size[0] = minW;
            }
        }
    }

    setFlowMode(mode, triggerAutoLayout = true) {
        const validModes = ["lr", "rl", "tb", "bt"];
        if (!validModes.includes(mode)) mode = "lr";

        this.flowMode = mode;
        if (this.graph) {
            this.graph.flowMode = mode;
            if (this.graph._nodes) {
                this.graph._nodes.forEach(node => {
                    this.applyNodeSlotDirections(node, node.flowMode || mode);
                });
            }
        }

        if (triggerAutoLayout) {
            this.autoLayout(mode);
        }

        this.updateFlowToolbarUI(mode);

        if (this.selectedNode) {
            this.updateInspector(this.selectedNode);
        }

        if (this.canvas) {
            this.canvas.setDirty(true, true);
            if (typeof this.canvas.draw === "function") {
                this.canvas.draw(true, true);
            }
        }

        const flowLabels = {
            lr: "Left → Right",
            rl: "Right ← Left",
            tb: "Top ↓ Bottom",
            bt: "Bottom ↑ Top"
        };
        this.toast(`Graph flow: ${flowLabels[mode] || mode}`, "info");
    }

    setNodeOrientation(nodeId, orientation) {
        if (!this.graph) return;
        const node = typeof nodeId === "object" ? nodeId : this.graph.getNodeById(nodeId);
        if (!node) return;

        if (orientation === null || orientation === "default" || orientation === "inherit") {
            delete node.flowMode;
        } else {
            node.flowMode = orientation;
        }

        this.applyNodeSlotDirections(node, node.flowMode || this.flowMode || "lr");

        if (this.canvas) {
            this.canvas.setDirty(true, true);
            if (typeof this.canvas.draw === "function") {
                this.canvas.draw(true, true);
            }
        }

        this.updateInspector(node);

        const currentActive = node.flowMode || this.flowMode || "lr";
        const flowLabels = { lr: "L→R", rl: "R←L", tb: "T↓B", bt: "B↑T" };
        const label = node.flowMode
            ? `${flowLabels[node.flowMode] || node.flowMode}`
            : `Graph Default (${flowLabels[this.flowMode] || this.flowMode})`;
        this.toast(`${node.title || "Layer"} ports: ${label}`, "info");
    }

    cycleNodeOrientation(nodeId) {
        if (!this.graph) return;
        const node = typeof nodeId === "object" ? nodeId : this.graph.getNodeById(nodeId);
        if (!node) return;

        const modes = ["lr", "rl", "tb", "bt"];
        const current = node.flowMode || this.flowMode || "lr";
        const nextIdx = (modes.indexOf(current) + 1) % modes.length;
        this.setNodeOrientation(node, modes[nextIdx]);
    }

    toggleSnap() {
        this.snapToGrid = !this.snapToGrid;
        if (this.canvas) {
            this.canvas.snap_to_grid = this.snapToGrid;
            this.canvas.align_to_grid = this.snapToGrid;
        }
        this.updateSnapToolbarUI();
        this.toast(`Snap to grid: ${this.snapToGrid ? "Enabled (20px)" : "Disabled (Free)"}`, "info");
    }

    updateFlowToolbarUI(mode) {
        const current = mode || this.flowMode || "lr";
        ["lr", "rl", "tb", "bt"].forEach(m => {
            const btn = document.getElementById(`flow-btn-${m}`);
            if (btn) {
                btn.classList.toggle("is-active", m === current);
                btn.setAttribute("aria-pressed", m === current ? "true" : "false");
            }
        });
    }

    updateSnapToolbarUI() {
        const btn = document.getElementById("toolbar-snap-btn");
        if (btn) {
            btn.classList.toggle("is-active", Boolean(this.snapToGrid));
            btn.setAttribute("aria-pressed", this.snapToGrid ? "true" : "false");
            const span = btn.querySelector("span");
            if (span) {
                span.textContent = this.snapToGrid ? "Snap: On" : "Snap: Off";
            }
        }
    }

    setupDefaultGraph() {
        this.flowMode = "lr";
        if (this.graph) this.graph.flowMode = "lr";

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

        if (this.graph && this.graph._nodes) {
            this.graph._nodes.forEach(n => this.applyNodeSlotDirections(n, "lr"));
        }
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

        // Add Model Wizard & Graph Flow Options to Canvas Right-Click Context Menu
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
            const flowSubmenu = {
                content: "Graph Flow Layout",
                has_submenu: true,
                submenu: {
                    options: [
                        { content: "Left to Right (L→R)", callback: () => this.setFlowMode("lr") },
                        { content: "Right to Left (R←L)", callback: () => this.setFlowMode("rl") },
                        { content: "Top to Bottom (T↓B)", callback: () => this.setFlowMode("tb") },
                        { content: "Bottom to Top (B↑T)", callback: () => this.setFlowMode("bt") }
                    ]
                }
            };
            const snapItem = {
                content: this.snapToGrid ? "Disable Snap to Grid" : "Enable Snap to Grid (20px)",
                callback: () => this.toggleSnap()
            };
            return [wizardItem, flowSubmenu, snapItem, null, ...(opts || [])];
        };

        // Add Port Orientation to Node Right-Click Context Menu
        const origGetNodeMenuOptions = this.canvas.getNodeMenuOptions;
        this.canvas.getNodeMenuOptions = (node) => {
            const opts = origGetNodeMenuOptions ? origGetNodeMenuOptions.call(this.canvas, node) : [];
            const flowSubmenu = {
                content: "Port Orientation",
                has_submenu: true,
                submenu: {
                    options: [
                        { content: "Left to Right (→ LR)", callback: () => this.setNodeOrientation(node.id, "lr") },
                        { content: "Right to Left (← RL)", callback: () => this.setNodeOrientation(node.id, "rl") },
                        { content: "Top to Bottom (↓ TB)", callback: () => this.setNodeOrientation(node.id, "tb") },
                        { content: "Bottom to Top (↑ BT)", callback: () => this.setNodeOrientation(node.id, "bt") },
                        null,
                        { content: "Inherit Graph Default", callback: () => this.setNodeOrientation(node.id, null) }
                    ]
                }
            };
            return [flowSubmenu, null, ...(opts || [])];
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

        const currentFlow = node.flowMode || this.flowMode || "lr";
        const flowLabels = { lr: "L→R", rl: "R←L", tb: "T↓B", bt: "B↑T" };
        const nodeFlowBadge = node.flowMode
            ? `${flowLabels[node.flowMode] || node.flowMode}`
            : `Graph (${flowLabels[this.flowMode] || this.flowMode})`;

        const orientationTileHtml = `
            <div class="tile mb-3">
                <div class="row-between mb-2">
                    <span class="text-xs text-muted fw-600 uppercase">Port Orientation</span>
                    <span class="badge badge-mono text-xs" id="inspector-node-flow-badge">${nodeFlowBadge}</span>
                </div>
                <div class="btn-group w-full mb-2">
                    <button type="button" class="btn btn-secondary btn-xs flex-1 ${currentFlow === 'lr' ? 'btn-primary' : ''}" data-flow-action="set" data-flow="lr" data-node-id="${node.id}" title="Left to Right (Inputs Left, Outputs Right)">&rarr; LR</button>
                    <button type="button" class="btn btn-secondary btn-xs flex-1 ${currentFlow === 'rl' ? 'btn-primary' : ''}" data-flow-action="set" data-flow="rl" data-node-id="${node.id}" title="Right to Left (Inputs Right, Outputs Left)">&larr; RL</button>
                    <button type="button" class="btn btn-secondary btn-xs flex-1 ${currentFlow === 'tb' ? 'btn-primary' : ''}" data-flow-action="set" data-flow="tb" data-node-id="${node.id}" title="Top to Bottom (Inputs Top, Outputs Bottom)">&darr; TB</button>
                    <button type="button" class="btn btn-secondary btn-xs flex-1 ${currentFlow === 'bt' ? 'btn-primary' : ''}" data-flow-action="set" data-flow="bt" data-node-id="${node.id}" title="Bottom to Top (Inputs Bottom, Outputs Top)">&uarr; BT</button>
                </div>
                <div class="row-between">
                    <button type="button" class="btn btn-ghost btn-xs text-muted" data-flow-action="cycle" data-node-id="${node.id}" title="Cycle orientation">
                        ${iconFn('rotate-cw', 'icon-xs')} <span>Cycle flow</span>
                    </button>
                    <button type="button" class="btn btn-ghost btn-xs text-muted" data-flow-action="reset" data-node-id="${node.id}" title="Inherit graph default flow">Reset default</button>
                </div>
            </div>
        `;

        let html = `
            <div class="mb-3">
                <div class="text-subhead">${node.title}</div>
                <div class="text-xs text-muted mono">Type: ${node.type} | ID: ${node.id}</div>
            </div>
            <div class="form-group tile mb-3">
                <label class="form-label">Layer name</label>
                <input class="form-control mono" id="inspector-custom-name" type="text" value="${safeCustomName}" placeholder="e.g. encoder_bottleneck">
            </div>
            ${orientationTileHtml}
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

        // Bind node port orientation action events
        body.querySelectorAll("[data-flow-action]").forEach(btn => {
            btn.addEventListener("click", () => {
                const action = btn.dataset.flowAction;
                const targetId = Number(btn.dataset.nodeId);
                if (action === "set") {
                    this.setNodeOrientation(targetId, btn.dataset.flow);
                } else if (action === "cycle") {
                    this.cycleNodeOrientation(targetId);
                } else if (action === "reset") {
                    this.setNodeOrientation(targetId, null);
                }
            });
        });

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

    autoLayout(direction) {
        if (!this.graph || !this.graph._nodes || this.graph._nodes.length === 0) {
            this.toast("No nodes to align", "warning");
            return;
        }

        const flow = direction || this.flowMode || "lr";
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

        // Check for unvisited nodes (e.g. if graph had an accidental cycle or unreached component)
        const unvisited = nodes.filter(n => (inDegree[n.id] || 0) > 0);
        if (unvisited.length > 0) {
            let nextD = 0;
            nodes.forEach(n => {
                if (depth[n.id] > nextD) nextD = depth[n.id];
            });
            nextD += 1;
            unvisited.forEach(n => {
                depth[n.id] = nextD++;
            });
        }

        // Group nodes by depth
        const depthGroups = {};
        let maxDepth = 0;
        nodes.forEach(n => {
            const d = depth[n.id] !== undefined ? depth[n.id] : 0;
            if (d > maxDepth) maxDepth = d;
            if (!depthGroups[d]) depthGroups[d] = [];
            depthGroups[d].push(n);
        });

        const colWidth = 260;
        const rowHeight = 160;
        const startX = 80;
        const startY = 140;
        const grid = (this.canvas && this.canvas.snap_grid_size) || 20;

        Object.keys(depthGroups).sort((a, b) => Number(a) - Number(b)).forEach(dStr => {
            const d = Number(dStr);
            const groupNodes = depthGroups[d];

            groupNodes.forEach((node, idx) => {
                let x, y;
                if (flow === "lr") {
                    x = startX + d * colWidth;
                    y = startY + idx * rowHeight;
                } else if (flow === "rl") {
                    x = startX + (maxDepth - d) * colWidth;
                    y = startY + idx * rowHeight;
                } else if (flow === "tb") {
                    x = startX + idx * colWidth;
                    y = startY + d * rowHeight;
                } else if (flow === "bt") {
                    x = startX + idx * colWidth;
                    y = startY + (maxDepth - d) * rowHeight;
                } else {
                    x = startX + d * colWidth;
                    y = startY + idx * rowHeight;
                }

                node.pos[0] = Math.round(x / grid) * grid;
                node.pos[1] = Math.round(y / grid) * grid;
            });
        });

        if (this.canvas) {
            this.canvas.setDirty(true, true);
            if (typeof this.canvas.draw === "function") {
                this.canvas.draw(true, true);
            }
        }

        const flowLabels = { lr: "Left → Right", rl: "Right ← Left", tb: "Top ↓ Bottom", bt: "Bottom ↑ Top" };
        this.toast(`Nodes aligned (${flowLabels[flow] || flow})`, "success");
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

        // Default to Left-to-Right for loaded models unless explicitly defined
        const loadedFlow = (graphData && (graphData.flowMode || graphData.flow_mode)) || "lr";
        this.flowMode = loadedFlow;
        if (this.graph) this.graph.flowMode = loadedFlow;

        if (graphData.nodes && Array.isArray(graphData.nodes)) {
            const grid = (this.canvas && this.canvas.snap_grid_size) || 20;
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
                // Snap loaded node positions to 20px grid
                n.pos[0] = Math.round(n.pos[0] / grid) * grid;
                n.pos[1] = Math.round(n.pos[1] / grid) * grid;

                if (!n.properties || typeof n.properties !== "object") {
                    n.properties = {};
                }
            }
        }

        this.graph.clear();
        this.graph.configure(graphData);

        if (this.graph && this.graph._nodes) {
            this.graph._nodes.forEach(node => {
                this.applyNodeSlotDirections(node, node.flowMode || loadedFlow);
            });
        }

        this.updateFlowToolbarUI(loadedFlow);

        if (this.canvas) {
            this.canvas.setDirty(true, true);
        }
        this.validateGraph();
    }

    serializeGraph() {
        if (!this.graph) return { nodes: [], links: [] };
        const serialized = this.graph.serialize();
        if (serialized) {
            serialized.flowMode = this.flowMode || "lr";
            if (serialized.nodes && this.graph._nodes) {
                serialized.nodes.forEach(n => {
                    const actualNode = this.graph.getNodeById(n.id);
                    if (actualNode) {
                        if (actualNode.title) n.title = actualNode.title;
                        if (actualNode.flowMode) n.flowMode = actualNode.flowMode;
                    }
                });
            }
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
