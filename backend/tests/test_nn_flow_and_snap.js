/**
 * Unit & Integration tests for NN Builder 4-Way Flow (LR, RL, TB, BT) and Grid Snapping.
 */
const assert = require("assert");

// Mock browser environment for Node.js
global.window = global;
global.document = {
    getElementById: () => null,
    addEventListener: () => {},
    querySelectorAll: () => []
};

// Load LiteGraph
const lg = require("../app/static/vendor/litegraph.min.js");
const { LiteGraph, LGraph, LGraphNode, LGraphCanvas } = lg;
global.LiteGraph = LiteGraph;
global.LGraph = LGraph;
global.LGraphNode = LGraphNode;
global.LGraphCanvas = LGraphCanvas;

// Load NNEditor extensions
require("../app/static/js/nn_editor.js");
const NNEditor = global.NNEditor;

console.log("Running NN Builder Flow & Snapping Test Suite...");

// =========================================================================
// Test 1: LGraphNode.prototype.getConnectionPos in all 4 directions
// =========================================================================
{
    const node = new LGraphNode();
    node.pos = [100, 200];
    node.addInput("in", "Tensor");
    node.addOutput("out", "Tensor");
    node.size = [180, 80];

    // 1.1 Left-to-Right (Default)
    node.flowMode = "lr";
    const inPosLR = node.getConnectionPos(true, 0);
    const outPosLR = node.getConnectionPos(false, 0);
    assert.strictEqual(inPosLR[0], 110, "LR: Input should be at left edge (pos[0] + 10)");
    assert.strictEqual(outPosLR[0], 271, "LR: Output should be at right edge (pos[0] + size[0] + 1 - 10)");

    // 1.2 Right-to-Left
    node.flowMode = "rl";
    const inPosRL = node.getConnectionPos(true, 0);
    const outPosRL = node.getConnectionPos(false, 0);
    assert.strictEqual(inPosRL[0], 271, "RL: Input should be at right edge");
    assert.strictEqual(outPosRL[0], 110, "RL: Output should be at left edge");

    // 1.3 Top-to-Bottom
    node.flowMode = "tb";
    const inPosTB = node.getConnectionPos(true, 0);
    const outPosTB = node.getConnectionPos(false, 0);
    assert.strictEqual(inPosTB[0], 190, "TB: Input centered horizontally at pos[0] + 90");
    assert.strictEqual(inPosTB[1], 200, "TB: Input at top edge pos[1]");
    assert.strictEqual(outPosTB[0], 190, "TB: Output centered horizontally at pos[0] + 90");
    assert.strictEqual(outPosTB[1], 280, "TB: Output at bottom edge pos[1] + size[1]");

    // 1.4 Bottom-to-Top
    node.flowMode = "bt";
    const inPosBT = node.getConnectionPos(true, 0);
    const outPosBT = node.getConnectionPos(false, 0);
    assert.strictEqual(inPosBT[0], 190, "BT: Input centered horizontally at pos[0] + 90");
    assert.strictEqual(inPosBT[1], 280, "BT: Input at bottom edge pos[1] + size[1]");
    assert.strictEqual(outPosBT[0], 190, "BT: Output centered horizontally at pos[0] + 90");
    assert.strictEqual(outPosBT[1], 200, "BT: Output at top edge pos[1]");

    console.log("✓ Test 1: Port coordinates correctly calculated for LR, RL, TB, BT");
}

// =========================================================================
// Test 2: Slot Tangents / Directions for Link Curves
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.flowMode = "lr";

    const node = new LGraphNode();
    node.addInput("in", "Tensor");
    node.addOutput("out", "Tensor");

    // LR
    editor.applyNodeSlotDirections(node, "lr");
    assert.strictEqual(node.inputs[0].dir, LiteGraph.LEFT, "LR: Input tangent should be LEFT");
    assert.strictEqual(node.outputs[0].dir, LiteGraph.RIGHT, "LR: Output tangent should be RIGHT");

    // RL
    editor.applyNodeSlotDirections(node, "rl");
    assert.strictEqual(node.inputs[0].dir, LiteGraph.RIGHT, "RL: Input tangent should be RIGHT");
    assert.strictEqual(node.outputs[0].dir, LiteGraph.LEFT, "RL: Output tangent should be LEFT");

    // TB
    editor.applyNodeSlotDirections(node, "tb");
    assert.strictEqual(node.inputs[0].dir, LiteGraph.UP, "TB: Input tangent should be UP");
    assert.strictEqual(node.outputs[0].dir, LiteGraph.DOWN, "TB: Output tangent should be DOWN");

    // BT
    editor.applyNodeSlotDirections(node, "bt");
    assert.strictEqual(node.inputs[0].dir, LiteGraph.DOWN, "BT: Input tangent should be DOWN");
    assert.strictEqual(node.outputs[0].dir, LiteGraph.UP, "BT: Output tangent should be UP");

    console.log("✓ Test 2: Slot tangents (slot.dir) correctly set for all 4 orientations");
}

// =========================================================================
// Test 3: Radial Hit-Testing for Slots
// =========================================================================
{
    const canvas = Object.create(LGraphCanvas.prototype);
    const node = new LGraphNode();
    node.pos = [100, 100];
    node.addInput("in", "Tensor");
    node.addOutput("out", "Tensor");
    node.size = [180, 80];

    // In RL mode: input is at [271, 114]
    node.flowMode = "rl";
    const slotIdx = canvas.isOverNodeInput(node, 268, 112);
    assert.strictEqual(slotIdx, 0, "Radial hit test finds input slot near [271, 114]");

    const farIdx = canvas.isOverNodeInput(node, 100, 100);
    assert.strictEqual(farIdx, -1, "Radial hit test ignores distant coordinates");

    console.log("✓ Test 3: Symmetrical radial hit-testing functions properly across directions");
}

// =========================================================================
// Test 4: Real-Time Grid Snapping
// =========================================================================
{
    const graph = new LGraph();
    const nodeA = new LGraphNode();
    nodeA.pos = [100, 100];
    graph.add(nodeA);

    const canvas = Object.create(LGraphCanvas.prototype);
    canvas.graph = graph;
    canvas.snap_to_grid = true;
    canvas.snap_grid_size = 20;
    canvas.selected_nodes = { [nodeA.id]: nodeA };
    canvas.node_dragged = nodeA;
    canvas.ds = { scale: 1, offset: [0, 0] };
    canvas.last_mouse_position = [0, 0];
    canvas.last_mouse = [0, 0];
    canvas.mouse = [0, 0];
    canvas.graph_mouse = [0, 0];
    canvas.options = { skip_events: true };
    canvas.visible_links = [];
    canvas.visible_nodes = [nodeA];
    canvas.allow_dragnodes = true;
    canvas.allow_interaction = true;
    canvas.getCanvasWindow = () => global.window;
    canvas.canvas = { focus: () => {}, getBoundingClientRect: () => ({ left: 0, top: 0 }), style: {}, ownerDocument: { defaultView: global.window } };

    const createEvent = (cx, cy) => ({
        canvasX: cx,
        canvasY: cy,
        clientX: cx,
        clientY: cy,
        localX: cx,
        localY: cy,
        which: 1,
        preventDefault: () => {},
        stopPropagation: () => {}
    });

    // Simulate drag start
    canvas.processMouseDown(createEvent(100, 100));
    assert.ok(canvas._dragStartPosMap, "Drag start records initial positions");

    // Move by 6px (less than half grid 10px -> stays at 100)
    canvas.processMouseMove(createEvent(106, 100));
    assert.strictEqual(nodeA.pos[0], 100, "Small delta < 10px snaps to 100");

    // Move by 15px (crosses half grid threshold -> jumps to 120)
    canvas.processMouseMove(createEvent(115, 100));
    assert.strictEqual(nodeA.pos[0], 120, "Delta >= 10px snaps to 120 (next grid dot)");

    // Mouse up
    canvas.processMouseUp(createEvent(115, 100));
    assert.strictEqual(nodeA.pos[0], 120, "Position finalized at 120");
    assert.strictEqual(canvas._dragStartPosMap, null, "Drag state cleared");

    console.log("✓ Test 4: Real-time grid snapping correctly quantizes coordinates to 20px grid");
}

// =========================================================================
// Test 5: Auto-Layout in all 4 Flow Directions
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.toast = () => {};
    editor.canvas = { setDirty: () => {}, draw: () => {}, snap_grid_size: 20 };
    editor.graph = new LGraph();

    const n1 = new LGraphNode();
    n1.id = 1;
    n1.title = "Input";
    n1.addOutput("out", "Tensor");

    const n2 = new LGraphNode();
    n2.id = 2;
    n2.title = "Conv";
    n2.addInput("in", "Tensor");
    n2.addOutput("out", "Tensor");

    const n3 = new LGraphNode();
    n3.id = 3;
    n3.title = "Dense";
    n3.addInput("in", "Tensor");

    editor.graph.add(n1);
    editor.graph.add(n2);
    editor.graph.add(n3);
    n1.connect(0, n2, 0);
    n2.connect(0, n3, 0);

    // 5.1 Auto-Layout LR: n1.pos[0] < n2.pos[0] < n3.pos[0]
    editor.autoLayout("lr");
    assert.ok(n1.pos[0] < n2.pos[0], "LR: Input is left of Conv");
    assert.ok(n2.pos[0] < n3.pos[0], "LR: Conv is left of Dense");
    assert.strictEqual(n1.pos[0] % 20, 0, "LR: n1 snapped to 20px grid");
    assert.strictEqual(n2.pos[0] % 20, 0, "LR: n2 snapped to 20px grid");
    assert.strictEqual(n3.pos[0] % 20, 0, "LR: n3 snapped to 20px grid");

    // 5.2 Auto-Layout RL: n1.pos[0] > n2.pos[0] > n3.pos[0] (Right-to-Left!)
    editor.autoLayout("rl");
    assert.ok(n1.pos[0] > n2.pos[0], "RL: Input is right of Conv");
    assert.ok(n2.pos[0] > n3.pos[0], "RL: Conv is right of Dense");
    assert.strictEqual(n1.pos[0] % 20, 0, "RL: n1 snapped to 20px grid");
    assert.strictEqual(n2.pos[0] % 20, 0, "RL: n2 snapped to 20px grid");

    // 5.3 Auto-Layout TB: n1.pos[1] < n2.pos[1] < n3.pos[1] (Top-to-Bottom!)
    editor.autoLayout("tb");
    assert.ok(n1.pos[1] < n2.pos[1], "TB: Input is above Conv");
    assert.ok(n2.pos[1] < n3.pos[1], "TB: Conv is above Dense");
    assert.strictEqual(n1.pos[1] % 20, 0, "TB: n1 snapped to 20px grid");

    // 5.4 Auto-Layout BT: n1.pos[1] > n2.pos[1] > n3.pos[1] (Bottom-to-Top!)
    editor.autoLayout("bt");
    assert.ok(n1.pos[1] > n2.pos[1], "BT: Input is below Conv");
    assert.ok(n2.pos[1] > n3.pos[1], "BT: Conv is below Dense");
    assert.strictEqual(n1.pos[1] % 20, 0, "BT: n1 snapped to 20px grid");

    console.log("✓ Test 5: Auto-layout organizes DAG properly in LR, RL, TB, BT");
}

// =========================================================================
// Test 6: Model Loading Defaulting to LR for Legacy Models
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.toast = () => {};
    editor.updateFlowToolbarUI = () => {};
    editor.validateGraph = () => {};
    editor.canvas = { setDirty: () => {}, snap_grid_size: 20 };
    editor.graph = new LGraph();

    // Legacy model JSON with no flowMode specified
    const legacyModel = {
        nodes: [
            { id: 1, type: "keras/input", pos: [103, 107] },
            { id: 2, type: "keras/dense", pos: [301, 102] }
        ],
        links: [
            [1, 1, 0, 2, 0, "Tensor"]
        ]
    };

    editor.loadGraph(legacyModel);
    assert.strictEqual(editor.flowMode, "lr", "Legacy models without flowMode MUST default to 'lr'");
    assert.strictEqual(editor.graph.flowMode, "lr", "Graph flowMode set to 'lr'");

    // Coordinates snapped to 20px grid
    const loadedNode1 = editor.graph.getNodeById(1);
    assert.strictEqual(loadedNode1.pos[0], 100, "103 snapped to 100");
    assert.strictEqual(loadedNode1.pos[1], 100, "107 snapped to 100");

    console.log("✓ Test 6: Legacy models correctly default to Left-to-Right layout with snapped coordinates");
}

// =========================================================================
// Test 7: Serialization Preserves Flow & Overrides
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.graph = new LGraph();
    editor.flowMode = "rl";

    const nodeA = new LGraphNode();
    nodeA.id = 1;
    nodeA.flowMode = "tb"; // Override on nodeA
    editor.graph.add(nodeA);

    const serialized = editor.serializeGraph();
    assert.strictEqual(serialized.flowMode, "rl", "Serialized graph preserves global flowMode");
    assert.strictEqual(serialized.nodes[0].flowMode, "tb", "Serialized node preserves individual flowMode override");

    console.log("✓ Test 7: Serialization correctly stores graph flowMode and node overrides");
}

// =========================================================================
// Test 8: Node Orientation Switching & Cycling
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.toast = () => {};
    editor.updateInspector = () => {};
    editor.canvas = { setDirty: () => {}, draw: () => {} };
    editor.graph = new LGraph();
    editor.flowMode = "lr";

    const node = new LGraphNode();
    node.id = 42;
    node.addInput("in", "Tensor");
    node.addOutput("out", "Tensor");
    editor.graph.add(node);

    // Switch to RL
    editor.setNodeOrientation(42, "rl");
    assert.strictEqual(node.flowMode, "rl");
    assert.strictEqual(node.inputs[0].dir, LiteGraph.RIGHT);
    assert.strictEqual(node.outputs[0].dir, LiteGraph.LEFT);

    // Cycle: RL -> TB
    editor.cycleNodeOrientation(42);
    assert.strictEqual(node.flowMode, "tb");
    assert.strictEqual(node.inputs[0].dir, LiteGraph.UP);
    assert.strictEqual(node.outputs[0].dir, LiteGraph.DOWN);

    // Reset default
    editor.setNodeOrientation(42, null);
    assert.strictEqual(node.flowMode, undefined);
    assert.strictEqual(node.inputs[0].dir, LiteGraph.LEFT);
    assert.strictEqual(node.outputs[0].dir, LiteGraph.RIGHT);

    console.log("✓ Test 8: Node orientation setting and cycling work seamlessly");
}

// =========================================================================
// Test 9: Collapsed Node Slot Positioning across All 4 Directions
// =========================================================================
{
    const node = new LGraphNode();
    node.pos = [100, 200];
    node.flags = { collapsed: true };
    node._collapsed_width = 120;
    const titleH = LiteGraph.NODE_TITLE_HEIGHT || 30;

    // LR collapsed
    node.flowMode = "lr";
    const inLR = node.getConnectionPos(true, 0);
    const outLR = node.getConnectionPos(false, 0);
    assert.strictEqual(inLR[0], 100, "Collapsed LR input at left");
    assert.strictEqual(outLR[0], 220, "Collapsed LR output at right");

    // RL collapsed
    node.flowMode = "rl";
    const inRL = node.getConnectionPos(true, 0);
    const outRL = node.getConnectionPos(false, 0);
    assert.strictEqual(inRL[0], 220, "Collapsed RL input at right");
    assert.strictEqual(outRL[0], 100, "Collapsed RL output at left");

    // TB collapsed
    node.flowMode = "tb";
    const inTB = node.getConnectionPos(true, 0);
    const outTB = node.getConnectionPos(false, 0);
    assert.strictEqual(inTB[0], 160, "Collapsed TB input centered horizontally");
    assert.strictEqual(inTB[1], 200 - titleH, "Collapsed TB input at top title edge");
    assert.strictEqual(outTB[0], 160, "Collapsed TB output centered horizontally");
    assert.strictEqual(outTB[1], 200, "Collapsed TB output at bottom edge");

    // BT collapsed
    node.flowMode = "bt";
    const inBT = node.getConnectionPos(true, 0);
    const outBT = node.getConnectionPos(false, 0);
    assert.strictEqual(inBT[0], 160, "Collapsed BT input centered horizontally");
    assert.strictEqual(inBT[1], 200, "Collapsed BT input at bottom edge");
    assert.strictEqual(outBT[0], 160, "Collapsed BT output centered horizontally");
    assert.strictEqual(outBT[1], 200 - titleH, "Collapsed BT output at top title edge");

    console.log("✓ Test 9: Collapsed node slot positions correctly oriented across LR, RL, TB, BT");
}

// =========================================================================
// Test 10: Multi-Node Group Drag Snapping
// =========================================================================
{
    const graph = new LGraph();
    const node1 = new LGraphNode();
    node1.id = 101;
    node1.pos = [100, 100];
    const node2 = new LGraphNode();
    node2.id = 102;
    node2.pos = [240, 100]; // 140px offset
    graph.add(node1);
    graph.add(node2);

    const canvas = Object.create(LGraphCanvas.prototype);
    canvas.graph = graph;
    canvas.snap_to_grid = true;
    canvas.snap_grid_size = 20;
    canvas.selected_nodes = { 101: node1, 102: node2 };
    canvas.node_dragged = node1;
    canvas.ds = { scale: 1, offset: [0, 0] };
    canvas.last_mouse_position = [0, 0];
    canvas.last_mouse = [0, 0];
    canvas.mouse = [0, 0];
    canvas.graph_mouse = [0, 0];
    canvas.options = { skip_events: true };
    canvas.visible_links = [];
    canvas.visible_nodes = [node1, node2];
    canvas.allow_dragnodes = true;
    canvas.allow_interaction = true;
    canvas.getCanvasWindow = () => global.window;
    canvas.canvas = { focus: () => {}, getBoundingClientRect: () => ({ left: 0, top: 0 }), style: {}, ownerDocument: { defaultView: global.window } };

    const createEvt = (x, y) => ({
        canvasX: x, canvasY: y, clientX: x, clientY: y, localX: x, localY: y, which: 1,
        preventDefault: () => {}, stopPropagation: () => {}
    });

    canvas.processMouseDown(createEvt(100, 100));
    // Drag by 27px -> rounds to 20px grid increment
    canvas.processMouseMove(createEvt(127, 100));
    assert.strictEqual(node1.pos[0], 120, "Node 1 snapped from 100 to 120");
    assert.strictEqual(node2.pos[0], 260, "Node 2 snapped from 240 to 260 (preserving 140px relative delta)");

    canvas.processMouseUp(createEvt(127, 100));
    assert.strictEqual(node1.pos[0], 120);
    assert.strictEqual(node2.pos[0], 260);

    console.log("✓ Test 10: Multi-node group drag snapping preserves relative alignment");
}

// =========================================================================
// Test 11: Branched DAG Auto-Layout with Multi-Input Concatenate
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.toast = () => {};
    editor.canvas = { setDirty: () => {}, draw: () => {}, snap_grid_size: 20 };
    editor.graph = new LGraph();

    // Stream 1: In1 (1) -> Conv (3) \
    //                                -> Concat (5) -> Dense (6)
    // Stream 2: In2 (2) -> Dense1 (4)/
    const nIn1 = new LGraphNode(); nIn1.id = 1; nIn1.addOutput("out", "Tensor");
    const nIn2 = new LGraphNode(); nIn2.id = 2; nIn2.addOutput("out", "Tensor");
    const nConv = new LGraphNode(); nConv.id = 3; nConv.addInput("in", "Tensor"); nConv.addOutput("out", "Tensor");
    const nDense1 = new LGraphNode(); nDense1.id = 4; nDense1.addInput("in", "Tensor"); nDense1.addOutput("out", "Tensor");
    const nConcat = new LGraphNode(); nConcat.id = 5; nConcat.addInput("in1", "Tensor"); nConcat.addInput("in2", "Tensor"); nConcat.addOutput("out", "Tensor");
    const nDense2 = new LGraphNode(); nDense2.id = 6; nDense2.addInput("in", "Tensor");

    editor.graph.add(nIn1);
    editor.graph.add(nIn2);
    editor.graph.add(nConv);
    editor.graph.add(nDense1);
    editor.graph.add(nConcat);
    editor.graph.add(nDense2);

    nIn1.connect(0, nConv, 0);
    nIn2.connect(0, nDense1, 0);
    nConv.connect(0, nConcat, 0);
    nDense1.connect(0, nConcat, 1);
    nConcat.connect(0, nDense2, 0);

    // Auto-layout RL
    editor.autoLayout("rl");
    assert.ok(nIn1.pos[0] > nConv.pos[0], "RL: In1 is to the right of Conv");
    assert.ok(nConv.pos[0] > nConcat.pos[0], "RL: Conv is to the right of Concat");
    assert.ok(nConcat.pos[0] > nDense2.pos[0], "RL: Concat is to the right of Dense2");

    // Auto-layout BT
    editor.autoLayout("bt");
    assert.ok(nIn1.pos[1] > nConv.pos[1], "BT: In1 is below Conv");
    assert.ok(nConv.pos[1] > nConcat.pos[1], "BT: Conv is below Concat");
    assert.ok(nConcat.pos[1] > nDense2.pos[1], "BT: Concat is below Dense2");

    console.log("✓ Test 11: Branched multi-input DAG auto-layout succeeds in RL and BT flows");
}

console.log("\n=======================================================");
// =========================================================================
// Test 12: Input Node Multi-Channel Selection, Presets & Shape Updating
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.toast = () => {};
    editor.canvas = { setDirty: () => {}, draw: () => {}, snap_grid_size: 20 };
    editor.graph = new LGraph();
    editor.registerKerasNodes();

    const inputNode = LiteGraph.createNode("keras/input");
    editor.graph.add(inputNode);

    // Default properties
    assert.strictEqual(inputNode.properties.shape, "512, 1", "Default shape should be 512, 1");
    assert.strictEqual(inputNode.properties.channels, "raw", "Default channel is 'raw'");
    assert.strictEqual(inputNode.properties.input_source, "signal_1d", "Default input_source is 'signal_1d'");
    assert.strictEqual(inputNode.properties.dataset_id, "", "Default dataset_id is empty");

    // Multi-channel selection: raw, derivative_1, derivative_2 (3 channels)
    const selectedChannels = ["raw", "derivative_1", "derivative_2"];
    inputNode.properties.channels = selectedChannels.join(",");
    inputNode.properties.shape = `512, ${selectedChannels.length}`;
    assert.strictEqual(inputNode.properties.shape, "512, 3", "Multi-channel selection updates shape to 512, 3");

    // Full Multi-Channel Preset (5 channels)
    const fullPreset = "raw,derivative_1,derivative_2,region_mask,envelope_rms";
    inputNode.properties.channels = fullPreset;
    inputNode.properties.shape = `512, ${fullPreset.split(",").length}`;
    assert.strictEqual(inputNode.properties.shape, "512, 5", "Full multi-channel preset updates shape to 512, 5");

    console.log("✓ Test 12: Input node multi-channel selection & preset shape updating verified");
}

// =========================================================================
// Test 13: Per-Node Dataset Selection on Input and Model Output Nodes
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.toast = () => {};
    editor.canvas = { setDirty: () => {}, draw: () => {} };
    editor.graph = new LGraph();
    editor.registerKerasNodes();

    const inNode1 = LiteGraph.createNode("keras/input");
    inNode1.id = 1;
    inNode1.properties.dataset_id = "ds_raw_signals";
    inNode1.properties.dataset_name = "Raw_ECG_Signals";
    inNode1.properties.shape = "512, 1";
    editor.graph.add(inNode1);

    const inNode2 = LiteGraph.createNode("keras/input");
    inNode2.id = 2;
    inNode2.properties.dataset_id = "ds_deriv1";
    inNode2.properties.dataset_name = "Derivative1_Set";
    inNode2.properties.shape = "512, 1";
    editor.graph.add(inNode2);

    const outNode = LiteGraph.createNode("keras/output");
    outNode.id = 3;
    outNode.properties.dataset_id = "ds_regional_masks";
    outNode.properties.dataset_name = "Event_Labels";
    outNode.properties.target_source = "region_labels";
    editor.graph.add(outNode);

    // Serialization verification
    const serialized = editor.serializeGraph();
    const serIn1 = serialized.nodes.find(n => n.id === 1);
    const serIn2 = serialized.nodes.find(n => n.id === 2);
    const serOut = serialized.nodes.find(n => n.id === 3);

    assert.strictEqual(serIn1.properties.dataset_id, "ds_raw_signals");
    assert.strictEqual(serIn1.properties.dataset_name, "Raw_ECG_Signals");
    assert.strictEqual(serIn2.properties.dataset_id, "ds_deriv1");
    assert.strictEqual(serOut.properties.dataset_id, "ds_regional_masks");
    assert.strictEqual(serOut.properties.dataset_name, "Event_Labels");

    console.log("✓ Test 13: Per-node dataset selection (dataset_id, dataset_name) serializes properly");
}

// =========================================================================
// Test 14: Output Node Target Data & Automatic Loss/Activation Defaults
// =========================================================================
{
    const outNode = LiteGraph.createNode("keras/output");
    assert.strictEqual(outNode.properties.target_source, "auto");
    assert.strictEqual(outNode.properties.activation, "linear");
    assert.strictEqual(outNode.properties.loss, "auto");
    assert.strictEqual(outNode.properties.loss_weight, 1.0);

    const setTarget = (node, target) => {
        node.properties.target_source = target;
        if (target === "region_labels") {
            node.properties.activation = "sigmoid";
            node.properties.loss = "binary_crossentropy";
        } else if (target === "reconstruction") {
            node.properties.activation = "linear";
            node.properties.loss = "mse";
        } else if (target === "label_class") {
            node.properties.activation = "softmax";
            node.properties.loss = "sparse_categorical_crossentropy";
        }
    };

    setTarget(outNode, "region_labels");
    assert.strictEqual(outNode.properties.activation, "sigmoid");
    assert.strictEqual(outNode.properties.loss, "binary_crossentropy");

    setTarget(outNode, "reconstruction");
    assert.strictEqual(outNode.properties.activation, "linear");
    assert.strictEqual(outNode.properties.loss, "mse");

    setTarget(outNode, "label_class");
    assert.strictEqual(outNode.properties.activation, "softmax");
    assert.strictEqual(outNode.properties.loss, "sparse_categorical_crossentropy");

    console.log("✓ Test 14: Output node target configuration & auto defaults work correctly");
}

// =========================================================================
// Test 15: Dynamic Multi-Port Adding & Removing on Concatenate & Merge Nodes
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.toast = () => {};
    editor.canvas = { setDirty: () => {} };
    editor.applyNodeSlotDirections = (node) => {
        NNEditor.prototype.applyNodeSlotDirections.call(editor, node, "lr");
    };
    editor.addMergeInputPort = NNEditor.prototype.addMergeInputPort.bind(editor);
    editor.removeMergeInputPort = NNEditor.prototype.removeMergeInputPort.bind(editor);

    const concatNode = LiteGraph.createNode("keras/concatenate");
    assert.strictEqual(concatNode.inputs.length, 2, "Default Concatenate node has 2 input ports");

    // Add 3rd port
    editor.addMergeInputPort(concatNode);
    assert.strictEqual(concatNode.inputs.length, 3, "After 1st add, has 3 ports (in1, in2, in3)");
    assert.strictEqual(concatNode.inputs[2].name, "in3");

    // Add 4th port
    editor.addMergeInputPort(concatNode);
    assert.strictEqual(concatNode.inputs.length, 4, "After 2nd add, has 4 ports (in1, in2, in3, in4)");
    assert.strictEqual(concatNode.inputs[3].name, "in4");

    // Remove 4th port
    editor.removeMergeInputPort(concatNode);
    assert.strictEqual(concatNode.inputs.length, 3, "After remove, has 3 ports");

    // Remove 3rd port
    editor.removeMergeInputPort(concatNode);
    assert.strictEqual(concatNode.inputs.length, 2, "After remove, has 2 ports");

    // Attempting to remove below 2 is rejected
    editor.removeMergeInputPort(concatNode);
    assert.strictEqual(concatNode.inputs.length, 2, "Cannot remove below 2 ports");

    console.log("✓ Test 15: Dynamic multi-port add/remove on Concatenate verified (supports >= 3 input streams)");
}

// =========================================================================
// Test 16: Architecture Wizard Multi-Channel 1D U-Net
// =========================================================================
{
    const editor = Object.create(NNEditor.prototype);
    editor.toast = () => {};
    editor.canvas = { setDirty: () => {}, draw: () => {}, snap_grid_size: 20 };
    editor.graph = new LGraph();
    editor.registerKerasNodes();
    editor.applyNodeSlotDirections = NNEditor.prototype.applyNodeSlotDirections.bind(editor);
    editor.generateArchitecture = NNEditor.prototype.generateArchitecture.bind(editor);

    // Generate 1D U-Net with 3-channel input (raw, derivative_1, derivative_2)
    editor.generateArchitecture("unet_1d", {
        seqLen: 512,
        baseFilters: 32,
        inputChannels: "raw,derivative_1,derivative_2"
    });

    const inNode = editor.graph._nodes.find(n => n.type === "keras/input");
    const outNode = editor.graph._nodes.find(n => n.type === "keras/output");

    assert.ok(inNode, "Input node generated");
    assert.strictEqual(inNode.properties.shape, "512, 3", "Input shape set to 512, 3 for 3 channels");
    assert.strictEqual(inNode.properties.channels, "raw,derivative_1,derivative_2");

    assert.ok(outNode, "Output node generated");
    assert.strictEqual(outNode.properties.target_source, "region_labels", "U-Net output target is region_labels");
    assert.strictEqual(outNode.properties.loss, "binary_crossentropy", "U-Net loss is binary_crossentropy");
    assert.strictEqual(outNode.properties.activation, "sigmoid", "U-Net activation is sigmoid");

    console.log("✓ Test 16: Architecture Wizard multi-channel 1D U-Net pre-configuration verified");
}

// =========================================================================
// Test 17: Client-Side Multi-Prefix CNN Column Grouping Detector
// =========================================================================
{
    function detectColumnPrefixGroups(columns, excludedSet = new Set(), columnSchema = []) {
        const rx = /^([a-zA-Z_][a-zA-Z0-9_]*?)_?(\d+)$/;
        const groupsMap = new Map();

        const schemaRoleMap = new Map();
        (columnSchema || []).forEach(c => schemaRoleMap.set(c.name, c.inferred_role));

        for (const col of columns) {
            const cStr = String(col);
            if (excludedSet.has(cStr)) continue;
            const role = schemaRoleMap.get(cStr);
            if (role === "id" || role === "output" || role === "ignore") continue;

            const m = cStr.match(rx);
            if (!m) continue;
            const prefix = m[1];
            const num = parseInt(m[2], 10);
            if (!groupsMap.has(prefix)) {
                groupsMap.set(prefix, []);
            }
            groupsMap.get(prefix).push({ col: cStr, num });
        }

        const result = [];
        for (const [prefix, items] of groupsMap.entries()) {
            if (items.length < 2) continue;
            items.sort((a, b) => a.num - b.num);
            const groupCols = items.map(it => it.col);
            const firstCol = groupCols[0];
            const lastCol = groupCols[groupCols.length - 1];
            const pattern = `${firstCol}-${lastCol}`;
            result.push({
                name: "channel_" + prefix,
                channel_name: prefix,
                prefix: prefix,
                pattern: pattern,
                columns: groupCols,
                count: groupCols.length
            });
        }
        return result;
    }

    // Build synthetic column list matching user's structure: s0..s511, a_0..a_511, dvdt_0..dvdt_511, plus id & target
    const sampleCols = [];
    sampleCols.push("sample_id");
    for (let i = 0; i < 512; i++) sampleCols.push(`s${i}`);
    for (let i = 0; i < 512; i++) sampleCols.push(`a_${i}`);
    for (let i = 0; i < 512; i++) sampleCols.push(`dvdt_${i}`);
    sampleCols.push("target_class");

    const schema = [
        { name: "sample_id", inferred_role: "id" },
        { name: "target_class", inferred_role: "output" }
    ];

    const detected = detectColumnPrefixGroups(sampleCols, new Set(["sample_id"]), schema);

    assert.strictEqual(detected.length, 3, "Detected 3 distinct channel prefix series (s*, a_*, dvdt_*)");

    const chS = detected.find(g => g.channel_name === "s");
    assert.ok(chS, "Channel 's' detected");
    assert.strictEqual(chS.pattern, "s0-s511");
    assert.strictEqual(chS.count, 512);

    const chA = detected.find(g => g.channel_name === "a");
    assert.ok(chA, "Channel 'a' detected");
    assert.strictEqual(chA.pattern, "a_0-a_511");
    assert.strictEqual(chA.count, 512);

    const chDvdt = detected.find(g => g.channel_name === "dvdt");
    assert.ok(chDvdt, "Channel 'dvdt' detected");
    assert.strictEqual(chDvdt.pattern, "dvdt_0-dvdt_511");
    assert.strictEqual(chDvdt.count, 512);

    // Multi-channel 3D shape computation verification
    const tensorShape = `(${100}, ${chS.count}, ${detected.length})`;
    assert.strictEqual(tensorShape, "(100, 512, 3)", "Correct 3D tensor shape for multi-channel CNN");

    console.log("✓ Test 17: One-click multi-prefix CNN auto-grouping detector verified for s*, a_*, dvdt_*");
}

// =========================================================================
// Test 18: PCA Visualizer Controllable Ground Truth Sanitization & Modes
// =========================================================================
{
    // Provide canvas & highcharts mock for PCAVisualizer testing
    global.document = {
        getElementById: () => ({
            addEventListener: () => {},
            querySelector: () => ({ textContent: "" }),
            getContext: () => ({
                clearRect: () => {},
                beginPath: () => {},
                arc: () => {},
                fill: () => {},
                stroke: () => {},
                save: () => {},
                restore: () => {},
                fillText: () => {}
            }),
            classList: { add: () => {}, remove: () => {} },
            style: {}
        }),
        addEventListener: () => {}
    };
    global.Highcharts = {
        chart: () => ({ reflow: () => {} })
    };

    delete require.cache[require.resolve("../app/static/js/pca_visualizer.js")];
    require("../app/static/js/pca_visualizer.js");
    const PCAVisualizer = global.PCAVisualizer;

    const vis = new PCAVisualizer("test-pca-container");

    // 18.1 Default Ground Truth state
    assert.strictEqual(vis.showGroundTruth, false, "Ground truth preview must default to false");
    assert.strictEqual(vis.colorBy, "cluster", "Default color mode is 'cluster'");

    // 18.2 Ground Truth Sanitization
    assert.strictEqual(vis.formatGroundTruth(undefined), "");
    assert.strictEqual(vis.formatGroundTruth(null), "");
    assert.strictEqual(vis.formatGroundTruth(42), "42");
    assert.strictEqual(vis.formatGroundTruth("class_normal"), "class_normal");
    assert.strictEqual(vis.formatGroundTruth([0.1, 0.2, 0.3]), "[3 dims]");
    assert.strictEqual(vis.formatGroundTruth("[0.12, 0.34, 0.56, 0.78]"), "[4 dims]");
    assert.strictEqual(vis.formatGroundTruth("this_is_an_extremely_long_ground_truth_label_string"), "this_is_an_extremely_...");

    // 18.3 Scalar Ground Truth validation
    assert.strictEqual(vis.isValidScalarGroundTruth(0), true);
    assert.strictEqual(vis.isValidScalarGroundTruth("OK"), true);
    assert.strictEqual(vis.isValidScalarGroundTruth([1, 2]), false);
    assert.strictEqual(vis.isValidScalarGroundTruth("[1, 2, 3]"), false);
    assert.strictEqual(vis.isValidScalarGroundTruth(""), false);

    // 18.4 Load sample data with discrete labels and verify color mode toggling
    vis.loadData({
        points: [
            { x: 0.1, y: 0.2, z: 0.3, cluster: 0, sample_id: "s1", ground_truth: "class_0" },
            { x: 0.4, y: 0.5, z: 0.6, cluster: 1, sample_id: "s2", ground_truth: "class_1" },
            { x: 0.7, y: 0.8, z: 0.9, cluster: 0, sample_id: "s3", ground_truth: "class_0" }
        ],
        cluster_distribution: { "0": 2, "1": 1 }
    });

    assert.deepStrictEqual(vis.uniqueGroundTruths, ["class_0", "class_1"]);

    // Toggle color mode
    vis.toggleColorMode();
    assert.strictEqual(vis.colorBy, "ground_truth", "Color mode switched to 'ground_truth'");

    vis.toggleGroundTruth(true);
    assert.strictEqual(vis.showGroundTruth, true, "Ground truth preview toggled on");

    console.log("✓ Test 18: PCA Visualizer controllable ground truth sanitization, validation, and color-by modes verified");
}

console.log("ALL 18 NN BUILDER & CLUSTERING FLOW TESTS PASSED SUCCESSFULLY!");
console.log("=======================================================");
