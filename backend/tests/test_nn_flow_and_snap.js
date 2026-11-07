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
console.log("ALL 11 NN BUILDER FLOW & SNAP TESTS PASSED SUCCESSFULLY!");
console.log("=======================================================");
