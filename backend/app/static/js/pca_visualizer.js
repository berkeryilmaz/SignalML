// Interactive 2D & 3D PCA Visualizer with Cluster Color-Coding & Tooltips

class PCAVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.points = [];
        this.mode = "2d"; // "2d" or "3d"
        this.chart2d = null;

        // 3D Canvas properties
        this.canvas3d = null;
        this.ctx3d = null;
        this.rotX = 0.4;
        this.rotY = 0.6;
        this.zoom = 1.0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.hoveredPoint = null;
        this.projectedPoints = [];

        this.init();

        window.addEventListener("signalml:themechange", () => {
            if (this.mode === "3d") {
                this.render3D();
            } else if (this.chart2d) {
                this.render2D();
            }
        });

        window.addEventListener("resize", () => {
            if (this.mode === "3d") {
                this.resize3DCanvas();
                this.render3D();
            }
        });
    }

    init() {
        if (!this.container) return;
        const icon2d = typeof SignalML !== "undefined" ? SignalML.icon("circle-dot", "icon-xs") : "";
        const icon3d = typeof SignalML !== "undefined" ? SignalML.icon("box", "icon-xs") : "";

        this.container.innerHTML = `
            <div class="cluster" style="position: absolute; top: 12px; right: 12px; z-index: 10;">
                <button type="button" class="btn btn-primary btn-sm" id="pca-toggle-2d">
                    ${icon2d} <span>2D Projection</span>
                </button>
                <button type="button" class="btn btn-secondary btn-sm" id="pca-toggle-3d">
                    ${icon3d} <span>3D Interactive</span>
                </button>
            </div>
            <div id="pca-2d-canvas" class="w-full h-full"></div>
            <canvas id="pca-3d-canvas" class="w-full h-full is-hidden" style="cursor: grab;"></canvas>
            <div id="pca-tooltip" class="tile is-hidden text-micro" style="position: absolute; pointer-events: none; z-index: 20; min-width: 140px;"></div>
        `;

        document.getElementById("pca-toggle-2d").addEventListener("click", () => this.setMode("2d"));
        document.getElementById("pca-toggle-3d").addEventListener("click", () => this.setMode("3d"));

        this.setup3DCanvas();
    }

    setMode(mode) {
        this.mode = mode;
        const btn2d = document.getElementById("pca-toggle-2d");
        const btn3d = document.getElementById("pca-toggle-3d");
        const div2d = document.getElementById("pca-2d-canvas");
        const cvs3d = document.getElementById("pca-3d-canvas");

        if (mode === "2d") {
            if (btn2d) btn2d.className = "btn btn-primary btn-sm";
            if (btn3d) btn3d.className = "btn btn-secondary btn-sm";
            if (div2d) {
                div2d.classList.remove("is-hidden");
                div2d.style.display = "block";
            }
            if (cvs3d) {
                cvs3d.classList.add("is-hidden");
                cvs3d.style.display = "none";
            }
            this.render2D();
        } else {
            if (btn2d) btn2d.className = "btn btn-secondary btn-sm";
            if (btn3d) btn3d.className = "btn btn-primary btn-sm";
            if (div2d) {
                div2d.classList.add("is-hidden");
                div2d.style.display = "none";
            }
            if (cvs3d) {
                cvs3d.classList.remove("is-hidden");
                cvs3d.style.display = "block";
            }
            this.resize3DCanvas();
            this.render3D();
        }
    }

    loadData(data) {
        this.points = data.points || [];
        this.variance2d = data.explained_variance_2d || [];
        this.variance3d = data.explained_variance_3d || [];
        this.distribution = data.cluster_distribution || {};

        if (this.mode === "2d") {
            this.render2D();
        } else {
            this.render3D();
        }
    }

    getClusterColor(clusterId) {
        const p = typeof SignalML !== "undefined" ? SignalML.getThemePalette() : null;
        if (clusterId === -1) {
            return p ? p.textMuted : "#777b84"; // Outlier / Noise
        }
        const seriesColors = [
            "#0090ff", "#30a46c", "#ffc53d", "#e5484d",
            "#8e4ec6", "#00b8d9", "#f76b15", "#a0a7b4"
        ];
        return seriesColors[Math.abs(clusterId) % seriesColors.length];
    }

    render2D() {
        if (!this.points || this.points.length === 0) return;

        // Group points by cluster
        const clustersMap = {};
        for (let pt of this.points) {
            const cid = pt.cluster;
            if (!clustersMap[cid]) clustersMap[cid] = [];
            clustersMap[cid].push({
                x: pt.x,
                y: pt.y,
                sample_id: pt.sample_id,
                input_id: pt.input_id,
                ground_truth: pt.ground_truth
            });
        }

        const seriesList = Object.keys(clustersMap).map(cid => {
            const clusterNum = Number(cid);
            const name = clusterNum === -1 ? "Noise (Outliers)" : `Cluster ${clusterNum}`;
            return {
                name: `${name} (${clustersMap[cid].length})`,
                color: this.getClusterColor(clusterNum),
                clusterId: clusterNum,
                data: clustersMap[cid].map(p => ({
                    x: p.x,
                    y: p.y,
                    sample_id: p.sample_id,
                    input_id: p.input_id,
                    ground_truth: p.ground_truth
                }))
            };
        });

        const self = this;
        this.chart2d = Highcharts.chart("pca-2d-canvas", {
            chart: {
                type: "scatter",
                zoomType: "xy"
            },
            title: { text: null },
            xAxis: {
                title: {
                    text: `PC 1 (${((this.variance2d[0] || 0) * 100).toFixed(1)}% var)`
                }
            },
            yAxis: {
                title: {
                    text: `PC 2 (${((this.variance2d[1] || 0) * 100).toFixed(1)}% var)`
                }
            },
            legend: {
                itemStyle: { cursor: "pointer" }
            },
            tooltip: {
                formatter: function() {
                    const idLabel = this.point.input_id ? `ID: ${this.point.input_id} (#${this.point.sample_id})` : `Sample #${this.point.sample_id}`;
                    let s = `<b>${idLabel}</b><br/>`;
                    s += `Cluster: ${this.series.userOptions.clusterId}<br/>`;
                    s += `PC1: ${this.x.toFixed(3)}, PC2: ${this.y.toFixed(3)}<br/>`;
                    if (this.point.ground_truth !== undefined) {
                        s += `Ground Truth: ${this.point.ground_truth}<br/>`;
                    }
                    s += `<span style="color: var(--accent); font-size: 10px;">Click to view group waveforms</span>`;
                    return s;
                }
            },
            plotOptions: {
                scatter: {
                    marker: {
                        radius: 4,
                        states: { hover: { enabled: true } }
                    },
                    point: {
                        events: {
                            click: function() {
                                if (self.onClusterClick) {
                                    self.onClusterClick(this.series.userOptions.clusterId);
                                }
                            }
                        }
                    },
                    events: {
                        legendItemClick: function() {
                            if (self.onClusterClick) {
                                self.onClusterClick(this.userOptions.clusterId);
                            }
                            return true;
                        }
                    }
                }
            },
            series: seriesList
        });
    }

    setup3DCanvas() {
        this.canvas3d = document.getElementById("pca-3d-canvas");
        if (!this.canvas3d) return;
        this.ctx3d = this.canvas3d.getContext("2d");

        this.canvas3d.addEventListener("mousedown", (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas3d.style.cursor = "grabbing";
        });

        window.addEventListener("mouseup", () => {
            this.isDragging = false;
            if (this.canvas3d) this.canvas3d.style.cursor = "grab";
        });

        this.canvas3d.addEventListener("mousemove", (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.lastMouseX;
                const dy = e.clientY - this.lastMouseY;
                this.rotY += dx * 0.01;
                this.rotX += dy * 0.01;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.render3D();
            } else {
                this.check3DHover(e);
            }
        });

        this.canvas3d.addEventListener("wheel", (e) => {
            e.preventDefault();
            this.zoom *= e.deltaY < 0 ? 1.1 : 0.9;
            this.zoom = Math.max(0.2, Math.min(this.zoom, 5.0));
            this.render3D();
        });
    }

    resize3DCanvas() {
        if (!this.canvas3d) return;
        this.canvas3d.width = this.canvas3d.clientWidth;
        this.canvas3d.height = this.canvas3d.clientHeight;
    }

    render3D() {
        if (!this.ctx3d || !this.points || this.points.length === 0) return;
        const ctx = this.ctx3d;
        const w = this.canvas3d.width;
        const h = this.canvas3d.height;
        ctx.clearRect(0, 0, w, h);

        const palette = typeof SignalML !== "undefined" ? SignalML.getThemePalette() : null;
        const theme = typeof SignalML !== "undefined" ? SignalML.getTheme() : "dark";

        const cx = w / 2;
        const cy = h / 2;
        const scale = Math.min(w, h) * 0.25 * this.zoom;

        const cosX = Math.cos(this.rotX);
        const sinX = Math.sin(this.rotX);
        const cosY = Math.cos(this.rotY);
        const sinY = Math.sin(this.rotY);

        // Project 3D coordinate axes
        const axisLen = 1.2;
        const axes = [
            { label: "PC1", x: axisLen, y: 0, z: 0 },
            { label: "PC2", x: 0, y: axisLen, z: 0 },
            { label: "PC3", x: 0, y: 0, z: axisLen }
        ];

        const projectCoord = (x, y, z) => {
            const x1 = x * cosY + z * sinY;
            const y1 = y;
            const z1 = -x * sinY + z * cosY;
            const x2 = x1;
            const y2 = y1 * cosX - z1 * sinX;
            return {
                sx: cx + x2 * scale,
                sy: cy - y2 * scale
            };
        };

        // Draw axes lines
        const origin = projectCoord(0, 0, 0);
        ctx.lineWidth = 1;
        ctx.font = '10px "JetBrains Mono", monospace';

        axes.forEach(a => {
            const end = projectCoord(a.x, a.y, a.z);
            ctx.beginPath();
            ctx.moveTo(origin.sx, origin.sy);
            ctx.lineTo(end.sx, end.sy);
            ctx.strokeStyle = palette ? palette.borderDefault : "rgba(128,128,128,0.3)";
            ctx.stroke();

            ctx.fillStyle = palette ? palette.textMuted : "#777b84";
            ctx.fillText(a.label, end.sx + 4, end.sy + 4);
        });

        // Transform 3D coordinates to 2D screen
        const projected = [];
        for (let pt of this.points) {
            // Rotate around Y
            let x1 = pt.x * cosY + pt.z * sinY;
            let y1 = pt.y;
            let z1 = -pt.x * sinY + pt.z * cosY;

            // Rotate around X
            let x2 = x1;
            let y2 = y1 * cosX - z1 * sinX;
            let z2 = y1 * sinX + z1 * cosX;

            const screenX = cx + x2 * scale;
            const screenY = cy - y2 * scale;

            projected.push({
                pt: pt,
                sx: screenX,
                sy: screenY,
                depth: z2
            });
        }

        // Sort by depth for correct 3D occlusion
        projected.sort((a, b) => b.depth - a.depth);

        // Draw points with theme-aware border
        const strokeColor = theme === "light" ? "rgba(0,0,0,0.2)" : "rgba(0,0,0,0.5)";
        for (let p of projected) {
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, 4, 0, Math.PI * 2);
            ctx.fillStyle = this.getClusterColor(p.pt.cluster);
            ctx.fill();
            ctx.strokeStyle = strokeColor;
            ctx.stroke();
        }

        this.projectedPoints = projected;
    }

    check3DHover(e) {
        if (!this.projectedPoints) return;
        const rect = this.canvas3d.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const hit = this.projectedPoints.find(p => {
            const dist = Math.hypot(p.sx - mx, p.sy - my);
            return dist < 6;
        });

        const tooltip = document.getElementById("pca-tooltip");
        if (hit && tooltip) {
            tooltip.classList.remove("is-hidden");
            tooltip.style.display = "block";
            tooltip.style.left = `${mx + 15}px`;
            tooltip.style.top = `${my + 10}px`;
            const idLabel = hit.pt.input_id ? `ID: ${hit.pt.input_id} (#${hit.pt.sample_id})` : `Sample #${hit.pt.sample_id}`;
            tooltip.innerHTML = `
                <div class="fw-600">${idLabel}</div>
                <div class="text-secondary">Cluster: ${hit.pt.cluster === -1 ? 'Noise' : hit.pt.cluster}</div>
                <div class="mono text-micro text-muted">PC1: ${hit.pt.x.toFixed(2)}, PC2: ${hit.pt.y.toFixed(2)}, PC3: ${hit.pt.z.toFixed(2)}</div>
                ${hit.pt.ground_truth !== undefined ? `<div class="text-secondary text-micro">Ground Truth: ${hit.pt.ground_truth}</div>` : ''}
            `;
        } else if (tooltip) {
            tooltip.classList.add("is-hidden");
            tooltip.style.display = "none";
        }
    }
}

window.PCAVisualizer = PCAVisualizer;
