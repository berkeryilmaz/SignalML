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

        this.init();
    }

    init() {
        if (!this.container) return;
        this.container.innerHTML = `
            <div style="position: absolute; top: 12px; right: 12px; z-index: 10; display: flex; gap: 6px;">
                <button class="btn btn-primary btn-sm" id="pca-toggle-2d">2D Projection</button>
                <button class="btn btn-secondary btn-sm" id="pca-toggle-3d">3D Interactive</button>
            </div>
            <div id="pca-2d-canvas" style="width: 100%; height: 100%;"></div>
            <canvas id="pca-3d-canvas" style="width: 100%; height: 100%; display: none; cursor: grab;"></canvas>
            <div id="pca-tooltip" style="position: absolute; display: none; background: #191f2e; border: 1px solid #334155; padding: 6px 10px; border-radius: 6px; font-size: 11px; color: #fff; pointer-events: none; z-index: 20;"></div>
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
            btn2d.className = "btn btn-primary btn-sm";
            btn3d.className = "btn btn-secondary btn-sm";
            div2d.style.display = "block";
            cvs3d.style.display = "none";
            this.render2D();
        } else {
            btn2d.className = "btn btn-secondary btn-sm";
            btn3d.className = "btn btn-primary btn-sm";
            div2d.style.display = "none";
            cvs3d.style.display = "block";
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
        if (clusterId === -1) return "#94a3b8"; // DBSCAN Outlier / Noise
        const palette = ["#38bdf8", "#34d399", "#f43f5e", "#fbbf24", "#a855f7", "#ec4899", "#f97316", "#2dd4bf"];
        return palette[Math.abs(clusterId) % palette.length];
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
                backgroundColor: "transparent",
                zoomType: "xy"
            },
            title: { text: null },
            credits: { enabled: false },
            xAxis: {
                title: {
                    text: `PC 1 (${((this.variance2d[0] || 0) * 100).toFixed(1)}% var)`,
                    style: { color: "#94a3b8" }
                },
                gridLineColor: "#1e293b",
                labels: { style: { color: "#94a3b8" } }
            },
            yAxis: {
                title: {
                    text: `PC 2 (${((this.variance2d[1] || 0) * 100).toFixed(1)}% var)`,
                    style: { color: "#94a3b8" }
                },
                gridLineColor: "#1e293b",
                labels: { style: { color: "#94a3b8" } }
            },
            legend: {
                itemStyle: { color: "#f1f5f9", cursor: "pointer" }
            },
            tooltip: {
                backgroundColor: "#191f2e",
                borderColor: "#334155",
                style: { color: "#fff" },
                formatter: function() {
                    const idLabel = this.point.input_id ? `ID: ${this.point.input_id} (#${this.point.sample_id})` : `Sample #${this.point.sample_id}`;
                    let s = `<b>${idLabel}</b><br/>`;
                    s += `Cluster: ${this.series.userOptions.clusterId}<br/>`;
                    s += `PC1: ${this.x.toFixed(3)}, PC2: ${this.y.toFixed(3)}<br/>`;
                    if (this.point.ground_truth !== undefined) {
                        s += `Ground Truth: ${this.point.ground_truth}<br/>`;
                    }
                    s += `<span style="color: #38bdf8; font-size: 10px;">Click to view group waveforms</span>`;
                    return s;
                }
            },
            plotOptions: {
                scatter: {
                    marker: {
                        radius: 4,
                        states: { hover: { enabled: true, lineColor: "#fff" } }
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

        const cx = w / 2;
        const cy = h / 2;
        const scale = Math.min(w, h) * 0.25 * this.zoom;

        const cosX = Math.cos(this.rotX);
        const sinX = Math.sin(this.rotX);
        const cosY = Math.cos(this.rotY);
        const sinY = Math.sin(this.rotY);

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

        // Draw points
        for (let p of projected) {
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, 4, 0, Math.PI * 2);
            ctx.fillStyle = this.getClusterColor(p.pt.cluster);
            ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,0.4)";
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
            tooltip.style.display = "block";
            tooltip.style.left = `${mx + 15}px`;
            tooltip.style.top = `${my + 10}px`;
            const idLabel = hit.pt.input_id ? `ID: ${hit.pt.input_id} (#${hit.pt.sample_id})` : `Sample #${hit.pt.sample_id}`;
            tooltip.innerHTML = `
                <strong>${idLabel}</strong><br/>
                Cluster: ${hit.pt.cluster === -1 ? 'Noise' : hit.pt.cluster}<br/>
                PC1: ${hit.pt.x.toFixed(2)}, PC2: ${hit.pt.y.toFixed(2)}, PC3: ${hit.pt.z.toFixed(2)}
                ${hit.pt.ground_truth !== undefined ? `<br/>Truth: ${hit.pt.ground_truth}` : ''}
            `;
        } else if (tooltip) {
            tooltip.style.display = "none";
        }
    }
}

window.PCAVisualizer = PCAVisualizer;
