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
        try {
            if (new URLSearchParams(window.location.search).get("mode") === "3d") {
                this.setMode("3d");
            }
        } catch (err) {}
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

        if (typeof ResizeObserver !== "undefined" && this.container) {
            this._ro = new ResizeObserver(() => {
                if (this.mode === "3d") {
                    this.resize3DCanvas();
                    this.render3D();
                } else if (this.chart2d && typeof this.chart2d.reflow === "function") {
                    this.chart2d.reflow();
                }
            });
            this._ro.observe(this.container);
        }

        let dragMoved = false;
        this.canvas3d.addEventListener("mousedown", (e) => {
            this.isDragging = true;
            dragMoved = false;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas3d.style.cursor = "grabbing";
        });

        window.addEventListener("mouseup", (e) => {
            if (this.isDragging && !dragMoved && e.target === this.canvas3d) {
                this.handle3DClick(e);
            }
            this.isDragging = false;
            if (this.canvas3d) this.canvas3d.style.cursor = "grab";
        });

        this.canvas3d.addEventListener("mousemove", (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.lastMouseX;
                const dy = e.clientY - this.lastMouseY;
                if (Math.abs(dx) > 2 || Math.abs(dy) > 2) dragMoved = true;
                this.rotY += dx * 0.01;
                this.rotX = Math.max(-1.45, Math.min(1.45, this.rotX + dy * 0.01));
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
            this.zoom = Math.max(0.3, Math.min(this.zoom, 4.5));
            this.render3D();
        }, { passive: false });
    }

    resize3DCanvas() {
        if (!this.canvas3d) return;
        const parent = this.container || this.canvas3d.parentElement;
        const rect = parent ? parent.getBoundingClientRect() : this.canvas3d.getBoundingClientRect();
        const w = Math.max(320, Math.round(rect.width || this.canvas3d.clientWidth || 800));
        const h = Math.max(360, Math.round(rect.height || this.canvas3d.clientHeight || 500));
        const dpr = window.devicePixelRatio || 1;
        this._cssWidth = w;
        this._cssHeight = h;
        this.canvas3d.style.width = w + "px";
        this.canvas3d.style.height = h + "px";
        this.canvas3d.width = Math.round(w * dpr);
        this.canvas3d.height = Math.round(h * dpr);
        if (this.ctx3d) {
            this.ctx3d.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
    }

    render3D() {
        if (!this.ctx3d || !this.canvas3d) return;
        if (!this._cssWidth || !this._cssHeight) {
            this.resize3DCanvas();
        }
        const ctx = this.ctx3d;
        const w = this._cssWidth || this.canvas3d.width;
        const h = this._cssHeight || this.canvas3d.height;
        ctx.clearRect(0, 0, w, h);

        const palette = typeof SignalML !== "undefined" ? SignalML.getThemePalette() : null;
        const theme = typeof SignalML !== "undefined" ? SignalML.getTheme() : "dark";

        const cx = w * 0.5;
        const cy = h * 0.53;
        const scale = Math.min(w, h) * 0.30 * this.zoom;

        const cosX = Math.cos(this.rotX);
        const sinX = Math.sin(this.rotX);
        const cosY = Math.cos(this.rotY);
        const sinY = Math.sin(this.rotY);

        const projectCoord = (x, y, z) => {
            const x1 = x * cosY + z * sinY;
            const y1 = y;
            const z1 = -x * sinY + z * cosY;
            const x2 = x1;
            const y2 = y1 * cosX - z1 * sinX;
            const z2 = y1 * sinX + z1 * cosX;
            // Subtle perspective factor so 3D depth feels natural without distortion
            const persp = 1.0 / (1.0 - Math.max(-0.6, Math.min(0.6, z2 * 0.18)));
            return {
                sx: cx + x2 * scale * persp,
                sy: cy - y2 * scale * persp,
                depth: z2
            };
        };

        // 1. Draw subtle 3D floor grid at y = -1.0
        ctx.save();
        ctx.lineWidth = 1;
        ctx.strokeStyle = palette ? palette.borderSubtle : "rgba(128,128,128,0.16)";
        const gridTicks = [-1.0, -0.5, 0.0, 0.5, 1.0];
        for (const g of gridTicks) {
            const a1 = projectCoord(g, -1.0, -1.0);
            const a2 = projectCoord(g, -1.0, 1.0);
            ctx.beginPath();
            ctx.moveTo(a1.sx, a1.sy);
            ctx.lineTo(a2.sx, a2.sy);
            ctx.stroke();

            const b1 = projectCoord(-1.0, -1.0, g);
            const b2 = projectCoord(1.0, -1.0, g);
            ctx.beginPath();
            ctx.moveTo(b1.sx, b1.sy);
            ctx.lineTo(b2.sx, b2.sy);
            ctx.stroke();
        }
        ctx.restore();

        // 2. Draw 3D coordinate axes through origin (0, 0, 0)
        const v3 = this.variance3d || [];
        const pc1Pct = v3[0] !== undefined ? ` (${(v3[0] * 100).toFixed(1)}%)` : "";
        const pc2Pct = v3[1] !== undefined ? ` (${(v3[1] * 100).toFixed(1)}%)` : "";
        const pc3Pct = v3[2] !== undefined ? ` (${(v3[2] * 100).toFixed(1)}%)` : "";
        const axisLen = 1.18;
        const axes = [
            { label: `PC1${pc1Pct}`, x: axisLen, y: 0, z: 0, color: "#0090ff" },
            { label: `PC2${pc2Pct}`, x: 0, y: axisLen, z: 0, color: "#30a46c" },
            { label: `PC3${pc3Pct}`, x: 0, y: 0, z: axisLen, color: "#ffc53d" }
        ];

        const origin = projectCoord(0, 0, 0);
        ctx.save();
        ctx.font = '500 11px "JetBrains Mono", monospace';
        axes.forEach(a => {
            const neg = projectCoord(-a.x * 0.85, -a.y * 0.85, -a.z * 0.85);
            const end = projectCoord(a.x, a.y, a.z);

            ctx.beginPath();
            ctx.setLineDash([3, 3]);
            ctx.moveTo(neg.sx, neg.sy);
            ctx.lineTo(origin.sx, origin.sy);
            ctx.strokeStyle = palette ? palette.borderDefault : "rgba(128,128,128,0.35)";
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.beginPath();
            ctx.setLineDash([]);
            ctx.moveTo(origin.sx, origin.sy);
            ctx.lineTo(end.sx, end.sy);
            ctx.strokeStyle = a.color;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ctx.fillStyle = palette ? palette.textSecondary : "#b0b4ba";
            ctx.fillText(a.label, end.sx + 6, end.sy + 4);
        });
        ctx.restore();

        if (!this.points || this.points.length === 0) {
            ctx.save();
            ctx.font = '500 13px "Inter", sans-serif';
            ctx.fillStyle = palette ? palette.textMuted : "#777b84";
            ctx.textAlign = "center";
            ctx.fillText("Run clustering & PCA to project samples in 3D space", cx, 32);
            ctx.restore();
            this.projectedPoints = [];
            return;
        }

        // 3. Normalize 3D point cloud into [-1, 1]^3 so any PCA scale fits the 3D cube cleanly
        let sumX = 0, sumY = 0, sumZ = 0;
        const n = this.points.length;
        for (let pt of this.points) {
            sumX += Number(pt.x) || 0;
            sumY += Number(pt.y) || 0;
            sumZ += Number(pt.z) || 0;
        }
        const meanX = sumX / n;
        const meanY = sumY / n;
        const meanZ = sumZ / n;

        let maxSpan = 1e-6;
        for (let pt of this.points) {
            const dx = Math.abs((Number(pt.x) || 0) - meanX);
            const dy = Math.abs((Number(pt.y) || 0) - meanY);
            const dz = Math.abs((Number(pt.z) || 0) - meanZ);
            if (dx > maxSpan) maxSpan = dx;
            if (dy > maxSpan) maxSpan = dy;
            if (dz > maxSpan) maxSpan = dz;
        }

        const projected = [];
        const clusterCounts = {};
        for (let pt of this.points) {
            const nx = ((Number(pt.x) || 0) - meanX) / maxSpan;
            const ny = ((Number(pt.y) || 0) - meanY) / maxSpan;
            const nz = ((Number(pt.z) || 0) - meanZ) / maxSpan;

            const proj = projectCoord(nx, ny, nz);
            projected.push({
                pt: pt,
                sx: proj.sx,
                sy: proj.sy,
                depth: proj.depth
            });
            const cid = pt.cluster;
            clusterCounts[cid] = (clusterCounts[cid] || 0) + 1;
        }

        // Sort back-to-front by depth for proper painter's algorithm occlusion
        projected.sort((a, b) => a.depth - b.depth);

        const strokeColor = theme === "light" ? "rgba(255,255,255,0.85)" : "rgba(17,17,19,0.85)";
        for (let p of projected) {
            const radius = Math.max(3.5, Math.min(6.0, 4.5 + p.depth * 1.0));
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, radius, 0, Math.PI * 2);
            ctx.fillStyle = this.getClusterColor(p.pt.cluster);
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = strokeColor;
            ctx.stroke();
        }

        // 4. Draw compact cluster legend in top-left corner
        const clusterKeys = Object.keys(clusterCounts).sort((a, b) => Number(a) - Number(b));
        ctx.save();
        ctx.font = '500 11px "Inter", sans-serif';
        let legendY = 22;
        for (const cidKey of clusterKeys) {
            const cid = Number(cidKey);
            const label = cid === -1 ? `Noise (${clusterCounts[cidKey]})` : `Cluster ${cid} (${clusterCounts[cidKey]})`;
            ctx.beginPath();
            ctx.arc(20, legendY - 4, 4.5, 0, Math.PI * 2);
            ctx.fillStyle = this.getClusterColor(cid);
            ctx.fill();
            ctx.fillStyle = palette ? palette.textSecondary : "#b0b4ba";
            ctx.fillText(label, 30, legendY);
            legendY += 18;
        }
        ctx.restore();

        this.projectedPoints = projected;
    }

    handle3DClick(e) {
        if (!this.projectedPoints || !this.onClusterClick) return;
        const rect = this.canvas3d.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        for (let i = this.projectedPoints.length - 1; i >= 0; i--) {
            const p = this.projectedPoints[i];
            if (Math.hypot(p.sx - mx, p.sy - my) < 8) {
                this.onClusterClick(p.pt.cluster);
                return;
            }
        }
    }

    check3DHover(e) {
        if (!this.projectedPoints) return;
        const rect = this.canvas3d.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        let hit = null;
        for (let i = this.projectedPoints.length - 1; i >= 0; i--) {
            const p = this.projectedPoints[i];
            if (Math.hypot(p.sx - mx, p.sy - my) < 7) {
                hit = p;
                break;
            }
        }

        const tooltip = document.getElementById("pca-tooltip");
        if (hit && tooltip) {
            this.canvas3d.style.cursor = "pointer";
            tooltip.classList.remove("is-hidden");
            tooltip.style.display = "block";
            tooltip.style.left = `${Math.min(mx + 14, (this._cssWidth || 600) - 180)}px`;
            tooltip.style.top = `${Math.max(10, my + 10)}px`;
            const idLabel = hit.pt.input_id ? `ID: ${hit.pt.input_id} (#${hit.pt.sample_id})` : `Sample #${hit.pt.sample_id}`;
            const zVal = hit.pt.z !== undefined ? Number(hit.pt.z).toFixed(2) : "0.00";
            tooltip.innerHTML = `
                <div class="fw-600">${idLabel}</div>
                <div class="text-secondary">Cluster: ${hit.pt.cluster === -1 ? 'Noise' : hit.pt.cluster}</div>
                <div class="mono text-micro text-muted">PC1: ${Number(hit.pt.x).toFixed(2)}, PC2: ${Number(hit.pt.y).toFixed(2)}, PC3: ${zVal}</div>
                ${hit.pt.ground_truth !== undefined ? `<div class="text-secondary text-micro">Ground Truth: ${hit.pt.ground_truth}</div>` : ''}
                <div class="text-accent text-micro mt-1">Click to inspect cluster waveforms</div>
            `;
        } else if (tooltip) {
            if (!this.isDragging && this.canvas3d) this.canvas3d.style.cursor = "grab";
            tooltip.classList.add("is-hidden");
            tooltip.style.display = "none";
        }
    }
}

window.PCAVisualizer = PCAVisualizer;
