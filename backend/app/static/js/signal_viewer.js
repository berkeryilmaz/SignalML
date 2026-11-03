// Interactive Highcharts Waveform Visualizer & Inspector

class SignalViewer {
    constructor(containerId, datasetId) {
        this.containerId = containerId;
        this.datasetId = datasetId;
        this.chart = null;
        this.currentIndex = 0;
        this.totalSamples = 0;
        this.overlayIndices = [0];
        this.isOverlayMode = false;
        this.init();
    }

    init() {
        this.setupChart();
        if (this.datasetId) {
            this.loadWaveforms();
        }
    }

    setupChart() {
        this.chart = Highcharts.chart(this.containerId, {
            chart: {
                type: "line",
                backgroundColor: "transparent",
                zoomType: "x",
                panning: true,
                panKey: "shift"
            },
            title: { text: null },
            credits: { enabled: false },
            xAxis: {
                title: { text: "Sample Point Index", style: { color: "#94a3b8" } },
                labels: { style: { color: "#94a3b8" } },
                gridLineColor: "#1e293b"
            },
            yAxis: {
                title: { text: "Amplitude", style: { color: "#94a3b8" } },
                labels: { style: { color: "#94a3b8" } },
                gridLineColor: "#1e293b"
            },
            legend: {
                itemStyle: { color: "#f1f5f9" },
                itemHoverStyle: { color: "#38bdf8" }
            },
            tooltip: {
                shared: true,
                backgroundColor: "#191f2e",
                borderColor: "#334155",
                style: { color: "#f8fafc" },
                valueDecimals: 4
            },
            plotOptions: {
                series: {
                    lineWidth: 1.5,
                    marker: { enabled: false }
                }
            },
            series: []
        });
    }

    async loadWaveforms() {
        const indices = this.isOverlayMode ? this.overlayIndices : [this.currentIndex];
        try {
            const res = await fetch("/api/signals/waveform", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    dataset_id: this.datasetId,
                    sample_indices: indices,
                    max_points: 3000
                })
            });

            const data = await SignalML.safeJson(res);
            if (!res.ok) {
                SignalML.toast(`Failed to load waveforms: ${data.detail || data.error || res.statusText}`, "error");
                return;
            }

            this.totalSamples = data.total_samples || 0;
            if (data.waveforms && data.waveforms.length > 0) {
                this.updateStats(data.waveforms[0]);
                this.renderSeries(data.waveforms);
            }
            this.updateNavigationUI();
        } catch (e) {
            SignalML.toast(`Error: ${e.message}`, "error");
        }
    }

    renderSeries(waveforms) {
        while (this.chart.series.length > 0) {
            this.chart.series[0].remove(false);
        }

        const colors = ["#38bdf8", "#34d399", "#f43f5e", "#fbbf24", "#a855f7", "#ec4899", "#f97316"];

        waveforms.forEach((wf, i) => {
            const seriesData = wf.x.map((xVal, idx) => [xVal, wf.y[idx]]);
            this.chart.addSeries({
                name: `Sample #${wf.sample_index} ${wf.label !== null ? `(Label: ${wf.label})` : ""}`,
                data: seriesData,
                color: colors[i % colors.length]
            }, false);
        });

        this.chart.redraw();
    }

    updateStats(wf) {
        if (!wf) return;
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        const minVal = (wf.min !== undefined && wf.min !== null && !isNaN(wf.min)) ? Number(wf.min).toFixed(3) : "0.000";
        const maxVal = (wf.max !== undefined && wf.max !== null && !isNaN(wf.max)) ? Number(wf.max).toFixed(3) : "0.000";
        const meanVal = (wf.mean !== undefined && wf.mean !== null && !isNaN(wf.mean)) ? Number(wf.mean).toFixed(3) : "0.000";
        const stdVal = (wf.std !== undefined && wf.std !== null && !isNaN(wf.std)) ? Number(wf.std).toFixed(3) : "0.000";

        setVal("stat-sample-idx", `#${wf.sample_index}`);
        setVal("stat-signal-len", `${wf.length || 0} points`);
        setVal("stat-min-max", `[${minVal}, ${maxVal}]`);
        setVal("stat-mean-std", `μ: ${meanVal}, σ: ${stdVal}`);
        setVal("stat-label", wf.label !== null && wf.label !== undefined ? wf.label : "None");
    }

    updateNavigationUI() {
        const counterEl = document.getElementById("sample-counter");
        if (counterEl) {
            counterEl.textContent = `${this.currentIndex + 1} / ${this.totalSamples}`;
        }
    }

    nextSample() {
        if (this.currentIndex < this.totalSamples - 1) {
            this.currentIndex++;
            this.loadWaveforms();
        }
    }

    prevSample() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            this.loadWaveforms();
        }
    }

    jumpToSample(idx) {
        if (idx >= 0 && idx < this.totalSamples) {
            this.currentIndex = idx;
            this.loadWaveforms();
        }
    }

    toggleOverlay(sampleIdx) {
        const found = this.overlayIndices.indexOf(sampleIdx);
        if (found >= 0) {
            if (this.overlayIndices.length > 1) {
                this.overlayIndices.splice(found, 1);
            }
        } else {
            if (this.overlayIndices.length < 10) {
                this.overlayIndices.push(sampleIdx);
            }
        }
        this.isOverlayMode = this.overlayIndices.length > 1;
        this.loadWaveforms();
    }

    async loadGroupAnalysis(labelValue = null) {
        try {
            const payload = {
                dataset_id: this.datasetId,
                max_waveforms: 80,
                max_points: 1200
            };
            if (labelValue !== null && labelValue !== undefined) {
                payload.label_value = labelValue;
            }

            const res = await fetch("/api/signals/group_waveform_analysis", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await SignalML.safeJson(res);
            if (!res.ok) {
                SignalML.toast(`Group analysis failed: ${data.detail || data.error || res.statusText}`, "error");
                return;
            }

            // Clear chart
            while (this.chart.series.length > 0) {
                this.chart.series[0].remove(false);
            }

            // 1. Faded waveforms
            data.waveforms.forEach((wf) => {
                const points = data.x.map((xv, idx) => [xv, wf.y[idx]]);
                this.chart.addSeries({
                    name: `Sample #${wf.sample_index}`,
                    type: "line",
                    data: points,
                    color: "rgba(56, 189, 248, 0.18)", // Faded / transparent
                    lineWidth: 1,
                    enableMouseTracking: false,
                    showInLegend: false,
                    marker: { enabled: false }
                }, false);
            });

            // 2. Std Dev Bounds
            if (data.upper_std && data.lower_std) {
                this.chart.addSeries({
                    name: "+1σ Bound",
                    type: "line",
                    data: data.x.map((xv, idx) => [xv, data.upper_std[idx]]),
                    color: "#94a3b8",
                    dashStyle: "Dash",
                    lineWidth: 1,
                    opacity: 0.5,
                    enableMouseTracking: false,
                    marker: { enabled: false }
                }, false);
                this.chart.addSeries({
                    name: "-1σ Bound",
                    type: "line",
                    data: data.x.map((xv, idx) => [xv, data.lower_std[idx]]),
                    color: "#94a3b8",
                    dashStyle: "Dash",
                    lineWidth: 1,
                    opacity: 0.5,
                    enableMouseTracking: false,
                    marker: { enabled: false }
                }, false);
            }

            // 3. Prominent Average Waveform
            this.chart.addSeries({
                name: `Group Mean / Ortalama (N=${data.sample_count})`,
                type: "line",
                data: data.x.map((xv, idx) => [xv, data.mean[idx]]),
                color: "#ffffff",
                lineWidth: 3.5,
                zIndex: 100,
                shadow: {
                    color: "#38bdf8",
                    width: 6,
                    opacity: 0.8
                },
                marker: { enabled: false }
            }, false);

            this.chart.redraw();
            SignalML.toast(`Showing group overlay with ${data.sample_count} faded waveforms + prominent average`, "success");
        } catch (e) {
            SignalML.toast(`Network error: ${e.message}`, "error");
        }
    }
}

window.SignalViewer = SignalViewer;
