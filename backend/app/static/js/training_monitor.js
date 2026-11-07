// Real-Time Training Monitor, Metrics Plotter, and Evaluation Viewer

class TrainingMonitor {
    constructor(chartContainerId) {
        this.chartContainerId = chartContainerId;
        this.chart = null;
        this.activeJobId = null;
        this.pollInterval = null;
        this.initChart();
    }

    initChart() {
        if (window.SignalML && typeof window.SignalML.chartTheme === "function") {
            window.SignalML.chartTheme();
        }
        this.chart = Highcharts.chart(this.chartContainerId, {
            chart: {
                type: "spline",
                backgroundColor: "transparent"
            },
            title: { text: null },
            credits: { enabled: false },
            xAxis: {
                title: { text: "Epoch" },
                allowDecimals: false
            },
            yAxis: [
                {
                    // yAxis 0 (Left): Loss
                    title: { text: "Loss" },
                    labels: {
                        formatter: function () {
                            return (typeof this.value === "number") ? this.value.toFixed(4) : this.value;
                        }
                    },
                    min: 0
                },
                {
                    // yAxis 1 (Right): Accuracy
                    title: { text: "Accuracy" },
                    labels: {
                        formatter: function () {
                            if (typeof this.value !== "number") return this.value;
                            return (this.value <= 1.0) ? (this.value * 100).toFixed(0) + "%" : this.value.toFixed(2);
                        }
                    },
                    opposite: true,
                    min: 0,
                    max: 1.0,
                    gridLineWidth: 0
                }
            ],
            tooltip: {
                shared: true,
                valueDecimals: 4,
                pointFormatter: function () {
                    const isAcc = this.series.name.toLowerCase().includes("accur");
                    let valStr = (typeof this.y === "number") ? this.y.toFixed(4) : this.y;
                    if (isAcc && typeof this.y === "number" && this.y <= 1.0) {
                        valStr += ` (${(this.y * 100).toFixed(2)}%)`;
                    }
                    return `<span style="color:${this.color}">\u25CF</span> ${this.series.name}: <b>${valStr}</b><br/>`;
                }
            },
            series: [
                { name: "Training Loss", data: [], color: "#0090ff", yAxis: 0 },
                { name: "Validation Loss", data: [], color: "#e5484d", dashStyle: "ShortDash", yAxis: 0 },
                { name: "Training Accuracy", data: [], color: "#30a46c", yAxis: 1 },
                { name: "Validation Accuracy", data: [], color: "#8e4ec6", dashStyle: "ShortDash", yAxis: 1 }
            ]
        });
    }

    extractAccuracySeries(hist, isVal = false) {
        if (!hist || typeof hist !== "object") return [];

        // 1. Direct standard keys
        const primaryKeys = isVal
            ? ["val_accuracy", "val_acc", "val_categorical_accuracy", "val_sparse_categorical_accuracy", "val_recon_accuracy"]
            : ["accuracy", "acc", "categorical_accuracy", "sparse_categorical_accuracy", "recon_accuracy"];

        for (const k of primaryKeys) {
            if (Array.isArray(hist[k]) && hist[k].length > 0) {
                return hist[k];
            }
        }

        // 2. Multi-output search: Look strictly for accuracy keys and strictly EXCLUDE any loss/mae/mse/error
        for (const [key, val] of Object.entries(hist)) {
            if (!Array.isArray(val) || val.length === 0) continue;
            const lk = key.toLowerCase();

            // Strict exclusion: NEVER allow loss, mae, mse, or error metrics to be matched as accuracy
            if (lk.includes("loss") || lk.includes("mae") || lk.includes("mse") || lk.includes("error")) {
                continue;
            }

            const prefixMatch = isVal ? lk.startsWith("val_") : !lk.startsWith("val_");
            if (prefixMatch && (lk.includes("accuracy") || lk.includes("acc"))) {
                return val;
            }
        }

        return [];
    }

    async startJob(config) {
        try {
            const res = await fetch("/api/training/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(config)
            });

            if (!res.ok) {
                const err = await SignalML.safeJson(res);
                SignalML.toast(`Failed to start training: ${err.detail || "Server error"}`, "error");
                return;
            }

            const data = await SignalML.safeJson(res);
            this.activeJobId = data.job_id;
            if (config.resume) {
                SignalML.toast("Resuming training job...", "info");
            } else {
                SignalML.toast("Training dispatched in background thread!", "success");
            }
            this.startPolling();
        } catch (e) {
            SignalML.toast(`Network Error: ${e.message}`, "error");
        }
    }

    startPolling() {
        if (this.pollInterval) clearInterval(this.pollInterval);

        this.pollInterval = setInterval(async () => {
            if (!this.activeJobId) return;

            try {
                const res = await fetch(`/api/training/status/${this.activeJobId}`);
                const status = await SignalML.safeJson(res);
                if (!res.ok) {
                    console.error("Status polling response error:", status);
                    return;
                }

                this.updateUI(status);

                if (["completed", "stopped", "failed"].includes(status.status)) {
                    clearInterval(this.pollInterval);
                    if (status.status === "completed") {
                        SignalML.toast("Training completed successfully!", "success");
                    } else if (status.status === "stopped") {
                        SignalML.toast("Training stopped by user.", "info");
                    } else {
                        SignalML.toast(`Training failed: ${status.error || "Unexpected training error"}`, "error");
                    }
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        }, 1000);
    }

    async cancelJob() {
        if (!this.activeJobId) return;
        try {
            const res = await fetch(`/api/training/cancel/${this.activeJobId}`, { method: "POST" });
            if (res.ok) {
                SignalML.toast("Stopping training job...", "info");
            }
        } catch (e) {
            SignalML.toast(`Cancel error: ${e.message}`, "error");
        }
    }

    updateCharts(statusData) {
        this.updateUI(statusData);
    }

    updateUI(status) {
        // Update pills
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setVal("train-stat-epoch", `${status.current_epoch} / ${status.total_epochs}`);
        setVal("train-stat-status", (status.status || "IDLE").toUpperCase());
        setVal("train-stat-elapsed", `${(status.elapsed_seconds || 0).toFixed(1)}s`);

        // Error banner
        const errBanner = document.getElementById("train-error-banner");
        if (errBanner) {
            if (status.error || status.status === "failed") {
                const warnIcon = (window.SignalML && typeof window.SignalML.icon === "function") ? window.SignalML.icon("triangle-exclamation-point") : "";
                errBanner.hidden = false;
                errBanner.style.display = "flex";
                errBanner.innerHTML = `${warnIcon}<div class="alert-body"><span class="alert-title">Training status:</span> ${status.error || "Training failed due to an unexpected error."}</div>`;
            } else {
                errBanner.hidden = true;
                errBanner.style.display = "none";
            }
        }

        const hist = status.history || {};
        const lossData = hist.loss || [];
        const valLossData = hist.val_loss || [];

        let accData = this.extractAccuracySeries(hist, false);
        let valAccData = this.extractAccuracySeries(hist, true);
        const hasAccuracy = accData.length > 0;

        let metricLabel = "Model Accuracy";
        let metricDisplayVal = "—";

        if (hasAccuracy) {
            // Configure series for actual Accuracy (0 to 1.0 / 0% to 100%)
            if (this.chart.series[2] && this.chart.series[2].name !== "Training Accuracy") {
                this.chart.series[2].update({ name: "Training Accuracy" }, false);
            }
            if (this.chart.series[3] && this.chart.series[3].name !== "Validation Accuracy") {
                this.chart.series[3].update({ name: "Validation Accuracy" }, false);
            }
            if (this.chart.yAxis[1]) {
                this.chart.yAxis[1].update({ title: { text: "Accuracy" }, max: 1.0 }, false);
            }

            const lastAcc = accData[accData.length - 1];
            if (typeof lastAcc === "number" && isFinite(lastAcc)) {
                metricDisplayVal = (lastAcc <= 1.0) ? `${(lastAcc * 100).toFixed(2)}%` : lastAcc.toFixed(4);
            }
        } else if (Array.isArray(hist.mae) && hist.mae.length > 0) {
            // Regression fallback: Do NOT disguise error as accuracy!
            // Explicitly label as MAE (Error)
            accData = hist.mae;
            valAccData = hist.val_mae || [];
            if (this.chart.series[2] && this.chart.series[2].name !== "Training MAE (Error)") {
                this.chart.series[2].update({ name: "Training MAE (Error)" }, false);
            }
            if (this.chart.series[3] && this.chart.series[3].name !== "Validation MAE (Error)") {
                this.chart.series[3].update({ name: "Validation MAE (Error)" }, false);
            }
            if (this.chart.yAxis[1]) {
                this.chart.yAxis[1].update({ title: { text: "MAE (Error)" }, max: null }, false);
            }

            metricLabel = "Mean Absolute Error";
            const lastMae = accData[accData.length - 1];
            if (typeof lastMae === "number" && isFinite(lastMae)) {
                metricDisplayVal = lastMae.toFixed(4);
            }
        } else {
            accData = [];
            valAccData = [];
        }

        setVal("train-stat-accuracy", metricDisplayVal);
        setVal("train-stat-acc-label", metricLabel);

        const sanitizeNum = (v) => (typeof v === "number" && !isNaN(v) && isFinite(v)) ? v : null;

        if (lossData.length > 0) {
            const lastL = lossData[lossData.length - 1];
            setVal("train-stat-loss", (typeof lastL === "number" && isFinite(lastL)) ? lastL.toFixed(4) : "Diverged");
        }
        if (valLossData.length > 0) {
            const lastVL = valLossData[valLossData.length - 1];
            setVal("train-stat-val-loss", (typeof lastVL === "number" && isFinite(lastVL)) ? lastVL.toFixed(4) : "—");
        }

        // Update Highcharts Series safely (all 4 series)
        if (this.chart.series[0]) this.chart.series[0].setData(lossData.map((v, i) => [i + 1, sanitizeNum(v)]), false);
        if (this.chart.series[1]) this.chart.series[1].setData(valLossData.map((v, i) => [i + 1, sanitizeNum(v)]), false);
        if (this.chart.series[2]) this.chart.series[2].setData(accData.map((v, i) => [i + 1, sanitizeNum(v)]), false);
        if (this.chart.series[3]) this.chart.series[3].setData(valAccData.map((v, i) => [i + 1, sanitizeNum(v)]), false);
        this.chart.redraw();

        // Render evaluation results if finished
        if (status.evaluation) {
            this.renderEvaluation(status.evaluation);
        }
    }

    renderEvaluation(evalData) {
        const container = document.getElementById("evaluation-results-container");
        if (!container) return;

        const iconFn = (window.SignalML && typeof window.SignalML.icon === "function") ? window.SignalML.icon.bind(window.SignalML) : (() => "");

        let html = `
            <div class="card">
                <div class="card-title">Model Evaluation Metrics</div>
        `;

        if (evalData.error) {
            html += `
                <div class="alert alert-danger mb-4">
                    ${iconFn("triangle-exclamation-point")}
                    <div class="alert-body"><span class="alert-title">Evaluation warning:</span> ${evalData.error}</div>
                </div>
            `;
        }

        // Check if IDEC Clustering evaluation
        if (evalData.task_type === "idec_clustering" || evalData.n_clusters !== undefined) {
            const confStr = evalData.mean_cluster_confidence != null 
                ? (evalData.mean_cluster_confidence * 100).toFixed(1) + "%" 
                : "—";

            html += `
                <div class="grid-cards mb-4">
                    <div class="card">
                        <div class="card-title">Clusters (K)</div>
                        <div class="stat-val mono text-info">${evalData.n_clusters}</div>
                        <div class="stat-desc">Learned Clusters</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Mean Cluster Confidence</div>
                        <div class="stat-val mono text-success">${confStr}</div>
                        <div class="stat-desc">Mean Confidence (max Q)</div>
                    </div>
                    ${evalData.silhouette_score != null ? `
                    <div class="card">
                        <div class="card-title">Silhouette Score</div>
                        <div class="stat-val mono">${typeof evalData.silhouette_score === "number" ? evalData.silhouette_score.toFixed(4) : evalData.silhouette_score}</div>
                        <div class="stat-desc">Separation (-1 to +1)</div>
                    </div>` : ""}
                    <div class="card">
                        <div class="card-title">Reconstruction MSE</div>
                        <div class="stat-val mono text-warning">${evalData.reconstruction_mse !== undefined ? evalData.reconstruction_mse : "—"}</div>
                        <div class="stat-desc">AE Reconstruction Loss</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Reconstruction MAE</div>
                        <div class="stat-val mono text-warning">${evalData.reconstruction_mae !== undefined ? evalData.reconstruction_mae : "—"}</div>
                        <div class="stat-desc">AE Reconstruction MAE</div>
                    </div>
                </div>
            `;

            // Cluster Distribution
            const dist = evalData.cluster_distribution || {};
            const clusterEntries = Object.entries(dist);
            const totalSamples = clusterEntries.reduce((sum, [, count]) => sum + (Number(count) || 0), 0);

            if (clusterEntries.length > 0) {
                const palette = ["#0090ff", "#30a46c", "#ffc53d", "#e5484d", "#8e4ec6", "#00b8d9", "#f76b15", "#a0a7b4"];
                html += `
                    <div class="tile mb-4">
                        <div class="row-between mb-3">
                            <span class="row fw-600 text-sm">${iconFn("chart-column", "icon-sm")} <span>Cluster Distribution</span></span>
                            <span class="text-xs text-muted mono">Total: ${totalSamples.toLocaleString()} samples</span>
                        </div>
                        <div class="stack-sm">
                `;

                clusterEntries.forEach(([cLabel, count], idx) => {
                    const cnt = Number(count) || 0;
                    const pct = totalSamples > 0 ? ((cnt / totalSamples) * 100).toFixed(1) : "0.0";
                    const color = palette[idx % palette.length];
                    html += `
                        <div>
                            <div class="row-between text-xs mb-1">
                                <span class="row"><span class="status-dot" style="background: ${color};"></span><strong>Cluster ${cLabel}</strong></span>
                                <span class="mono"><span class="fw-600">${cnt.toLocaleString()}</span> samples (${pct}%)</span>
                            </div>
                            <div class="progress progress-sm">
                                <div class="progress-bar" style="background: ${color}; width: ${pct}%;"></div>
                            </div>
                        </div>
                    `;
                });

                html += `
                        </div>
                    </div>
                `;
            }

            // Action Buttons to Clustering and NN Builder
            html += `
                <div class="cluster mt-3">
                    <a href="/clustering" class="btn btn-primary">
                        ${iconFn("orbit")} <span>Inspect in Clustering &amp; 3D PCA</span>
                    </a>
                    <a href="/nn-builder" class="btn btn-secondary">
                        ${iconFn("wand-sparkles")} <span>Architecture wizard</span>
                    </a>
                </div>
            `;
        } else if (evalData.accuracy !== undefined && evalData.accuracy !== null) {
            html += `
                <div class="grid-cards mb-4">
                    <div class="card">
                        <div class="card-title">Test Accuracy</div>
                        <div class="stat-val mono">${(evalData.accuracy * 100).toFixed(2)}%</div>
                    </div>
                </div>
            `;
        } else if (evalData.rmse !== undefined && evalData.rmse !== null) {
            html += `
                <div class="grid-cards mb-4">
                    <div class="card">
                        <div class="card-title">RMSE</div>
                        <div class="stat-val mono">${evalData.rmse}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">MAE</div>
                        <div class="stat-val mono">${evalData.mae}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">R&sup2; Score</div>
                        <div class="stat-val mono">${evalData.r2}</div>
                    </div>
                </div>
            `;
        }

        // Confusion matrix if available
        if (evalData.confusion_matrix) {
            html += `
                <div class="section-title">Confusion Matrix</div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>True \\ Pred</th>
                                ${evalData.class_labels.map(l => `<th class="num">Class ${l}</th>`).join("")}
                            </tr>
                        </thead>
                        <tbody>
                            ${evalData.confusion_matrix.map((row, i) => `
                                <tr>
                                    <td><strong>Class ${evalData.class_labels[i]}</strong></td>
                                    ${row.map(val => `<td class="num fw-600">${val}</td>`).join("")}
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            `;
        }

        html += `</div>`;
        container.innerHTML = html;
    }
}

window.TrainingMonitor = TrainingMonitor;
