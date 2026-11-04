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
        this.chart = Highcharts.chart(this.chartContainerId, {
            chart: {
                type: "spline",
                backgroundColor: "transparent"
            },
            title: { text: null },
            credits: { enabled: false },
            xAxis: {
                title: { text: "Epoch", style: { color: "#94a3b8" } },
                labels: { style: { color: "#94a3b8" } },
                gridLineColor: "#1e293b",
                allowDecimals: false
            },
            yAxis: [
                {
                    // yAxis 0 (Left): Loss
                    title: { text: "Loss (Kayıp)", style: { color: "#38bdf8" } },
                    labels: {
                        style: { color: "#94a3b8" },
                        formatter: function () {
                            return (typeof this.value === "number") ? this.value.toFixed(4) : this.value;
                        }
                    },
                    gridLineColor: "#1e293b",
                    min: 0
                },
                {
                    // yAxis 1 (Right): Accuracy
                    title: { text: "Accuracy (Doğruluk)", style: { color: "#34d399" } },
                    labels: {
                        style: { color: "#34d399" },
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
            legend: {
                itemStyle: { color: "#f1f5f9" }
            },
            tooltip: {
                shared: true,
                backgroundColor: "#191f2e",
                borderColor: "#334155",
                style: { color: "#f8fafc" },
                valueDecimals: 4,
                pointFormatter: function () {
                    const isAcc = this.series.name.toLowerCase().includes("accur") || this.series.name.toLowerCase().includes("doğruluk");
                    let valStr = (typeof this.y === "number") ? this.y.toFixed(4) : this.y;
                    if (isAcc && typeof this.y === "number" && this.y <= 1.0) {
                        valStr += ` (${(this.y * 100).toFixed(2)}%)`;
                    }
                    return `<span style="color:${this.color}">●</span> ${this.series.name}: <b>${valStr}</b><br/>`;
                }
            },
            series: [
                { name: "Training Loss", data: [], color: "#38bdf8", yAxis: 0 },
                { name: "Validation Loss", data: [], color: "#f43f5e", dashStyle: "ShortDash", yAxis: 0 },
                { name: "Training Accuracy", data: [], color: "#34d399", yAxis: 1 },
                { name: "Validation Accuracy", data: [], color: "#a78bfa", dashStyle: "ShortDash", yAxis: 1 }
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
                SignalML.toast("Eğitim kaldığı yerden devam ettiriliyor (Resume Training)...", "info");
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
                errBanner.style.display = "block";
                errBanner.innerHTML = `<strong>⚠️ Eğitim Durumu:</strong> ${status.error || "Bilinmeyen bir hata nedeniyle eğitim tamamlanamadı."}`;
            } else {
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
                this.chart.yAxis[1].update({ title: { text: "Accuracy (Doğruluk)" }, max: 1.0 }, false);
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
                this.chart.yAxis[1].update({ title: { text: "MAE (Hata)" }, max: null }, false);
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

        let html = `
            <div class="card" style="margin-top: 1.5rem;">
                <div class="card-title">Model Evaluation Metrics</div>
        `;

        if (evalData.error) {
            html += `
                <div style="padding: 0.75rem 1rem; background: rgba(244, 63, 94, 0.15); border: 1px solid var(--accent-rose); border-radius: 6px; color: #fca5a5; margin-bottom: 1rem; font-size: 13px;">
                    <strong>Evaluation Warning:</strong> ${evalData.error}
                </div>
            `;
        }

        // Check if IDEC Clustering evaluation
        if (evalData.task_type === "idec_clustering" || evalData.n_clusters !== undefined) {
            const confStr = evalData.mean_cluster_confidence != null 
                ? (evalData.mean_cluster_confidence * 100).toFixed(1) + "%" 
                : "—";

            html += `
                <div class="grid-cards" style="margin-bottom: 1.25rem;">
                    <div class="card">
                        <div class="card-title">Küme Sayısı (K)</div>
                        <div class="stat-val" style="color: var(--accent-cyan, #38bdf8);">${evalData.n_clusters}</div>
                        <div class="stat-desc">Learned Clusters</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Ortalama Küme Güveni</div>
                        <div class="stat-val" style="color: var(--accent-green, #4ade80);">${confStr}</div>
                        <div class="stat-desc">Mean Confidence (max Q)</div>
                    </div>
                    ${evalData.silhouette_score != null ? `
                    <div class="card">
                        <div class="card-title">Silhouette Skoru</div>
                        <div class="stat-val" style="color: #a78bfa;">${typeof evalData.silhouette_score === "number" ? evalData.silhouette_score.toFixed(4) : evalData.silhouette_score}</div>
                        <div class="stat-desc">Separation (-1 to +1)</div>
                    </div>` : ""}
                    <div class="card">
                        <div class="card-title">Rekonstrüksiyon MSE</div>
                        <div class="stat-val" style="font-size: 20px; font-family: var(--font-mono); color: #f59e0b;">${evalData.reconstruction_mse !== undefined ? evalData.reconstruction_mse : "—"}</div>
                        <div class="stat-desc">AE Reconstruction Loss</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Rekonstrüksiyon MAE</div>
                        <div class="stat-val" style="font-size: 20px; font-family: var(--font-mono); color: #f59e0b;">${evalData.reconstruction_mae !== undefined ? evalData.reconstruction_mae : "—"}</div>
                        <div class="stat-desc">AE Reconstruction MAE</div>
                    </div>
                </div>
            `;

            // Cluster Distribution
            const dist = evalData.cluster_distribution || {};
            const clusterEntries = Object.entries(dist);
            const totalSamples = clusterEntries.reduce((sum, [, count]) => sum + (Number(count) || 0), 0);

            if (clusterEntries.length > 0) {
                const palette = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb7185", "#fb923c", "#facc15", "#4ade80", "#2dd4bf", "#94a3b8"];
                html += `
                    <div style="margin-bottom: 1.25rem; background: rgba(15, 23, 42, 0.4); border: 1px solid #1e293b; border-radius: 8px; padding: 1rem;">
                        <div style="font-weight: 600; font-size: 14px; margin-bottom: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
                            <span>📊 Küme Dağılımı (Cluster Distribution)</span>
                            <span style="font-size: 12px; color: var(--text-muted);">Toplam: ${totalSamples.toLocaleString()} örnek</span>
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 8px;">
                `;

                clusterEntries.forEach(([cLabel, count], idx) => {
                    const cnt = Number(count) || 0;
                    const pct = totalSamples > 0 ? ((cnt / totalSamples) * 100).toFixed(1) : "0.0";
                    const color = palette[idx % palette.length];
                    html += `
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
                                <span><strong style="color: ${color};">Cluster ${cLabel}</strong></span>
                                <span><span style="font-family: var(--font-mono); font-weight: 600;">${cnt.toLocaleString()}</span> örnek (${pct}%)</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.06); height: 8px; border-radius: 4px; overflow: hidden;">
                                <div style="background: ${color}; width: ${pct}%; height: 100%; border-radius: 4px; transition: width 0.4s ease;"></div>
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
                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1rem;">
                    <a href="/clustering" class="btn btn-primary" style="text-decoration: none; display: inline-flex; align-items: center; gap: 6px;">
                        🌐 Clustering & 3D PCA Sayfasında İncele
                    </a>
                    <a href="/nn-builder" class="btn btn-secondary" style="text-decoration: none; display: inline-flex; align-items: center; gap: 6px;">
                        🪄 NN Model Sihirbazı
                    </a>
                </div>
            `;
        } else if (evalData.accuracy !== undefined && evalData.accuracy !== null) {
            html += `
                <div class="grid-cards" style="margin-bottom: 1rem;">
                    <div class="card">
                        <div class="card-title">Test Accuracy</div>
                        <div class="stat-val">${(evalData.accuracy * 100).toFixed(2)}%</div>
                    </div>
                </div>
            `;
        } else if (evalData.rmse !== undefined && evalData.rmse !== null) {
            html += `
                <div class="grid-cards" style="margin-bottom: 1rem;">
                    <div class="card">
                        <div class="card-title">RMSE</div>
                        <div class="stat-val">${evalData.rmse}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">MAE</div>
                        <div class="stat-val">${evalData.mae}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">R² Score</div>
                        <div class="stat-val">${evalData.r2}</div>
                    </div>
                </div>
            `;
        }

        // Confusion matrix if available
        if (evalData.confusion_matrix) {
            html += `
                <div style="font-weight: 600; margin-bottom: 0.5rem;">Confusion Matrix</div>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>True \\ Pred</th>
                                ${evalData.class_labels.map(l => `<th>Class ${l}</th>`).join("")}
                            </tr>
                        </thead>
                        <tbody>
                            ${evalData.confusion_matrix.map((row, i) => `
                                <tr>
                                    <td><strong>Class ${evalData.class_labels[i]}</strong></td>
                                    ${row.map(val => `<td style="font-family: var(--font-mono); font-weight: 600;">${val}</td>`).join("")}
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
