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
                gridLineColor: "#1e293b"
            },
            yAxis: {
                title: { text: "Metric Value", style: { color: "#94a3b8" } },
                labels: { style: { color: "#94a3b8" } },
                gridLineColor: "#1e293b"
            },
            legend: {
                itemStyle: { color: "#f1f5f9" }
            },
            tooltip: {
                shared: true,
                backgroundColor: "#191f2e",
                borderColor: "#334155",
                style: { color: "#f8fafc" },
                valueDecimals: 4
            },
            series: [
                { name: "Training Loss", data: [], color: "#38bdf8" },
                { name: "Validation Loss", data: [], color: "#f43f5e" },
                { name: "Accuracy / Metric", data: [], color: "#34d399" }
            ]
        });
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
            SignalML.toast("Training dispatched in background thread!", "success");
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
        const accData = hist.accuracy || hist.acc || hist.mae || [];

        const sanitizeNum = (v) => (typeof v === "number" && !isNaN(v) && isFinite(v)) ? v : null;

        if (lossData.length > 0) {
            const lastL = lossData[lossData.length - 1];
            setVal("train-stat-loss", (typeof lastL === "number" && isFinite(lastL)) ? lastL.toFixed(4) : "Diverged");
        }
        if (valLossData.length > 0) {
            const lastVL = valLossData[valLossData.length - 1];
            setVal("train-stat-val-loss", (typeof lastVL === "number" && isFinite(lastVL)) ? lastVL.toFixed(4) : "—");
        }

        // Update Highcharts Series safely
        this.chart.series[0].setData(lossData.map((v, i) => [i + 1, sanitizeNum(v)]), false);
        this.chart.series[1].setData(valLossData.map((v, i) => [i + 1, sanitizeNum(v)]), false);
        this.chart.series[2].setData(accData.map((v, i) => [i + 1, sanitizeNum(v)]), false);
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

        if (evalData.accuracy !== undefined && evalData.accuracy !== null) {
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
