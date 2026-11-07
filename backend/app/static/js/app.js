// Global SignalML Client Application State & Utilities

const SignalML = {
    currentProjectId: localStorage.getItem("signalml_active_project") || "default-project",
    projects: [],
    activeDataset: null,
    activeModel: null,
    _graphCanvases: new Set(),

    _iconAliases: {
        "alert-triangle": "triangle-exclamation-point",
        "triangle-alert": "triangle-exclamation-point",
        "alert-circle": "circle-exclamation-point",
        "circle-alert": "circle-exclamation-point",
        "alert-octagon": "octagon-exclamation-point",
        "check-circle": "circle-check",
        "check-circle-2": "circle-check",
        "x-circle": "circle-x",
        "help-circle": "circle-question-mark",
        "circle-help": "circle-question-mark",
        "bar-chart": "chart-bar",
        "bar-chart-2": "chart-column",
        "bar-chart-3": "chart-column",
        "line-chart": "chart-line",
        "scatter-chart": "chart-scatter",
        "pie-chart": "chart-pie",
        "area-chart": "chart-area",
        "trash-2": "trash",
        "edit": "pencil",
        "edit-2": "pencil",
        "edit-3": "pen-line",
        "wand-2": "wand-sparkles",
        "loader-2": "loader",
        "filter": "list-filter",
        "sliders": "sliders-horizontal",
        "upload-cloud": "cloud-upload",
        "download-cloud": "cloud-download",
        "unlock": "lock-open",
        "waves": "audio-waveform",
        "file-code-2": "file-code",
        "code-2": "code",
        "play-circle": "circle-play",
        "pause-circle": "circle-pause",
        "stop-circle": "circle-stop",
        "plus-circle": "circle-plus",
        "home": "layout-dashboard",
        "history": "clock",
        "git-fork": "git-branch",
        "sidebar": "panel-left",
        "grid": "grid-3x3"
    },

    icon: function(name, cls = "", label = null) {
        const resolved = this._iconAliases[name] || name;
        const classes = ("icon " + (cls || "")).trim();
        if (label) {
            return `<svg class="${classes}" role="img" aria-label="${this.escapeHtml(label)}"><use href="/static/vendor/lucide/sprite.svg#${resolved}"/></svg>`;
        }
        return `<svg class="${classes}" aria-hidden="true"><use href="/static/vendor/lucide/sprite.svg#${resolved}"/></svg>`;
    },

    getTheme: function() {
        const attr = document.documentElement.getAttribute("data-theme");
        if (attr === "light" || attr === "dark") return attr;
        try {
            const params = new URLSearchParams(window.location.search);
            const qp = params.get("theme");
            if (qp === "light" || qp === "dark") return qp;
            const saved = localStorage.getItem("signalml_theme");
            if (saved === "light" || saved === "dark") return saved;
        } catch (err) { console.error("SML_DRAW_ERR:", err && err.stack || err); }
        if (typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: light)").matches) {
            return "light";
        }
        return "dark";
    },

    getThemePalette: function(theme) {
        const t = (theme || this.getTheme()) === "light" ? "light" : "dark";
        if (t === "light") {
            return {
                theme: "light",
                surface1: "#fcfcfd",
                surface2: "#f9f9fb",
                surface3: "#f0f0f3",
                surface4: "#e8e8ec",
                surface5: "#e0e1e6",
                borderSubtle: "#e8e8ec",
                borderDefault: "#d9d9e0",
                borderStrong: "#cdced6",
                textPrimary: "#1c2024",
                textSecondary: "#60646c",
                textMuted: "#80838d",
                accent: "#0090ff",
                accentSubtle: "#e6f4fe",
                success: "#218358",
                warning: "#ab6400",
                danger: "#ce2c31",
                canvasDot: "#dcdde3"
            };
        }
        return {
            theme: "dark",
            surface1: "#111113",
            surface2: "#18191b",
            surface3: "#212225",
            surface4: "#272a2d",
            surface5: "#2e3135",
            borderSubtle: "#272a2d",
            borderDefault: "#363a3f",
            borderStrong: "#43484e",
            textPrimary: "#edeef0",
            textSecondary: "#b0b4ba",
            textMuted: "#777b84",
            accent: "#0090ff",
            accentSubtle: "#0d2847",
            success: "#30a46c",
            warning: "#ffc53d",
            danger: "#e5484d",
            canvasDot: "#2a2d31"
        };
    },

    setTheme: function(theme, persist = true) {
        const t = theme === "light" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", t);
        if (persist) {
            try {
                localStorage.setItem("signalml_theme", t);
            } catch (err) { console.error("SML_DRAW_ERR:", err && err.stack || err); }
        }
        this.updateThemeToggleButton(t);
        this.chartTheme(t);
        this.refreshCharts(t);
        this.applyGraphTheme(null, t);
        if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
            window.dispatchEvent(new CustomEvent("signalml:themechange", { detail: { theme: t } }));
        }
        return t;
    },

    toggleTheme: function() {
        return this.setTheme(this.getTheme() === "dark" ? "light" : "dark", true);
    },

    updateThemeToggleButton: function(theme) {
        const btn = document.getElementById("theme-toggle-btn");
        if (!btn) return;
        const t = theme || this.getTheme();
        const nextLabel = t === "dark" ? "Switch to light theme" : "Switch to dark theme";
        btn.setAttribute("aria-label", nextLabel);
        btn.setAttribute("title", nextLabel);
        btn.innerHTML = this.icon(t === "dark" ? "sun" : "moon");
    },

    chartTheme: function(theme) {
        const p = this.getThemePalette(theme);
        const seriesColors = [
            "#0090ff", "#30a46c", "#ffc53d", "#e5484d",
            "#8e4ec6", "#00b8d9", "#f76b15", "#a0a7b4"
        ];
        const axisDefaults = {
            gridLineColor: p.borderSubtle,
            gridLineWidth: 1,
            lineColor: p.borderDefault,
            tickColor: p.borderDefault,
            labels: {
                style: {
                    color: p.textMuted,
                    fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                    fontSize: "11px"
                }
            },
            title: {
                style: {
                    color: p.textSecondary,
                    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                    fontSize: "12px",
                    fontWeight: "500"
                }
            }
        };
        const opts = {
            colors: seriesColors,
            chart: {
                backgroundColor: "transparent",
                plotBackgroundColor: "transparent",
                plotBorderWidth: 0,
                style: {
                    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                    fontSize: "12px",
                    color: p.textPrimary
                }
            },
            title: {
                style: {
                    color: p.textPrimary,
                    fontSize: "13px",
                    fontWeight: "600"
                }
            },
            subtitle: {
                style: {
                    color: p.textSecondary,
                    fontSize: "12px"
                }
            },
            xAxis: axisDefaults,
            yAxis: axisDefaults,
            legend: {
                backgroundColor: "transparent",
                itemStyle: {
                    color: p.textSecondary,
                    fontSize: "12px",
                    fontWeight: "500"
                },
                itemHoverStyle: {
                    color: p.textPrimary
                },
                itemHiddenStyle: {
                    color: p.textMuted
                }
            },
            tooltip: {
                backgroundColor: p.surface3,
                borderColor: p.borderStrong,
                borderRadius: 6,
                borderWidth: 1,
                shadow: false,
                style: {
                    color: p.textPrimary,
                    fontSize: "12px",
                    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                }
            },
            plotOptions: {
                series: {
                    shadow: false
                }
            },
            credits: {
                enabled: false
            },
            accessibility: {
                enabled: false
            }
        };
        if (typeof Highcharts !== "undefined" && typeof Highcharts.setOptions === "function") {
            Highcharts.setOptions(opts);
        }
        return opts;
    },

    refreshCharts: function(theme) {
        const opts = this.chartTheme(theme);
        if (typeof Highcharts === "undefined" || !Array.isArray(Highcharts.charts)) return;
        Highcharts.charts.forEach(chart => {
            if (!chart || typeof chart.update !== "function") return;
            try {
                const xCount = Array.isArray(chart.xAxis) ? chart.xAxis.length : 1;
                const yCount = Array.isArray(chart.yAxis) ? chart.yAxis.length : 1;
                chart.update({
                    colors: opts.colors,
                    chart: {
                        backgroundColor: "transparent",
                        plotBackgroundColor: "transparent"
                    },
                    title: opts.title,
                    subtitle: opts.subtitle,
                    xAxis: Array.from({ length: Math.max(1, xCount) }, () => ({
                        gridLineColor: opts.xAxis.gridLineColor,
                        lineColor: opts.xAxis.lineColor,
                        tickColor: opts.xAxis.tickColor,
                        labels: opts.xAxis.labels,
                        title: opts.xAxis.title
                    })),
                    yAxis: Array.from({ length: Math.max(1, yCount) }, () => ({
                        gridLineColor: opts.yAxis.gridLineColor,
                        lineColor: opts.yAxis.lineColor,
                        tickColor: opts.yAxis.tickColor,
                        labels: opts.yAxis.labels,
                        title: opts.yAxis.title
                    })),
                    legend: opts.legend,
                    tooltip: opts.tooltip
                }, true, false, false);
            } catch (err) { console.error("SML_DRAW_ERR:", err && err.stack || err); }
        });
    },

    _buildDotPatternCanvas: function(palette) {
        if (typeof document === "undefined") return null;
        const tile = document.createElement("canvas");
        const step = 20;
        tile.width = step;
        tile.height = step;
        const tctx = tile.getContext("2d");
        if (tctx) {
            tctx.fillStyle = palette.surface1;
            tctx.fillRect(0, 0, step, step);
            tctx.fillStyle = palette.canvasDot;
            tctx.fillRect(9, 9, 1.5, 1.5);
        }
        return tile;
    },

    applyGraphTheme: function(graphCanvas, theme) {
        const p = this.getThemePalette(theme);

        if (typeof LiteGraph !== "undefined") {
            LiteGraph.NODE_DEFAULT_BGCOLOR = p.surface2;
            LiteGraph.NODE_DEFAULT_COLOR = p.surface3;
            LiteGraph.NODE_DEFAULT_BOXCOLOR = p.borderStrong;
            LiteGraph.NODE_BOX_OUTLINE_COLOR = p.accent;
            LiteGraph.NODE_TITLE_COLOR = p.textPrimary;
            LiteGraph.NODE_SELECTED_TITLE_COLOR = p.textPrimary;
            LiteGraph.NODE_TEXT_COLOR = p.textSecondary;
            LiteGraph.WIDGET_BGCOLOR = p.surface1;
            LiteGraph.WIDGET_OUTLINE_COLOR = p.borderDefault;
            LiteGraph.WIDGET_TEXT_COLOR = p.textPrimary;
            LiteGraph.WIDGET_SECONDARY_TEXT_COLOR = p.textMuted;
            LiteGraph.LINK_COLOR = p.borderStrong;
            LiteGraph.EVENT_LINK_COLOR = p.accent;
            LiteGraph.CONNECTING_LINK_COLOR = p.accent;
            LiteGraph.DEFAULT_SHADOW_COLOR = "transparent";
        }

        if (typeof LGraphCanvas !== "undefined" && LGraphCanvas.link_type_colors) {
            LGraphCanvas.link_type_colors["-1"] = p.accent;
            LGraphCanvas.link_type_colors.number = p.borderStrong;
            LGraphCanvas.link_type_colors.node = p.borderStrong;
        }

        if (graphCanvas) {
            this._graphCanvases.add(graphCanvas);
        }

        const targets = graphCanvas ? [graphCanvas] : Array.from(this._graphCanvases);
        const dotTile = this._buildDotPatternCanvas(p);

        targets.forEach(canvas => {
            if (!canvas) return;
            canvas._smlPalette = p;
            canvas._smlDotTile = dotTile;
            canvas._smlDotPattern = null;

            canvas.background_image = null;
            canvas._pattern = null;
            canvas._bg_img = null;
            canvas.clear_background = true;
            canvas.clear_background_color = p.surface1;
            canvas.background_color = p.surface1;
            canvas.grid_color = p.canvasDot;

            canvas.render_shadows = false;
            canvas.render_connections_shadows = false;
            canvas.render_connections_border = false;
            canvas.render_canvas_border = false;
            canvas.use_gradients = false;
            canvas.round_radius = 6;

            canvas.node_title_color = p.textPrimary;
            canvas.default_link_color = p.borderStrong;
            canvas.default_connection_color = {
                input_off: p.borderStrong,
                input_on: p.accent,
                output_off: p.borderStrong,
                output_on: p.accent
            };
            canvas.title_text_font = '600 12px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
            canvas.inner_text_font = '500 11px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';

            if (canvas.canvas && canvas.canvas.style) {
                canvas.canvas.style.backgroundColor = p.surface1;
            }

            if (!canvas._smlBackgroundHooked) {
                const userOnDrawBg = canvas.onDrawBackground;
                canvas.onDrawBackground = function(ctx, visibleArea) {
                    const pal = this._smlPalette || p;
                    const area = visibleArea || [0, 0, this.canvas.width, this.canvas.height];
                    ctx.save();
                    if (!this._smlDotPattern && this._smlDotTile) {
                        try {
                            this._smlDotPattern = ctx.createPattern(this._smlDotTile, "repeat");
                        } catch (err) { console.error("SML_DRAW_ERR:", err && err.stack || err); }
                    }
                    ctx.fillStyle = this._smlDotPattern || pal.surface1;
                    ctx.fillRect(area[0], area[1], area[2], area[3]);
                    ctx.restore();
                    if (typeof userOnDrawBg === "function") {
                        userOnDrawBg.call(this, ctx, visibleArea);
                    }
                };
                canvas._smlBackgroundHooked = true;
            }

            if (!canvas._smlNodeShapeHooked && typeof canvas.drawNodeShape === "function") {
                const origDrawNodeShape = canvas.drawNodeShape;
                canvas.drawNodeShape = function(node, ctx, size, fgcolor, bgcolor, selected, mouseOver) {
                    const pal = this._smlPalette || p;
                    origDrawNodeShape.call(this, node, ctx, size, pal.surface3, pal.surface2, selected, mouseOver);
                    const titleH = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_TITLE_HEIGHT) ? LiteGraph.NODE_TITLE_HEIGHT : 30;
                    const collapsed = Boolean(node && node.flags && node.flags.collapsed);
                    const x = 0;
                    const y = -titleH;
                    const w = size[0] + 1;
                    const h = collapsed ? titleH : size[1] + titleH;
                    const r = this.round_radius || 6;

                    ctx.save();
                    if (!collapsed) {
                        ctx.beginPath();
                        ctx.moveTo(x + 1, 0.5);
                        ctx.lineTo(x + w - 1, 0.5);
                        ctx.strokeStyle = pal.borderSubtle;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                    ctx.beginPath();
                    if (typeof ctx.roundRect === "function") {
                        ctx.roundRect(x, y, w, h, r);
                    } else {
                        ctx.rect(x, y, w, h);
                    }
                    ctx.strokeStyle = selected ? pal.accent : pal.borderDefault;
                    ctx.lineWidth = selected ? 1.5 : 1;
                    ctx.stroke();
                    ctx.restore();
                };
                canvas._smlNodeShapeHooked = true;
            }

            if (!canvas._smlResizeHooked && typeof canvas.resize === "function") {
                canvas.resize = function(w, h) {
                    const parent = this.canvas ? this.canvas.parentElement : null;
                    if ((!w || !h) && parent) {
                        const rect = parent.getBoundingClientRect();
                        w = Math.round(rect.width) || parent.clientWidth || parent.offsetWidth;
                        h = Math.round(rect.height) || parent.clientHeight || parent.offsetHeight;
                    }
                    if (w && h && this.canvas) {
                        if (this.canvas.width !== w) this.canvas.width = w;
                        if (this.canvas.height !== h) this.canvas.height = h;
                        if (this.bgcanvas) {
                            this.bgcanvas.width = this.canvas.width;
                            this.bgcanvas.height = this.canvas.height;
                        }
                    }
                    this.setDirty(true, true);
                    this.draw(true, true);
                };
                if (typeof ResizeObserver !== "undefined" && canvas.canvas && canvas.canvas.parentElement) {
                    const ro = new ResizeObserver((entries) => {
                        for (const entry of entries) {
                            const cr = entry.contentRect;
                            const nw = Math.round(cr.width);
                            const nh = Math.round(cr.height);
                            if (nw > 0 && nh > 0 && (canvas.canvas.width !== nw || canvas.canvas.height !== nh)) {
                                canvas.resize(nw, nh);
                            }
                        }
                    });
                    ro.observe(canvas.canvas.parentElement);
                    canvas._smlResizeObserver = ro;
                }
                canvas._smlResizeHooked = true;
            }

            if (canvas.graph && Array.isArray(canvas.graph._nodes)) {
                canvas.graph._nodes.forEach(n => {
                    if (!n) return;
                    n.color = p.surface3;
                    n.bgcolor = p.surface2;
                    n.boxcolor = p.borderStrong;
                });
            }

            if (typeof canvas.setDirty === "function") {
                canvas.setDirty(true, true);
            }
            if (typeof canvas.draw === "function") {
                try {
                    canvas.draw(true, true);
                } catch (_) {}
            }
        });
    },

    getProjectId: function() {
        return this.currentProjectId || localStorage.getItem("signalml_active_project") || "default-project";
    },

    init: async function() {
        this.updateThemeToggleButton(this.getTheme());
        this.chartTheme();
        this.applyGraphTheme(null);
        this.setupHeaderNav();
        await this.loadProjects();
        this.setupProjectSwitcher();
        this.checkSystemHealth();
    },

    setupHeaderNav: function() {
        const groups = Array.from(document.querySelectorAll(".nav-group"));
        if (!groups.length) return;

        const closeAll = (except = null) => {
            groups.forEach(g => {
                if (g !== except) {
                    g.classList.remove("is-open");
                    const trig = g.querySelector(".nav-trigger");
                    if (trig) trig.setAttribute("aria-expanded", "false");
                }
            });
        };

        groups.forEach(group => {
            const trigger = group.querySelector(".nav-trigger");
            const menu = group.querySelector(".nav-menu");
            if (!trigger || !menu) return;
            const items = Array.from(menu.querySelectorAll(".nav-item"));

            const openGroup = (focusIndex = null) => {
                closeAll(group);
                group.classList.add("is-open");
                trigger.setAttribute("aria-expanded", "true");
                if (focusIndex !== null && items[focusIndex]) {
                    items[focusIndex].focus();
                }
            };

            const closeGroup = (returnFocus = false) => {
                group.classList.remove("is-open");
                trigger.setAttribute("aria-expanded", "false");
                if (returnFocus) trigger.focus();
            };

            trigger.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (group.classList.contains("is-open")) {
                    closeGroup(false);
                } else {
                    openGroup(null);
                }
            });

            trigger.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " " || e.key === "ArrowDown") {
                    e.preventDefault();
                    openGroup(0);
                } else if (e.key === "ArrowUp") {
                    e.preventDefault();
                    openGroup(items.length - 1);
                } else if (e.key === "Escape") {
                    closeGroup(true);
                }
            });

            menu.addEventListener("keydown", (e) => {
                const idx = items.indexOf(document.activeElement);
                if (e.key === "Escape") {
                    e.preventDefault();
                    closeGroup(true);
                } else if (e.key === "ArrowDown") {
                    e.preventDefault();
                    const next = idx < items.length - 1 ? idx + 1 : 0;
                    if (items[next]) items[next].focus();
                } else if (e.key === "ArrowUp") {
                    e.preventDefault();
                    const prev = idx > 0 ? idx - 1 : items.length - 1;
                    if (items[prev]) items[prev].focus();
                } else if (e.key === "Tab") {
                    closeGroup(false);
                }
            });
        });

        document.addEventListener("click", (e) => {
            if (!e.target.closest(".nav-group")) {
                closeAll();
            }
        });

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                closeAll();
            }
        });
    },

    toast: function(message, type = "info", duration = 4000) {
        let container = document.getElementById("toast-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "toast-container";
            document.body.appendChild(container);
        }

        const normType = type === "danger" ? "error" : (type || "info");
        const iconMap = {
            success: "circle-check",
            error: "circle-x",
            warning: "triangle-exclamation-point",
            info: "info"
        };
        const iconName = iconMap[normType] || "info";

        const toast = document.createElement("div");
        toast.className = `toast ${normType}`;
        toast.setAttribute("role", "status");
        toast.innerHTML = `
            ${this.icon(iconName, "toast-icon")}
            <span class="toast-message">${message}</span>
            <button type="button" class="toast-close" aria-label="Dismiss notification">${this.icon("x", "icon-xs")}</button>
        `;

        const closeBtn = toast.querySelector(".toast-close");
        const dismiss = () => {
            toast.classList.add("is-leaving");
            toast.style.opacity = "0";
            setTimeout(() => toast.remove(), 180);
        };
        if (closeBtn) {
            closeBtn.addEventListener("click", dismiss);
        }

        container.appendChild(toast);
        setTimeout(dismiss, duration);
    },

    safeJson: async function(res) {
        let text = "";
        try {
            text = await res.text();
            return JSON.parse(text);
        } catch (_) {
            return { detail: text || res.statusText || `HTTP ${res.status}` };
        }
    },

    escapeHtml: function(str) {
        if (str === null || str === undefined) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    },

    formatNumber: function(num, decimals = 0) {
        if (num === null || num === undefined || Number.isNaN(Number(num))) return "—";
        return Number(num).toLocaleString("en-US", {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    },

    api: async function(url, options = {}) {
        const res = await fetch(url, options);
        const data = await this.safeJson(res);
        if (!res.ok) {
            throw new Error(data.detail || `HTTP ${res.status}`);
        }
        return data;
    },

    loadProjects: async function() {
        try {
            const res = await fetch("/api/projects");
            if (res.ok) {
                this.projects = await this.safeJson(res);
                if (Array.isArray(this.projects) && this.projects.length > 0) {
                    const exists = this.projects.find(p => p.id === this.currentProjectId);
                    if (!exists) {
                        this.currentProjectId = this.projects[0].id;
                        localStorage.setItem("signalml_active_project", this.currentProjectId);
                    }
                }
                this.updateProjectHeader();
            }
        } catch (e) {
            console.error("Failed to load projects", e);
        }
    },

    updateProjectHeader: function() {
        const el = document.getElementById("active-project-name");
        if (el) {
            const current = this.projects.find(p => p.id === this.currentProjectId);
            el.textContent = current ? current.name : "Default Project";
        }
    },

    renderProjectModalList: function() {
        const list = document.getElementById("project-modal-list");
        if (!list) return;
        list.innerHTML = "";
        this.projects.forEach(p => {
            const isCur = p.id === this.currentProjectId;
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = `list-item${isCur ? " active" : ""}`;
            if (isCur) btn.setAttribute("aria-current", "true");
            btn.innerHTML = `
                ${this.icon(isCur ? "check" : "folder")}
                <span class="fw-500 truncate">${this.escapeHtml(p.name)}</span>
                <span class="list-item-meta">${this.escapeHtml(String(p.id).slice(0, 8))}</span>
            `;
            btn.addEventListener("click", () => {
                this.switchProject(p.id);
                const modal = document.getElementById("project-modal");
                if (modal) modal.style.display = "none";
            });
            list.appendChild(btn);
        });
    },

    setupProjectSwitcher: function() {
        const btn = document.getElementById("project-switcher-btn");
        if (btn && !btn._smlBound) {
            btn._smlBound = true;
            btn.addEventListener("click", () => {
                this.renderProjectModalList();
                const modal = document.getElementById("project-modal");
                if (modal) modal.style.display = "flex";
            });
        }
    },

    switchProject: function(projectId) {
        this.currentProjectId = projectId;
        localStorage.setItem("signalml_active_project", projectId);
        this.updateProjectHeader();
        this.toast("Switched active project", "success");
        setTimeout(() => window.location.reload(), 400);
    },

    createProject: async function(name, description = "") {
        try {
            const res = await fetch("/api/projects", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, description })
            });
            if (res.ok) {
                const proj = await this.safeJson(res);
                this.toast(`Project '${name}' created`, "success");
                this.switchProject(proj.id);
            } else {
                const err = await this.safeJson(res);
                this.toast(`Error: ${err.detail || "Could not create project"}`, "error");
            }
        } catch (e) {
            this.toast(`Error: ${e.message}`, "error");
        }
    },

    updateProject: async function(id, name, description = "") {
        try {
            const res = await fetch(`/api/projects/${id}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, description })
            });
            if (res.ok) {
                this.toast("Project updated", "success");
                await this.loadProjects();
                return true;
            } else {
                const err = await this.safeJson(res);
                this.toast(`Error: ${err.detail || "Could not update project"}`, "error");
                return false;
            }
        } catch (e) {
            this.toast(`Error: ${e.message}`, "error");
            return false;
        }
    },

    deleteProject: async function(id) {
        try {
            const res = await fetch(`/api/projects/${id}`, {
                method: "DELETE"
            });
            if (res.ok) {
                this.toast("Project deleted", "success");
                await this.loadProjects();
                if (this.currentProjectId === id) {
                    if (this.projects.length > 0) {
                        this.switchProject(this.projects[0].id);
                    } else {
                        window.location.reload();
                    }
                }
                return true;
            } else {
                const err = await this.safeJson(res);
                this.toast(`Error: ${err.detail || "Could not delete project"}`, "error");
                return false;
            }
        } catch (e) {
            this.toast(`Error: ${e.message}`, "error");
            return false;
        }
    },

    duplicateProject: async function(id) {
        try {
            const res = await fetch(`/api/projects/${id}/duplicate`, {
                method: "POST"
            });
            if (res.ok) {
                const proj = await this.safeJson(res);
                this.toast(`Project duplicated as '${proj.name}'`, "success");
                await this.loadProjects();
                return proj;
            } else {
                const err = await this.safeJson(res);
                this.toast(`Error: ${err.detail || "Could not duplicate project"}`, "error");
                return null;
            }
        } catch (e) {
            this.toast(`Error: ${e.message}`, "error");
            return null;
        }
    },

    checkSystemHealth: async function() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await this.safeJson(res);
                const gpuEl = document.getElementById("health-gpu-pill");
                if (gpuEl) {
                    if (data.gpu_available) {
                        gpuEl.classList.add("is-gpu");
                        gpuEl.innerHTML = `${this.icon("cpu")}<span>GPU Active</span>`;
                    } else {
                        gpuEl.classList.remove("is-gpu");
                        gpuEl.innerHTML = `${this.icon("cpu")}<span>CPU Mode</span>`;
                    }
                }
            }
        } catch (e) {
            console.warn("Health check unreachable", e);
        }
    }
};

if (typeof window !== "undefined") {
    window.SignalML = SignalML;
}

document.addEventListener("DOMContentLoaded", () => {
    SignalML.init();
});
