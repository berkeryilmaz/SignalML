// Global SignalML Client Application State & Utilities

const SignalML = {
    currentProjectId: localStorage.getItem("signalml_active_project") || "default-project",
    projects: [],
    activeDataset: null,
    activeModel: null,

    init: async function() {
        await this.loadProjects();
        this.setupProjectSwitcher();
        this.checkSystemHealth();
    },

    toast: function(message, type = "info", duration = 4000) {
        let container = document.getElementById("toast-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "toast-container";
            document.body.appendChild(container);
        }

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            setTimeout(() => toast.remove(), 300);
        }, duration);
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

    setupProjectSwitcher: function() {
        const btn = document.getElementById("project-switcher-btn");
        if (btn) {
            btn.addEventListener("click", () => {
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
                this.toast(`Project updated successfully`, "success");
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
                this.toast("Project deleted successfully", "success");
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
                        gpuEl.textContent = "GPU Active";
                        gpuEl.style.color = "var(--accent-green)";
                    } else {
                        gpuEl.textContent = "CPU Mode";
                        gpuEl.style.color = "var(--text-muted)";
                    }
                }
            }
        } catch (e) {
            console.warn("Health check unreachable", e);
        }
    }
};

document.addEventListener("DOMContentLoaded", () => {
    SignalML.init();
});
