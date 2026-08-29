(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // -------------------------------------------------------------------
  // Tabs
  // -------------------------------------------------------------------
  $$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach((b) => b.classList.remove("active"));
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`#tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "history") loadHistory(true);
    });
  });

  // -------------------------------------------------------------------
  // Health check
  // -------------------------------------------------------------------
  async function checkHealth() {
    const dot = $("#health-dot");
    const text = $("#health-text");
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data.model_loaded) {
        dot.className = "health-dot ok";
        text.textContent = "backend online";
      } else {
        dot.className = "health-dot bad";
        text.textContent = "models not loaded";
      }
    } catch (e) {
      dot.className = "health-dot bad";
      text.textContent = "backend unreachable";
    }
  }
  checkHealth();
  setInterval(checkHealth, 30000);

  // -------------------------------------------------------------------
  // File selection
  // -------------------------------------------------------------------
  let selectedFiles = [];
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const selectedFilesEl = $("#selected-files");
  const analyzeBtn = $("#analyze-btn");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    addFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
    fileInput.value = "";
  });

  function addFiles(fileList) {
    for (const f of fileList) {
      if (f.type.startsWith("image/")) selectedFiles.push(f);
    }
    renderSelectedFiles();
  }

  function renderSelectedFiles() {
    selectedFilesEl.innerHTML = "";
    selectedFilesEl.hidden = selectedFiles.length === 0;
    selectedFiles.forEach((file, idx) => {
      const wrap = document.createElement("div");
      wrap.className = "file-thumb";
      const url = URL.createObjectURL(file);
      wrap.innerHTML = `
        <img src="${url}" alt="${escapeHtml(file.name)}" />
        <div class="file-name">${escapeHtml(file.name)}</div>
        <button class="remove-file" data-idx="${idx}" title="Remove">&times;</button>
      `;
      selectedFilesEl.appendChild(wrap);
    });
    $$(".remove-file", selectedFilesEl).forEach((btn) => {
      btn.addEventListener("click", () => {
        selectedFiles.splice(Number(btn.dataset.idx), 1);
        renderSelectedFiles();
      });
    });
    analyzeBtn.disabled = selectedFiles.length === 0;
  }

  // -------------------------------------------------------------------
  // Analyze
  // -------------------------------------------------------------------
  const errorBanner = $("#error-banner");
  const loadingEl = $("#loading");
  const resultEmpty = $("#result-empty");
  const resultSingle = $("#result-single");
  const resultBatch = $("#result-batch");

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.hidden = false;
  }
  function clearError() { errorBanner.hidden = true; errorBanner.textContent = ""; }

  function setResultView(view) {
    resultEmpty.hidden = view !== "empty";
    loadingEl.hidden = view !== "loading";
    resultSingle.hidden = view !== "single";
    resultBatch.hidden = view !== "batch";
  }

  analyzeBtn.addEventListener("click", async () => {
    if (selectedFiles.length === 0) return;
    clearError();
    setResultView("loading");
    analyzeBtn.disabled = true;

    try {
      if (selectedFiles.length === 1) {
        const form = new FormData();
        form.append("file", selectedFiles[0]);
        const res = await fetch("/api/analyze", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Analysis failed.");
        resultSingle.innerHTML = "";
        resultSingle.appendChild(renderAnalysisDetail(data));
        setResultView("single");
      } else {
        const form = new FormData();
        selectedFiles.forEach((f) => form.append("files", f));
        const res = await fetch("/api/analyze/batch", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Batch analysis failed.");
        renderBatchResults(data);
        setResultView("batch");
      }
    } catch (e) {
      showError(e.message || "Something went wrong while analyzing the image(s).");
      setResultView("empty");
    } finally {
      analyzeBtn.disabled = selectedFiles.length === 0;
    }
  });

  function renderBatchResults(items) {
    resultBatch.innerHTML = `<div class="section-title">${items.length} image(s) analyzed</div><div class="batch-grid"></div>`;
    const grid = $(".batch-grid", resultBatch);
    items.forEach((item) => {
      const card = document.createElement("div");
      if (item.error) {
        card.className = "batch-card error";
        card.innerHTML = `<div class="body"><div class="filename">${escapeHtml(item.original_filename || "file")}</div><p>${escapeHtml(item.error)}</p></div>`;
      } else {
        card.className = "batch-card";
        card.innerHTML = `
          <img src="${item.image_url}" alt="" />
          <div class="body">
            <div class="filename">${escapeHtml(item.original_filename)}</div>
            <div class="score" style="color:${scoreColor(item.quality_label)}">${item.quality_score}</div>
            <span class="label-badge ${item.quality_label}">${item.quality_label}</span>
          </div>`;
        card.addEventListener("click", () => openDetailModal(item.id));
      }
      grid.appendChild(card);
    });
  }

  // -------------------------------------------------------------------
  // Shared result-rendering
  // -------------------------------------------------------------------
  function scoreColor(label) {
    return { ACCEPTABLE: "var(--success)", DEGRADED: "var(--warning)", DEFECTIVE: "var(--danger)" }[label] || "var(--text)";
  }

  const FEATURE_LABELS = {
    sharpness_lap_var: "Sharpness (Laplacian var)",
    sharpness_tenengrad: "Sharpness (Tenengrad)",
    brightness_mean: "Brightness (mean)",
    contrast_std: "Contrast (std dev)",
    underexposed_ratio: "Underexposed px ratio",
    overexposed_ratio: "Overexposed px ratio",
    noise_sigma: "Noise sigma",
    colorfulness: "Colorfulness",
    saturation_mean: "Saturation (mean)",
    entropy: "Entropy (bits)",
    edge_density: "Edge density",
    blockiness: "Blockiness",
  };

  function renderAnalysisDetail(data) {
    const root = document.createElement("div");

    const issuesHtml = data.issues.length
      ? `<div class="issues-list">${data.issues.map(issueCardHtml).join("")}</div>`
      : `<div class="no-issues">No quality issues detected.</div>`;

    const featuresHtml = Object.entries(data.features || {})
      .map(([key, val]) => `
        <div class="feature-tile">
          <div class="label">${escapeHtml(FEATURE_LABELS[key] || key)}</div>
          <div class="value">${formatNumber(val)}</div>
        </div>`).join("");

    const hasHeatmap = !!data.heatmap_url;

    root.innerHTML = `
      <div class="result-image-wrap">
        <img id="detail-img" src="${data.image_url}" alt="${escapeHtml(data.original_filename)}" />
      </div>
      ${hasHeatmap ? `
        <div class="heatmap-toggle">
          <label><input type="checkbox" id="heatmap-checkbox" /> Show anomaly reconstruction-error heatmap (localization)</label>
        </div>` : ""}

      <div class="score-row">
        <div class="score-ring" style="--pct:${Math.max(0, Math.min(100, data.quality_score))}; --ring-color:${scoreColor(data.quality_label)}">
          <span class="score-value">${Math.round(data.quality_score)}</span>
        </div>
        <div>
          <span class="label-badge ${data.quality_label}">${data.quality_label}</span>
          <div class="meta-line">${data.width}×${data.height}px · ${formatBytes(data.file_size_bytes)} · ${data.processing_time_ms.toFixed(0)} ms</div>
          <div class="meta-line">${escapeHtml(data.original_filename)}</div>
        </div>
      </div>

      <div class="section-title">Detected issues</div>
      ${issuesHtml}

      <div class="section-title">Image statistics (explainability)</div>
      <div class="features-grid">${featuresHtml}</div>
      ${data.anomaly_score != null ? `<p class="meta-line" style="margin-top:8px;">Anomaly (reconstruction-error) score: ${data.anomaly_score.toFixed(5)} — secondary diagnostic signal, threshold ${data.anomaly_threshold?.toFixed(5)}. See README for why this is advisory-only.</p>` : ""}
      <p class="meta-line">Model: ${escapeHtml(data.model_version)}</p>
    `;

    if (hasHeatmap) {
      const cb = $("#heatmap-checkbox", root);
      const img = $("#detail-img", root);
      cb.addEventListener("change", () => {
        img.src = cb.checked ? data.heatmap_url : data.image_url;
      });
    }
    return root;
  }

  function issueCardHtml(issue) {
    const pct = Math.round(issue.confidence * 100);
    return `
      <div class="issue-card">
        <div class="issue-head">
          <span class="issue-type">${escapeHtml(issue.type)}</span>
          <span class="severity-badge ${issue.severity}">${issue.severity}</span>
        </div>
        <p class="issue-explanation">${escapeHtml(issue.explanation)}</p>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${pct}%"></div></div>
        <p class="confidence-label">confidence ${pct}% · source: ${escapeHtml(issue.confidence_source)}</p>
      </div>`;
  }

  // -------------------------------------------------------------------
  // History
  // -------------------------------------------------------------------
  const historyGrid = $("#history-grid");
  const historyEmpty = $("#history-empty");
  const loadMoreBtn = $("#load-more");
  let historyOffset = 0;
  const HISTORY_PAGE = 12;

  async function loadHistory(reset) {
    if (reset) { historyOffset = 0; historyGrid.innerHTML = ""; }
    try {
      const res = await fetch(`/api/analyses?limit=${HISTORY_PAGE}&offset=${historyOffset}`);
      const data = await res.json();
      historyEmpty.hidden = data.total > 0;
      data.items.forEach((item) => historyGrid.appendChild(historyCardEl(item)));
      historyOffset += data.items.length;
      loadMoreBtn.hidden = historyOffset >= data.total;
    } catch (e) {
      historyEmpty.hidden = false;
      historyEmpty.querySelector("p").textContent = "Could not load history — is the backend running?";
    }
  }

  function historyCardEl(item) {
    const card = document.createElement("div");
    card.className = "history-card";
    card.innerHTML = `
      <img src="${item.image_url}" alt="${escapeHtml(item.original_filename)}" />
      <div class="body">
        <div class="filename">${escapeHtml(item.original_filename)}</div>
        <div class="row">
          <span class="score" style="color:${scoreColor(item.quality_label)}">${Math.round(item.quality_score)}</span>
          <span class="label-badge ${item.quality_label}">${item.quality_label}</span>
        </div>
        <div class="timestamp">${formatDate(item.created_at)} · ${item.issue_count} issue(s)</div>
      </div>`;
    card.addEventListener("click", () => openDetailModal(item.id));
    return card;
  }

  $("#refresh-history").addEventListener("click", () => loadHistory(true));
  loadMoreBtn.addEventListener("click", () => loadHistory(false));

  // -------------------------------------------------------------------
  // Detail modal
  // -------------------------------------------------------------------
  const modal = $("#detail-modal");
  const modalBody = $("#modal-body");

  async function openDetailModal(id) {
    modalBody.innerHTML = `<div class="loading"><div class="spinner"></div><p>Loading…</p></div>`;
    modal.hidden = false;
    try {
      const res = await fetch(`/api/analyses/${id}`);
      if (!res.ok) throw new Error("Analysis not found.");
      const data = await res.json();
      modalBody.innerHTML = "";
      modalBody.appendChild(renderAnalysisDetail(data));
    } catch (e) {
      modalBody.innerHTML = `<p>${escapeHtml(e.message)}</p>`;
    }
  }
  $$("[data-close]").forEach((el) => el.addEventListener("click", () => { modal.hidden = true; }));

  // -------------------------------------------------------------------
  // Utils
  // -------------------------------------------------------------------
  function escapeHtml(str) {
    return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  }
  function formatNumber(n) {
    if (typeof n !== "number") return n;
    return Math.abs(n) >= 100 ? n.toFixed(1) : n.toFixed(3);
  }
  function formatDate(iso) {
    try {
      return new Date(iso + (iso.endsWith("Z") ? "" : "Z")).toLocaleString();
    } catch (e) { return iso; }
  }

  setResultView("empty");
})();
