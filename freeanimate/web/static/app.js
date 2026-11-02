"use strict";

async function getJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = j.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function el(sel, root) { return (root || document).querySelector(sel); }
function els(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------------------------------------------------------- create */
function initNewForm() {
  const form = el("#create-form");
  if (!form) return;
  const prompt = el("#prompt");
  const preview = el("#plan-preview");
  const status = el("#form-status");
  const submit = el("#create-submit");
  let timer = null;
  let seq = 0;

  function durationSeconds() {
    const sel = el("#duration");
    const custom = el("#duration-custom");
    if (sel.value === "custom") {
      return parseFloat(custom.value) || null;
    }
    return parseFloat(sel.value) || null;
  }

  function syncCustomDuration() {
    const sel = el("#duration");
    const custom = el("#duration-custom");
    if (!custom) return;
    custom.style.display = sel.value === "custom" ? "block" : "none";
  }

  async function refreshPlan() {
    const text = prompt.value.trim();
    if (text.length < 8) { preview.innerHTML = '<p class="muted">Describe your video to see a plan preview.</p>'; return; }
    const my = ++seq;
    try {
      const data = await getJSON("/api/plan", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text, method: el("#method").value,
          duration_seconds: durationSeconds() }),
      });
      if (my !== seq) return;
      const chips = Object.entries(data.capabilities || {})
        .filter(([, v]) => v === "available").map(([k]) => `<span class="badge ok">${esc(k.replace(/_/g, " "))}</span>`).join(" ");
      const chars = (data.characters || []).map((c) => `<span class="badge">${esc(c.name)} · ${esc(c.species)}</span>`).join(" ");
      const dlg = (data.dialogue || []).slice(0, 4).map((d) => `<div class="dialogue-line"><b>${esc(d.speaker)}</b> ${esc(d.text)}</div>`).join("");
      preview.innerHTML = `
        <h3>${esc(data.title)}</h3>
        <p class="small muted">${esc(data.project_type)} · ${esc(data.environment)} · ${Math.round(data.duration_seconds)}s · ${esc(data.aspect_ratio)}</p>
        <p class="small"><b>Requested:</b> ${esc(data.requested_method)}<br><b>Will run:</b> ${esc(data.supported_method)}</p>
        <p class="small muted">${esc(data.method_note)}</p>
        <div class="pill-list" style="margin:.5rem 0">${chars}</div>
        <div class="pill-list" style="margin:.5rem 0">${chips}</div>
        ${dlg || '<p class="small muted">Dialogue will be generated (marked as generated and editable).</p>'}
        ${(data.warnings || []).map((w) => `<div class="alert warn small">${esc(w)}</div>`).join("")}`;
    } catch (err) {
      preview.innerHTML = `<div class="alert bad small">Plan preview failed: ${esc(err.message)}</div>`;
    }
  }

  prompt.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(refreshPlan, 550);
  });
  el("#method").addEventListener("change", refreshPlan);
  el("#duration").addEventListener("change", () => {
    syncCustomDuration();
    if (el("#duration").value !== "custom") refreshPlan();
  });
  if (el("#duration-custom")) {
    el("#duration-custom").addEventListener("change", refreshPlan);
  }
  syncCustomDuration();

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    submit.disabled = true;
    status.innerHTML = '<div class="row"><span class="spinner"></span><span>Creating project and starting the render job…</span></div>';
    const payload = {
      prompt: prompt.value.trim(),
      title: el("#title").value.trim() || null,
      method: el("#method").value,
      resolution: el("#resolution").value,
      fps: parseInt(el("#fps").value, 10),
      language: el("#language").value,
      music: el("#music").value,
      burn_subtitles: el("#burn").checked,
      aspect_ratio: el("#aspect").value,
      duration_seconds: durationSeconds(),
      dialogue_text: el("#dialogue-text").value.trim() || null,
    };
    try {
      const data = await getJSON("/api/projects", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      window.location.href = `/jobs/${data.job_id}`;
    } catch (err) {
      submit.disabled = false;
      status.innerHTML = `<div class="alert bad">Could not start: ${esc(err.message)}</div>`;
    }
  });
}

/* ------------------------------------------------------------------ job */
function initJob() {
  const root = el("#job");
  if (!root) return;
  const id = root.dataset.jobId;
  const bar = el("#bar"), pct = el("#pct"), stage = el("#stage"), msg = el("#message");
  const badge = el("#job-status"), elapsed = el("#elapsed"), result = el("#job-result");
  const cancel = el("#cancel");
  let startedAt = Date.now();

  async function tick() {
    try {
      const job = await getJSON(`/api/jobs/${id}`);
      badge.textContent = job.status;
      badge.className = "badge " + (job.status === "done" ? "ok" : job.status === "failed" ? "bad" : job.status === "cancelled" ? "warn" : "");
      const p = Math.round((job.progress || 0) * 100);
      bar.style.width = p + "%";
      pct.textContent = p + "%";
      stage.textContent = job.stage || job.status;
      msg.textContent = job.message || "";
      if (!startedAt && job.started_at) startedAt = job.started_at * 1000;
      elapsed.textContent = Math.max(0, Math.round((Date.now() - startedAt) / 1000)) + "s";
      if (job.status === "queued") { cancel.disabled = false; }
      if (job.status === "running" || job.status === "queued") {
        setTimeout(tick, 1000);
      } else {
        cancel.disabled = true;
        renderResult(job);
      }
    } catch (err) {
      msg.textContent = "Status check failed: " + err.message;
      setTimeout(tick, 2500);
    }
  }

  function renderResult(job) {
    if (job.status === "done" && job.result) {
      const r = job.result;
      const link = r.project_slug ? `/projects/${r.project_slug}` : null;
      result.innerHTML = `
        <div class="alert ok"><b>Video ready.</b> ${esc(r.method || "")} · ${Math.round(r.duration || 0)}s · ${(r.scene_count || 0)} scene(s)</div>
        ${link ? `<p><a class="btn primary" href="${link}">Open project</a></p>` : ""}
        ${(r.limitations || []).map((x) => `<div class="alert warn small">${esc(x)}</div>`).join("")}`;
      if (link) setTimeout(() => { window.location.href = link; }, 1600);
    } else if (job.status === "failed") {
      result.innerHTML = `<div class="alert bad"><b>Render failed.</b><pre class="mono" style="white-space:pre-wrap">${esc(job.error || job.message || "unknown error")}</pre></div>`;
    } else if (job.status === "cancelled") {
      result.innerHTML = '<div class="alert warn">Cancelled. Any completed project files remain on disk.</div>';
    }
  }

  cancel.addEventListener("click", async () => {
    cancel.disabled = true;
    try { await getJSON(`/api/jobs/${id}/cancel`, { method: "POST" }); }
    catch (err) { msg.textContent = err.message; }
  });
  tick();
}

/* -------------------------------------------------------------- project */
function initRerender() {
  const form = el("#rerender-form");
  if (!form) return;
  const status = el("#rerender-status");
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = el("#rerender-submit");
    btn.disabled = true;
    status.innerHTML = '<div class="row"><span class="spinner"></span><span>Starting a new version…</span></div>';
    const payload = {
      dialogue_text: el("#dialogue-edit").value,
      resolution: el("#rr-resolution") ? el("#rr-resolution").value : "720p",
      music: el("#rr-music") ? el("#rr-music").value : "none",
    };
    try {
      const data = await getJSON(`/api/projects/${form.dataset.slug}/rerender`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      window.location.href = `/jobs/${data.job_id}`;
    } catch (err) {
      btn.disabled = false;
      status.innerHTML = `<div class="alert bad">${esc(err.message)}</div>`;
    }
  });
}

/* ---------------------------------------------------------- scene rerender */
function initSceneRerender() {
  const root = el("#scene-list");
  if (!root) return;
  const status = el("#scene-status");
  root.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("[data-scene]");
    if (!btn) return;
    btn.disabled = true;
    if (status) status.innerHTML = `<div class="row"><span class="spinner"></span><span>Re-rendering scene ${esc(btn.dataset.scene)}…</span></div>`;
    try {
      const data = await getJSON(`/api/projects/${root.dataset.slug}/scenes/${btn.dataset.scene}/rerender`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
      });
      window.location.href = `/jobs/${data.job_id}`;
    } catch (err) {
      btn.disabled = false;
      if (status) status.innerHTML = `<div class="alert bad">${esc(err.message)}</div>`;
    }
  });
}

/* ------------------------------------------------------------- settings */
function initSettings() {
  const form = el("#settings-form");
  if (!form) return;
  const status = el("#settings-status");
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = el("#settings-submit");
    btn.disabled = true;
    const updates = {};
    els("[data-setting]", form).forEach((node) => {
      updates[node.dataset.setting] = node.type === "checkbox" ? node.checked : node.value;
    });
    try {
      const data = await getJSON("/api/settings", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates }),
      });
      const source = el("#config-source");
      if (source && data.source_path) source.textContent = data.source_path;
      status.innerHTML = '<div class="alert ok">Settings validated and saved. Existing projects are unchanged.</div>';
    } catch (err) {
      status.innerHTML = `<div class="alert bad">${esc(err.message)}</div>`;
    } finally {
      btn.disabled = false;
    }
  });
}

/* --------------------------------------------------------------- models */
function initModels() {
  els("[data-plan-model]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const out = el("#plan-out");
      try {
        const plan = await getJSON("/api/models/plan", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: btn.dataset.planModel }),
        });
        out.innerHTML = `<div class="alert"><b>Download plan — nothing downloaded yet.</b>
          <div class="mono small">repo: ${esc(plan.repo_id)}<br>approx: ${esc(plan.approx_gb)} GB<br>target: ${esc(plan.target_dir)}</div>
          <p class="small muted">${esc(plan.license_note)}</p>
          <p class="small">${esc(plan.note)}</p></div>`;
      } catch (err) { out.innerHTML = `<div class="alert bad">${esc(err.message)}</div>`; }
    });
  });
}


/* ------------------------------------------------------------- image lab */
function initImageLab() {
  const form = el("#image-form");
  if (!form) return;
  const status = el("#image-status");
  const result = el("#image-result");
  const submit = el("#image-submit");
  const useBtn = el("#image-use");
  const useStatus = el("#image-use-status");
  let current = null;

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    submit.disabled = true;
    useBtn.disabled = true;
    current = null;
    status.innerHTML = '<div class="row"><span class="spinner"></span><span>Loading the local model and generating. This can take minutes on CPU...</span></div>';
    result.innerHTML = '<p class="muted">Generating...</p>';
    const payload = {
      prompt: el("#image-prompt").value.trim(),
      model: el("#image-model").value,
      negative_prompt: el("#image-negative").value.trim(),
      width: parseInt(el("#image-width").value, 10) || 512,
      height: parseInt(el("#image-height").value, 10) || 512,
      steps: el("#image-steps").value ? parseInt(el("#image-steps").value, 10) : null,
      seed: el("#image-seed").value ? parseInt(el("#image-seed").value, 10) : null,
    };
    try {
      const data = await getJSON("/api/image/generate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      current = data;
      const d = data.detail || {};
      result.innerHTML = `
        <a href="${esc(data.url)}" target="_blank" rel="noopener"><img src="${esc(data.url)}" alt="Generated image" style="width:100%;height:auto;border-radius:.6rem"></a>
        <p class="small muted">${esc(data.model || "")} &middot; ${esc(d.device || "")} &middot; ${esc(d.steps || "")} steps &middot; ${esc(d.seconds || "")}s</p>
        <p class="small"><a href="${esc(data.url)}" download>Download PNG</a></p>`;
      status.innerHTML = '<div class="alert ok">Image generated locally and saved.</div>';
      useBtn.disabled = !el("#image-project").value;
    } catch (err) {
      status.innerHTML = `<div class="alert bad">${esc(err.message)}</div>`;
      result.innerHTML = '<p class="muted">No image generated.</p>';
    } finally {
      submit.disabled = false;
    }
  });

  el("#image-project").addEventListener("change", () => {
    useBtn.disabled = !current || !el("#image-project").value;
  });

  useBtn.addEventListener("click", async () => {
    if (!current) return;
    useBtn.disabled = true;
    useStatus.innerHTML = '<div class="row"><span class="spinner"></span><span>Saving to project...</span></div>';
    try {
      const data = await getJSON("/api/image/use", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: current.image, project: el("#image-project").value }),
      });
      useStatus.innerHTML = `<div class="alert ok">${esc(data.note)}<br><span class="mono small">${esc(data.path)}</span></div>`;
    } catch (err) {
      useStatus.innerHTML = `<div class="alert bad">${esc(err.message)}</div>`;
    } finally {
      useBtn.disabled = false;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initNewForm(); initJob(); initRerender(); initModels(); initSettings(); initSceneRerender(); initImageLab();
});
