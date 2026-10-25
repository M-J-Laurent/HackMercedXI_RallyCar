/* Rally Coach - talks to the local server (server/app.py) when there is one.

   With the server:  every lap you open is analysed live by the anomaly model, and an on-device LLM writes
                     the tips. The AI coach panel can download that LLM with one click.
   Without it (GitHub Pages, or index.html opened from disk): this file does nothing and the page keeps
                     using the data in data/bundle.js and its built-in tips. */
(() => {
  "use strict";
  const App = window.RallyApp;
  if (!App) return;
  const $ = (id) => document.getElementById(id);
  const state = App.state;

  let health = null;          // /api/health, null when there is no server
  let llm = null;             // last /api/llm/status
  let model = null;
  let tipRun = 0;             // bumped to cancel an in-flight tip stream
  let poll = null;
  let tipsProgress = "";
  let busy = "";

  try { model = localStorage.getItem("rally-model"); } catch (_) {}

  // ------------------------------------------------------------------ http
  async function http(path, { method = "GET", body, type = "application/json", timeout = 0, signal } = {}) {
    const ctl = new AbortController();
    const t = timeout ? setTimeout(() => ctl.abort(), timeout) : null;
    if (signal) signal.addEventListener("abort", () => ctl.abort());
    try {
      const res = await fetch(path, {
        method, signal: ctl.signal,
        headers: body !== undefined ? { "Content-Type": type } : {},
        body: body === undefined ? undefined : (type === "application/json" ? JSON.stringify(body) : body),
      });
      return res;
    } finally { if (t) clearTimeout(t); }
  }
  async function json(path, opts) {
    const res = await http(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
    return data;
  }

  // ------------------------------------------------------------------ analysis
  async function analyze(lap) {
    if (lap.analyzing || lap.analyzed || lap.is_reference) return;
    lap.analyzing = true;
    busy = "Analysing this lap with the model…";
    App.note(busy);
    try {
      const res = lap.raw
        ? await json("/api/analyze", { method: "POST", body: lap.raw, type: "text/csv" })
        : await json(`/api/analyze?lap=${encodeURIComponent(lap.file)}`, { method: "POST", body: "" , type: "text/plain" });
      lap.segments = res.segments;
      lap.segments_source = "model";
      lap.analyzed = true;
      busy = "";
      if (state.lap === lap) App.setLapIndex(state.laps.indexOf(lap));
      writeTips(lap);
    } catch (err) {
      busy = "";
      if (state.lap === lap) App.note(`Live analysis failed: ${err.message} Showing the saved results instead.`);
    } finally { lap.analyzing = false; }
  }

  // ------------------------------------------------------------------ tips from the LLM
  function setTip(lap, i, tip, by) {
    const seg = lap.segments[i];
    if (!seg) return;
    seg.tip = tip; seg.tip_by = by;
    if (state.lap !== lap) return;
    const shown = state.segs.find((s) => s.start_m === seg.start_m && s.end_m === seg.end_m);
    if (shown) { shown.tip = tip; shown.tip_by = by; }
    App.renderTips();
  }

  async function writeTips(lap, force = false) {
    if (!llm || !llm.model_ready || lap.is_reference || !lap.segments || !lap.segments.length) return;
    if (!lap.analyzed) return;                       // the tip model needs the telemetry context from the server
    if (lap.tippedBy === llm.model && !force) return;
    const run = ++tipRun;
    lap.tippedBy = llm.model;
    const ctl = new AbortController();
    lap.abort = ctl;
    tipsProgress = `Writing tips with ${llm.model}… 0/${lap.segments.length}`;
    renderCoach();
    try {
      const res = await http("/api/tips", { method: "POST", body: { segments: lap.segments, model: llm.model, stage: health && health.stage }, signal: ctl.signal });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
      const reader = res.body.getReader(), dec = new TextDecoder();
      let buf = "", done = 0;
      for (;;) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
          if (!line) continue;
          const m = JSON.parse(line);
          done += 1;
          if (m.tip) setTip(lap, m.index, m.tip, m.model);
          if (run === tipRun) { tipsProgress = `Writing tips with ${llm.model}… ${done}/${lap.segments.length}`; renderCoach(); }
        }
      }
    } catch (err) {
      lap.tippedBy = null;
      if (err.name !== "AbortError" && state.lap === lap) App.note(`The AI coach couldn't write tips: ${err.message}`);
    } finally {
      if (run === tipRun) { tipsProgress = ""; renderCoach(); }
    }
  }

  function rewriteTips() {
    const lap = state.lap;
    if (!lap || lap.is_reference) return;
    lap.segments.forEach((s) => { delete s.tip; delete s.tip_by; });
    state.segs.forEach((s) => { s.tip = undefined; s.tip_by = undefined; });
    App.setLapIndex(state.laps.indexOf(lap));       // re-enrich so the built-in tips show while the LLM writes
    writeTips(lap, true);
  }

  // ------------------------------------------------------------------ AI coach panel
  async function refreshLLM() {
    try {
      llm = await json(`/api/llm/status${model ? `?model=${encodeURIComponent(model)}` : ""}`, { timeout: 4000 });
      model = llm.model;
    } catch (_) { llm = null; }
    renderCoach();
    if (llm && (llm.download.active || llm.install.active) && !poll) poll = setInterval(pollProgress, 1000);
  }

  async function pollProgress() {
    try {
      llm = await json(`/api/llm/status${model ? `?model=${encodeURIComponent(model)}` : ""}`, { timeout: 4000 });
      if (!llm.download.active && !llm.install.active) {
        clearInterval(poll); poll = null;
        if (llm.model_ready) writeTips(state.lap);
      }
      renderCoach();
    } catch (_) { /* server hiccup; try again next tick */ }
  }
  const startPolling = () => { if (!poll) poll = setInterval(pollProgress, 1000); };

  const gb = (x) => `${x.toFixed(1)} GB`;
  function renderCoach() {
    const msg = $("coachMsg"), btn = $("coachBtn"), cancel = $("coachCancel"), link = $("coachLink"),
      sel = $("coachModel"), bar = $("coachBar"), fill = $("coachBarFill");
    btn.hidden = cancel.hidden = link.hidden = sel.hidden = bar.hidden = true;
    bar.classList.remove("is-indeterminate");
    btn.disabled = false;
    btn.onclick = null;
    if (!llm) { msg.textContent = "Couldn't reach the AI coach service."; btn.hidden = false; btn.textContent = "Check again"; btn.onclick = refreshLLM; return; }

    const dl = llm.download || {};
    const inst = llm.install || {};
    const offered = llm.models.find((m) => m.name === llm.model) || { size_gb: 0 };

    if (inst.active) {   // installing Ollama itself
      bar.hidden = false;
      const known = inst.stage === "downloading" && inst.total_mb;
      bar.classList.toggle("is-indeterminate", !known);
      fill.style.width = known ? `${inst.percent}%` : "";
      msg.textContent = known
        ? `${inst.label}: ${Math.round(inst.percent)}% (${Math.round(inst.completed_mb)} of ${Math.round(inst.total_mb)} MB). You can keep using the page.`
        : `${inst.label}… this can take a minute.`;
      if (inst.stage === "downloading") {
        cancel.hidden = false;
        cancel.onclick = async () => { await json("/api/llm/install/cancel", { method: "POST", body: {} }).catch(() => {}); };
      }
      return;
    }
    if (dl.active) {
      bar.hidden = false; cancel.hidden = false;
      const indeterminate = !dl.total_mb;
      bar.classList.toggle("is-indeterminate", indeterminate);
      fill.style.width = indeterminate ? "" : `${dl.percent}%`;
      msg.textContent = indeterminate
        ? `Starting the download of ${dl.model}…`
        : `Downloading ${dl.model}: ${Math.round(dl.percent)}% (${Math.round(dl.completed_mb)} of ${Math.round(dl.total_mb)} MB). You can keep using the page.`;
      cancel.onclick = async () => { await json("/api/llm/cancel", { method: "POST", body: {} }).catch(() => {}); };
      return;
    }
    if (llm.model_ready) {
      msg.textContent = tipsProgress || `Ready: ${llm.model} is running on this computer and writes the coaching tips.`;
      btn.hidden = false; btn.textContent = "Rewrite tips";
      btn.disabled = !!tipsProgress;
      btn.onclick = rewriteTips;
      if (llm.models.length > 1) fillModelSelect(sel, false);
      return;
    }
    if (!llm.ollama_installed) {
      if (!llm.install_supported) {   // e.g. Linux: installing needs administrator rights, so we can only show the command
        msg.textContent = `Ollama (the free program that runs the AI coach) needs administrator rights to install here. In a terminal run:  ${llm.install_command}  then click Check again.`;
        link.hidden = false; link.href = llm.download_page; link.textContent = "Ollama download page";
        btn.hidden = false; btn.textContent = "Check again"; btn.onclick = refreshLLM;
        return;
      }
      msg.textContent = inst.error
        ? `The install didn't finish: ${inst.error}`
        : "The AI coach runs on your own computer through Ollama, a free program. Click Install and Rally Coach downloads the official installer from ollama.com and sets it up for your account (no administrator prompt). Until then the page uses its built-in tips.";
      btn.hidden = false; btn.textContent = inst.error ? "Try again" : "Install Ollama";
      btn.onclick = async () => {
        btn.disabled = true;
        try { llm.install = await json("/api/llm/install", { method: "POST", body: {} }); } catch (e) { msg.textContent = e.message; btn.disabled = false; return; }
        renderCoach(); startPolling();
      };
      return;
    }
    if (!llm.ollama_running) {
      msg.textContent = "Ollama is installed but not running.";
      btn.hidden = false; btn.textContent = "Start Ollama";
      btn.onclick = async () => {
        btn.disabled = true; msg.textContent = "Starting Ollama…";
        try { const r = await json("/api/llm/start", { method: "POST", body: {} }); llm = r; } catch (e) { msg.textContent = e.message; }
        refreshLLM();
      };
      return;
    }
    // running, model not downloaded yet
    fillModelSelect(sel, true);
    btn.hidden = false; btn.textContent = `Download AI coach (${gb(offered.size_gb)})`;
    msg.textContent = dl.error
      ? `The download stopped: ${dl.error}`
      : `One-time download of ${gb(offered.size_gb)}. It stays on this computer and nothing about your laps is sent anywhere. ${offered.note || ""}`;
    btn.onclick = async () => {
      btn.disabled = true;
      try { llm.download = await json("/api/llm/pull", { method: "POST", body: { model: llm.model } }); } catch (e) { msg.textContent = e.message; btn.disabled = false; return; }
      renderCoach(); startPolling();
    };
  }

  function fillModelSelect(sel, canChoose) {
    sel.hidden = false; sel.disabled = !canChoose;
    sel.replaceChildren();
    for (const m of llm.models) {
      const o = document.createElement("option");
      o.value = m.name; o.textContent = `${m.label} (${gb(m.size_gb)})`;
      if (llm.installed.includes(m.name)) o.textContent += " ✓";
      sel.append(o);
    }
    sel.value = llm.model;
    sel.onchange = async () => {
      model = sel.value;
      try { localStorage.setItem("rally-model", model); } catch (_) {}
      await refreshLLM();
      if (llm && llm.model_ready) writeTips(state.lap);
    };
  }

  // ------------------------------------------------------------------ start
  document.addEventListener("rally-lap", (e) => {
    const lap = e.detail.lap;
    if (lap && !lap.is_reference) { if (lap.analyzed) writeTips(lap); else analyze(lap); }
    else App.note("");
  });

  (async () => {
    try { health = await json("/api/health", { timeout: 1500 }); } catch (_) { return; }   // no server: stay out of the way
    $("coachPanel").hidden = false;
    await refreshLLM();
    const lap = state.lap;
    if (lap && !lap.is_reference) analyze(lap);
  })();
})();
