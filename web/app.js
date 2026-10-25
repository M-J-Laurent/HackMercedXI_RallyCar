/* Rally Coach - static site. Data comes from data/bundle.js (window.RALLY_DATA, made by build_data.py)
   or from files the viewer drops in. No dependencies. */
(() => {
  "use strict";

  // ---------------------------------------------------------------- data
  const DATA = window.RALLY_DATA || { laps: [], grid_m: 5, stage_map: null, map_flip: false };
  const GRID = DATA.grid_m || 5;
  const STAGE = DATA.stage_map || null;
  const APPROACH_M = 100, RUNOUT_M = 40;

  const LABELS = {
    braked_too_early: { name: "Braked too early", did: "Started braking well before the ideal point and lost speed into the corner.", shows: "More brake than the reference, lower speed on entry." },
    braked_too_late: { name: "Braked too late", did: "Carried too much speed to the braking point, overshot the apex, then had to scrub speed.", shows: "Faster than the reference into the corner, slower after it." },
    lifted_mid_corner: { name: "Lifted mid-corner", did: "Came off the throttle through or just after the apex.", shows: "Less throttle and lower speed through the corner." },
    late_throttle: { name: "Late on the throttle", did: "Waited too long to get back on the power after the apex.", shows: "Less throttle and lower speed on the exit." },
    oversteer: { name: "Oversteer", did: "The rear stepped out and you had to counter-steer.", shows: "Large slide angle; steering against the corner." },
    understeer: { name: "Understeer", did: "The front washed wide and you added more steering.", shows: "More steering than the reference, lower speed." },
    wrong_gear: { name: "Wrong gear", did: "One gear off from the ideal for that speed.", shows: "Gear one above or below the reference." },
    spinout: { name: "Spin", did: "Lost the car and spun, or came to a stop.", shows: "Speed drops to near zero; engine stalls or reverse gear." },
    collision: { name: "Collision", did: "Hit something hard.", shows: "A sharp g spike followed by a stop." },
  };

  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const state = {
    laps: DATA.laps.slice(),
    lap: null, ref: null,
    n: 0, gap: [],
    segs: [],
    domain: null,           // [d0, d1] in metres
    hover: null,            // metres
    sel: null,              // index into state.segs
    sort: "distance",
    mapMode: "gap",
    brush: null,
  };

  // ---------------------------------------------------------------- formatting
  const fmtTime = (s) => {
    if (!isFinite(s)) return "–";
    const m = Math.floor(s / 60), r = s - m * 60;
    return `${m}:${r.toFixed(1).padStart(4, "0")}`;
  };
  const fmtSigned = (v, unit = "s", dp = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(dp) + (unit ? " " + unit : "");
  const fmtM = (m) => `${Math.round(m).toLocaleString()} m`;

  // ---------------------------------------------------------------- CSV upload -> trace
  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const head = lines[0].split(",").map((s) => s.trim());
    const rows = [];
    for (let k = 1; k < lines.length; k++) {
      const p = lines[k].split(",");
      if (p.length < head.length) continue;
      const o = {};
      head.forEach((h, j) => (o[h] = parseFloat(p[j])));
      rows.push(o);
    }
    return { head, rows };
  }
  const CH = { time: "t", speed: "v", throttle: "thr", brake: "brk", steer: "str", gear: "gear", rpm: "rpm",
    g_lat: "glat", g_long: "glong", slip_angle_deg: "slip", x: "x", z: "z" };
  function resampleRows(rows, head) {
    const d = rows.map((r) => r.lap_distance);
    const n = Math.floor(d[d.length - 1] / GRID) + 1;
    const tr = { d: [] };
    for (let i = 0; i < n; i++) tr.d.push(i * GRID);
    for (const c of Object.keys(CH)) {
      if (!head.includes(c)) continue;
      const out = new Array(n);
      let j = 0;
      for (let i = 0; i < n; i++) {
        const x = i * GRID;
        while (j < d.length && d[j] < x) j++;
        let v;
        if (j <= 0) v = rows[0][c];
        else if (j >= d.length) v = rows[d.length - 1][c];
        else if (c === "gear") v = rows[j][c];
        else { const w = (x - d[j - 1]) / (d[j] - d[j - 1]); v = rows[j - 1][c] + (rows[j][c] - rows[j - 1][c]) * w; }
        out[i] = c === "speed" ? v * 3.6 : v;
      }
      tr[CH[c]] = out;
    }
    return tr;
  }

  // ---------------------------------------------------------------- stage map lookup
  function locate(start, end) {
    if (!STAGE) return null;
    const cs = STAGE.corners;
    let best = null, bestOv = 0;
    for (const c of cs) {
      const ov = Math.min(c.end_m + RUNOUT_M, end) - Math.max(c.start_m - APPROACH_M, start);
      if (ov >= Math.max(10, 0.3 * (end - start)) && ov > bestOv) { best = c; bestOv = ov; }
    }
    if (best) return { number: best.number, direction: best.direction, kind: best.kind, apex_m: best.apex_m };
    const mid = (start + end) / 2;
    const before = cs.filter((c) => c.apex_m <= mid), after = cs.filter((c) => c.apex_m > mid);
    return { on_straight: true, after_corner: before.length ? before[before.length - 1].number : null,
      before_corner: after.length ? after[0].number : null };
  }
  function whereText(c, s) {
    const range = `${fmtM(s.start_m)}–${fmtM(s.end_m)}`;
    if (!c) return range;
    if (c.number) return `Corner ${c.number} · ${c.direction} ${c.kind} · ${range}`;
    if (c.after_corner && c.before_corner) return `Straight between corners ${c.after_corner} and ${c.before_corner} · ${range}`;
    return `Straight · ${range}`;
  }
  const cornerPhrase = (c) => (c && c.number ? ` at corner ${c.number}` : c && c.on_straight ? " on the straight" : "");

  // ---------------------------------------------------------------- analysis
  const idxAt = (m) => clamp(Math.round(m / GRID), 0, state.n - 1);

  function enrich(seg) {
    const L = state.lap.trace, R = state.ref.trace;
    const i0 = idxAt(seg.start_m), i1 = Math.max(idxAt(seg.end_m), i0);
    let vMin = Infinity, vRefMin = Infinity, slipMax = 0, earlyBrake = false, lift = false;
    for (let i = i0; i <= i1; i++) {
      vMin = Math.min(vMin, L.v[i]); vRefMin = Math.min(vRefMin, R.v[i]);
      if (L.slip) slipMax = Math.max(slipMax, Math.abs(L.slip[i]));
      if (L.brk[i] > 0.3 && R.brk[i] < 0.1) earlyBrake = true;
      if (L.thr[i] < 0.2 && R.thr[i] > 0.6) lift = true;
    }
    const lost = seg.evidence && isFinite(seg.evidence.time_lost_s)
      ? seg.evidence.time_lost_s : state.gap[i1] - state.gap[i0];
    const label = seg.suggested_type || (seg.rule && seg.rule.label) || null;
    const corner = seg.corner || locate(seg.start_m, seg.end_m);
    const m = { vMin, vRefMin, slipMax, earlyBrake, lift, lost, corner };
    return { ...seg, label, lost, corner, metrics: m, tip: seg.tip || makeTip(label, m, seg) };
  }

  function makeTip(label, m, seg) {
    const at = cornerPhrase(m.corner);
    const dv = Math.round(m.vRefMin - m.vMin);
    const g = seg.evidence && seg.evidence.gear_delta;
    switch (label) {
      case "braked_too_early": return `You got on the brakes too early${at}. Brake later and carry more speed in; the reference was ${dv} km/h faster at the slowest point.`;
      case "braked_too_late": return `Too much speed into the braking zone${at}: you overshot and had to scrub speed. Brake a little earlier so you can get back on the power sooner.`;
      case "lifted_mid_corner": return `You came off the throttle mid-corner${at}. Hold a steady part-throttle to keep the car balanced.`;
      case "late_throttle": return `You waited too long to get back on the power after the apex${at}. Start feeding in throttle as soon as the car is pointed at the exit.`;
      case "oversteer": return `The rear stepped out${at} (slide angle up to ${Math.round(m.slipMax)}°). Smoother throttle and a little less entry speed keep the car straighter.`;
      case "understeer": return `The front washed wide${at}. Slow down a bit more before turning in instead of adding steering.`;
      case "wrong_gear": return `You were a gear ${g > 0 ? "higher" : "lower"} than the reference${at}. Match the gear to the speed for better drive.`;
      case "spinout": return `You spun${at} and lost ${Math.abs(m.lost).toFixed(1)} s. Back off the entry speed here until the line feels comfortable.`;
      case "collision": return `You hit something${at}. A slower, tidier line through here is worth more than the risk.`;
      default: {
        const bits = [];
        if (m.earlyBrake) bits.push("you were braking where the reference wasn't");
        if (m.lift) bits.push("you were off the throttle where the reference was flat out");
        if (m.slipMax > 25) bits.push(`the car was sliding (up to ${Math.round(m.slipMax)}°)`);
        const why = bits.length ? ` Here ${bits.join(", and ")}.` : "";
        return `You lost ${Math.abs(m.lost).toFixed(1)} s${at}. Your slowest speed was ${Math.round(m.vMin)} km/h against ${Math.round(m.vRefMin)} km/h on the reference.${why}`;
      }
    }
  }

  // ---------------------------------------------------------------- lap selection
  function setLap(k) {
    state.lap = state.laps[k];
    state.ref = state.laps.find((l) => l.is_reference) || state.laps[0];
    const L = state.lap.trace, R = state.ref.trace;
    state.n = Math.min(L.d.length, R.d.length);
    state.gap = new Array(state.n);
    for (let i = 0; i < state.n; i++) state.gap[i] = L.t[i] - R.t[i];
    state.segs = state.lap === state.ref ? [] : (state.lap.segments || [])
      .filter((s) => s.start_m < state.n * GRID).map(enrich);
    state.domain = [0, (state.n - 1) * GRID];
    state.sel = null; state.hover = null;
    renderAll();
  }

  function renderAll() {
    renderStats(); renderTips(); drawMap(); charts.forEach((c) => c.draw()); renderMapLegend();
  }

  // ---------------------------------------------------------------- stats
  function renderStats() {
    const L = state.lap.trace, R = state.ref.trace;
    const tLap = L.t[L.t.length - 1], tRef = R.t[R.t.length - 1], gap = tLap - tRef;
    $("stTime").textContent = fmtTime(tLap);
    $("stRef").textContent = fmtTime(tRef);
    const g = $("stGap");
    g.textContent = state.lap === state.ref ? "Reference lap" : fmtSigned(gap);
    g.className = "stat-value" + (state.lap === state.ref ? " stat-small" : gap > 0 ? " loss" : " gain");
    $("stCount").textContent = state.lap === state.ref ? "–" : String(state.segs.length);
    const worst = state.segs.slice().sort((a, b) => b.lost - a.lost)[0];
    $("stWorst").textContent = worst ? `${fmtSigned(worst.lost)} · ${worst.corner && worst.corner.number ? "corner " + worst.corner.number : fmtM(worst.start_m)}` : "–";
    $("refName").textContent = `${state.ref.name} (${fmtTime(tRef)})`;
    $("stageName").textContent = [state.lap.stage, state.lap.car].filter(Boolean).join(" · ");
  }

  // ---------------------------------------------------------------- tips
  function renderTips() {
    const ol = $("tips"); ol.replaceChildren();
    const src = state.lap.segments_source;
    $("sourceNote").textContent = state.lap === state.ref
      ? "This is the reference lap, so there is nothing to compare. Pick another lap."
      : src === "driver_notes"
        ? "Mistake locations come from the driver's notes; the model's output hasn't been added for this lap yet. Tips are auto-written from the telemetry."
        : src === "model" ? "Mistakes found by the anomaly model. Tips come from the LLM where available, otherwise they're auto-written from the telemetry."
        : "";
    if (!state.segs.length) {
      const li = document.createElement("li"); li.className = "empty";
      li.textContent = state.lap === state.ref ? "Reference lap" : "No mistakes found on this lap. Clean run!";
      ol.append(li); return;
    }
    const order = state.segs.map((s, i) => i);
    if (state.sort === "loss") order.sort((a, b) => state.segs[b].lost - state.segs[a].lost);
    order.forEach((i, rank) => {
      const s = state.segs[i];
      const li = document.createElement("li");
      li.className = "tip" + (state.sel === i ? " is-selected" : "");
      li.tabIndex = 0; li.dataset.i = i;
      const num = el("span", "tip-num", String(rank + 1));
      const title = el("span", "tip-title", s.label && LABELS[s.label] ? LABELS[s.label].name : "Time lost");
      const loss = el("span", "tip-loss " + (s.lost > 0 ? "loss" : "gain"), fmtSigned(s.lost));
      const where = el("span", "tip-where", whereText(s.corner, s));
      const text = el("p", "tip-text", s.tip);
      li.append(num, title, loss, where, text);
      if (typeof s.confidence === "number") li.append(el("span", "tip-src", `${s.source || "model"} · confidence ${Math.round(s.confidence * 100)}%`));
      li.addEventListener("click", () => selectSeg(i, false));
      li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectSeg(i, false); } });
      ol.append(li);
    });
  }
  function el(tag, cls, text) { const e = document.createElement(tag); e.className = cls; e.textContent = text; return e; }

  function selectSeg(i, scroll) {
    if (i === null || state.sel === i) { state.sel = null; state.domain = [0, (state.n - 1) * GRID]; }
    else {
      state.sel = i;
      const s = state.segs[i], pad = Math.max(150, (s.end_m - s.start_m));
      state.domain = [Math.max(0, s.start_m - pad), Math.min((state.n - 1) * GRID, s.end_m + pad)];
    }
    renderTips(); drawMap(); charts.forEach((c) => c.draw());
    if (scroll && state.sel !== null) {
      const li = document.querySelector(`.tip[data-i="${state.sel}"]`);
      if (li) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
  const segAt = (m) => { const k = state.segs.findIndex((s) => m >= s.start_m - GRID && m <= s.end_m + GRID); return k < 0 ? null : k; };

  // ---------------------------------------------------------------- charts
  function niceTicks(lo, hi, count) {
    const span = hi - lo || 1, raw = span / count, p = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }

  class Chart {
    constructor(spec) {
      this.spec = spec; this.isLast = true; // every panel gets its own distance axis
      const wrap = document.createElement("section"); wrap.className = "chart";
      wrap.hidden = !spec.show;
      const head = document.createElement("div"); head.className = "chart-head";
      head.append(el("h3", "chart-title", spec.title));
      if (spec.unit) head.append(el("span", "chart-unit", spec.unit));
      wrap.append(head, el("p", "chart-desc", spec.desc));
      this.canvas = document.createElement("canvas");
      this.canvas.setAttribute("role", "img"); this.canvas.setAttribute("aria-label", `${spec.title}: ${spec.desc}`);
      wrap.append(this.canvas); $("charts").append(wrap);
      this.wrap = wrap;
      this.pad = { l: 44, r: 10, t: 8, b: 22 };
      this.h = spec.h + this.pad.t + this.pad.b;
      this.canvas.style.height = this.h + "px";
      new ResizeObserver(() => this.draw()).observe(this.canvas);
      this.bind();
    }
    get w() { return this.canvas.clientWidth; }
    xOf(m) { const [a, b] = state.domain; return this.pad.l + (m - a) / (b - a) * (this.w - this.pad.l - this.pad.r); }
    mOf(x) { const [a, b] = state.domain; return a + (x - this.pad.l) / (this.w - this.pad.l - this.pad.r) * (b - a); }
    yDomain(i0, i1) {
      if (this.spec.domain) return this.spec.domain;
      let lo = Infinity, hi = -Infinity;
      for (const s of this.spec.series) {
        const arr = s.arr(); if (!arr) continue;
        for (let i = i0; i <= i1; i++) { const v = arr[i]; if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
      }
      if (this.spec.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      const p = (hi - lo) * 0.08 || 1; return [lo - p, hi + p];
    }
    draw() {
      if (!state.lap) return;
      const c = this.canvas, dpr = window.devicePixelRatio || 1, W = this.w, H = this.h;
      if (!W) return;
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
      const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      const { l, r, t, b } = this.pad, plotH = H - t - b;
      const i0 = idxAt(state.domain[0]), i1 = idxAt(state.domain[1]);
      const [y0, y1] = this.yDomain(i0, i1);
      const yOf = (v) => t + (1 - (v - y0) / (y1 - y0)) * plotH;

      // mistake shading
      state.segs.forEach((s, k) => {
        const xa = this.xOf(s.start_m), xb = this.xOf(s.end_m);
        if (xb < l || xa > W - r) return;
        g.fillStyle = k === state.sel ? css("--mistake-strong") : css("--mistake");
        g.fillRect(Math.max(xa, l), t, Math.min(xb, W - r) - Math.max(xa, l), plotH);
      });

      // grid + y labels
      g.font = "11px " + css("--font"); g.textBaseline = "middle"; g.textAlign = "right";
      for (const v of niceTicks(y0, y1, 3)) {
        const y = Math.round(yOf(v)) + 0.5;
        g.strokeStyle = v === 0 && (this.spec.zero || this.spec.domain) ? css("--text-muted") : css("--grid");
        g.lineWidth = 1; g.beginPath(); g.moveTo(l, y); g.lineTo(W - r, y); g.stroke();
        g.fillStyle = css("--text-muted"); g.fillText(this.spec.tick ? this.spec.tick(v) : String(v), l - 6, y);
      }
      // x labels
      if (this.isLast) {
        g.textAlign = "center"; g.textBaseline = "top";
        const span = state.domain[1] - state.domain[0];
        for (const m of niceTicks(state.domain[0], state.domain[1], Math.max(2, Math.floor(W / 110)))) {
          const x = this.xOf(m); if (x < l || x > W - r) continue;
          const lab = span < 3000 || m < 1000 ? Math.round(m).toLocaleString() + " m" : (m / 1000).toFixed(m % 1000 ? 1 : 0) + " km";
          g.fillStyle = css("--text-muted"); g.fillText(lab, x, H - b + 6);
        }
      }

      // series (reference under, you on top)
      g.save(); g.beginPath(); g.rect(l, t, W - l - r, plotH); g.clip();
      for (const s of this.spec.series.slice().reverse()) {
        const arr = s.arr(); if (!arr) continue;
        g.strokeStyle = css(s.color); g.lineWidth = s.width || 2; g.lineJoin = "round"; g.beginPath();
        let started = false;
        let py = null;
        for (let i = Math.max(0, i0 - 1); i <= Math.min(state.n - 1, i1 + 1); i++) {
          const x = this.xOf(i * GRID), y = yOf(arr[i]);
          if (!started) { g.moveTo(x, y); started = true; }
          else if (this.spec.step) { g.lineTo(x, py); g.lineTo(x, y); }
          else g.lineTo(x, y);
          py = y;
        }
        g.stroke();
      }
      g.restore();

      // brush
      if (state.brush && state.brush.chart === this) {
        const xa = this.xOf(Math.min(state.brush.a, state.brush.b)), xb = this.xOf(Math.max(state.brush.a, state.brush.b));
        g.fillStyle = "rgba(42,120,214,0.12)"; g.fillRect(xa, t, xb - xa, plotH);
      }

      // crosshair
      if (state.hover !== null && state.hover >= state.domain[0] && state.hover <= state.domain[1]) {
        const x = Math.round(this.xOf(state.hover)) + 0.5, i = idxAt(state.hover);
        g.strokeStyle = css("--text-secondary"); g.lineWidth = 1;
        g.beginPath(); g.moveTo(x, t); g.lineTo(x, t + plotH); g.stroke();
        for (const s of this.spec.series) {
          const arr = s.arr(); if (!arr) continue;
          g.beginPath(); g.arc(x, yOf(arr[i]), 4, 0, Math.PI * 2);
          g.fillStyle = css(s.color); g.fill(); g.lineWidth = 2; g.strokeStyle = css("--surface-1"); g.stroke();
        }
      }
    }
    bind() {
      const c = this.canvas;
      const mAt = (e) => clamp(this.mOf(e.offsetX), state.domain[0], state.domain[1]);
      c.addEventListener("pointerdown", (e) => { state.brush = { chart: this, a: mAt(e), b: mAt(e), x: e.offsetX }; c.setPointerCapture(e.pointerId); });
      c.addEventListener("pointermove", (e) => {
        const m = mAt(e);
        if (state.brush && state.brush.chart === this) state.brush.b = m;
        setHover(m, e.clientX, e.clientY);
      });
      c.addEventListener("pointerup", (e) => {
        const br = state.brush; state.brush = null;
        if (!br) return;
        if (Math.abs(e.offsetX - br.x) > 8) {
          const a = Math.min(br.a, br.b), b = Math.max(br.a, br.b);
          if (b - a > 20) { state.domain = [a, b]; charts.forEach((ch) => ch.draw()); }
        } else selectSeg(segAt(br.a), true);
      });
      c.addEventListener("pointerleave", () => { if (!state.brush) setHover(null); });
      c.addEventListener("dblclick", () => { state.domain = [0, (state.n - 1) * GRID]; charts.forEach((ch) => ch.draw()); });
    }
  }

  const you = (k) => () => state.lap && state.lap.trace[k];
  const ref = (k) => () => state.ref && state.ref.trace[k];
  const pct = (v) => Math.round(v * 100) + "%";
  const both = (k) => [{ arr: you(k), color: "--you" }, { arr: ref(k), color: "--ref", width: 1.5 }];
  const CHART_SPECS = [
    { id: "gap", title: "Time gap", unit: "seconds", show: true, h: 120, zero: true,
      desc: "How far behind (+) or ahead (−) of the reference you are. Where the line climbs, you're losing time.",
      series: [{ arr: () => state.gap, color: "--you" }] },
    { id: "speed", title: "Speed", unit: "km/h", show: true, h: 170, series: both("v"),
      desc: "Where the blue line dips below the orange one, you were slower than the reference." },
    { id: "throttle", title: "Throttle", unit: "% pressed", show: true, h: 90, domain: [-0.05, 1.05], tick: pct, series: both("thr"),
      desc: "How far the gas pedal was pressed. Gaps in blue where orange is at 100% = time you could be flat out." },
    { id: "brake", title: "Brake", unit: "% pressed", show: true, h: 90, domain: [-0.05, 1.05], tick: pct, series: both("brk"),
      desc: "How hard you braked. A blue spike before the orange one means you braked earlier than the reference." },
    { id: "gear", title: "Gear", unit: "gear", show: false, h: 90, step: true, series: both("gear"),
      desc: "Which gear you were in. A different gear from the reference at the same spot can cost drive." },
    { id: "steer", title: "Steering", unit: "left is up", show: false, h: 100, domain: [-1.05, 1.05], series: both("str"),
      desc: "Steering input. Quick swings from one side to the other usually mean you were catching a slide." },
    { id: "slip", title: "Slide angle", unit: "degrees", show: true, h: 110, zero: true, series: both("slip"),
      desc: "How sideways the car is. Near 0 = going where it points. Big numbers = sliding (the bigger, the more time it costs)." },
  ];
  let charts = [];

  // ---------------------------------------------------------------- tooltip + hover
  function setHover(m, cx, cy) {
    state.hover = m;
    charts.forEach((c) => c.draw()); drawMap();
    const tt = $("tooltip");
    if (m === null) { tt.hidden = true; return; }
    const i = idxAt(m), L = state.lap.trace, R = state.ref.trace;
    tt.replaceChildren();
    const c = locate(m, m);
    tt.append(el("div", "tt-head", `${fmtM(m)}${c && c.number ? " · corner " + c.number : ""}`));
    const row = (label, val, color) => {
      const r = document.createElement("div"); r.className = "tt-row";
      if (color) { const k = document.createElement("i"); k.style.background = css(color); r.append(k); }
      r.append(document.createTextNode(label)); r.append(el("b", "", val)); tt.append(r);
    };
    row("Gap", fmtSigned(state.gap[i], "s", 2));
    row("You", `${Math.round(L.v[i])} km/h · gear ${L.gear[i]}`, "--you");
    row("Ref", `${Math.round(R.v[i])} km/h · gear ${R.gear[i]}`, "--ref");
    row("Throttle", `${pct(L.thr[i])} / ${pct(R.thr[i])}`);
    row("Brake", `${pct(L.brk[i])} / ${pct(R.brk[i])}`);
    if (L.slip) row("Slide", `${Math.round(L.slip[i])}° / ${R.slip ? Math.round(R.slip[i]) : "–"}°`);
    tt.hidden = false;
    if (cx !== undefined) {
      const w = tt.offsetWidth, h = tt.offsetHeight;
      tt.style.left = clamp(cx + 14, 8, window.innerWidth - w - 8) + "px";
      tt.style.top = clamp(cy - h - 12, 8, window.innerHeight - h - 8) + "px";
    }
  }

  // ---------------------------------------------------------------- map
  const mapCanvas = $("map");
  let mapProj = null;
  function lerpHex(a, b, t) {
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const ch = (s) => Math.round(((pa >> s) & 255) + (((pb >> s) & 255) - ((pa >> s) & 255)) * t);
    return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
  }
  const SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
  const seqColor = (t) => { t = clamp(t, 0, 1) * (SEQ.length - 1); const k = Math.min(Math.floor(t), SEQ.length - 2); return lerpHex(SEQ[k], SEQ[k + 1], t - k); };
  const divColor = (t) => { // -1 gain (blue) .. 0 gray .. +1 loss (red)
    const mid = document.documentElement.dataset.theme === "dark" || (matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light") ? "#5c5b56" : "#c9c7c0";
    t = clamp(t, -1, 1);
    return t < 0 ? lerpHex(mid, "#2a78d6", -t) : lerpHex(mid, "#d03b3b", t);
  };
  function gapRate() {
    const k = 4, out = new Array(state.n);
    for (let i = 0; i < state.n; i++) out[i] = state.gap[Math.min(i + k, state.n - 1)] - state.gap[Math.max(i - k, 0)];
    const s = out.map(Math.abs).sort((a, b) => a - b); const q = s[Math.floor(s.length * 0.95)] || 1;
    return out.map((v) => v / q);
  }

  function drawMap() {
    if (!state.lap) return;
    const c = mapCanvas, dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
    if (!W) return;
    c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const R = state.ref.trace, L = state.lap.trace;
    if (!R.x || !L.x) { g.fillStyle = css("--text-muted"); g.fillText("No position data", 12, 20); return; }
    const ys = DATA.map_flip ? 1 : -1;
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (let i = 0; i < state.n; i++) { x0 = Math.min(x0, R.x[i]); x1 = Math.max(x1, R.x[i]); z0 = Math.min(z0, R.z[i] * ys); z1 = Math.max(z1, R.z[i] * ys); }
    const pad = 28, s = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (z1 - z0));
    const ox = (W - (x1 - x0) * s) / 2, oy = (H - (z1 - z0) * s) / 2;
    const P = (x, z) => [ox + (x - x0) * s, oy + (z * ys - z0) * s];
    mapProj = { P, L };

    // road underlay (reference line)
    g.lineCap = "round"; g.lineJoin = "round";
    g.strokeStyle = css("--surface-2"); g.lineWidth = 9; g.beginPath();
    for (let i = 0; i < state.n; i++) { const [x, y] = P(R.x[i], R.z[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke();

    // mistakes halo
    state.segs.forEach((sg, k) => {
      g.strokeStyle = k === state.sel ? css("--critical") : css("--mistake-strong"); g.lineWidth = k === state.sel ? 12 : 11;
      g.beginPath();
      for (let i = idxAt(sg.start_m); i <= idxAt(sg.end_m); i++) { const [x, y] = P(L.x[i], L.z[i]); i === idxAt(sg.start_m) ? g.moveTo(x, y) : g.lineTo(x, y); }
      g.stroke();
    });

    // your line colored
    const rate = state.mapMode === "gap" ? gapRate() : null;
    let vmax = 0; for (let i = 0; i < state.n; i++) vmax = Math.max(vmax, L.v[i]);
    g.lineWidth = 3.5;
    for (let i = 1; i < state.n; i++) {
      const [xa, ya] = P(L.x[i - 1], L.z[i - 1]), [xb, yb] = P(L.x[i], L.z[i]);
      g.strokeStyle = rate ? divColor(rate[i]) : seqColor(L.v[i] / vmax);
      g.beginPath(); g.moveTo(xa, ya); g.lineTo(xb, yb); g.stroke();
    }

    // corner numbers (outside of each bend)
    if (STAGE) {
      g.font = "600 10px " + css("--font"); g.textAlign = "center"; g.textBaseline = "middle";
      for (const cn of STAGE.corners) {
        const i = idxAt(cn.apex_m), a = P(R.x[Math.max(i - 2, 0)], R.z[Math.max(i - 2, 0)]), b = P(R.x[Math.min(i + 2, state.n - 1)], R.z[Math.min(i + 2, state.n - 1)]);
        let tx = b[0] - a[0], ty = b[1] - a[1]; const len = Math.hypot(tx, ty) || 1; tx /= len; ty /= len;
        const [nx, ny] = cn.direction === "left" ? [-ty, tx] : [ty, -tx]; // outside of the bend
        const [px, py] = P(R.x[i], R.z[i]), lx = px + nx * 15, ly = py + ny * 15;
        g.beginPath(); g.arc(lx, ly, 8, 0, Math.PI * 2); g.fillStyle = css("--surface-1"); g.fill();
        g.lineWidth = 1; g.strokeStyle = css("--border"); g.stroke();
        g.fillStyle = css("--text-secondary"); g.fillText(String(cn.number), lx, ly + 0.5);
      }
    }

    // start / finish
    const mark = (i, txt, col) => {
      const [x, y] = P(L.x[i], L.z[i]);
      g.beginPath(); g.arc(x, y, 6, 0, Math.PI * 2); g.fillStyle = col; g.fill(); g.lineWidth = 2; g.strokeStyle = css("--surface-1"); g.stroke();
      g.font = "600 11px " + css("--font"); g.fillStyle = css("--text-primary"); g.textAlign = "left"; g.fillText(txt, x + 10, y);
    };
    mark(0, "Start", css("--good")); mark(state.n - 1, "Finish", css("--text-primary"));

    // hover marker
    if (state.hover !== null) {
      const i = idxAt(state.hover), [x, y] = P(L.x[i], L.z[i]);
      g.beginPath(); g.arc(x, y, 6, 0, Math.PI * 2); g.fillStyle = css("--you"); g.fill();
      g.lineWidth = 3; g.strokeStyle = css("--surface-1"); g.stroke();
    }
  }
  function nearestOnMap(ex, ey) {
    if (!mapProj) return null;
    let best = null, bd = 400; // within 20 px
    for (let i = 0; i < state.n; i++) {
      const [x, y] = mapProj.P(mapProj.L.x[i], mapProj.L.z[i]), d = (x - ex) ** 2 + (y - ey) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best === null ? null : best * GRID;
  }
  mapCanvas.addEventListener("pointermove", (e) => { const m = nearestOnMap(e.offsetX, e.offsetY); setHover(m, e.clientX, e.clientY); });
  mapCanvas.addEventListener("pointerleave", () => setHover(null));
  mapCanvas.addEventListener("click", (e) => { const m = nearestOnMap(e.offsetX, e.offsetY); if (m !== null) selectSeg(segAt(m), true); });
  new ResizeObserver(drawMap).observe(mapCanvas);

  function renderMapLegend() {
    const box = $("mapLegend"); box.replaceChildren();
    const ramp = document.createElement("span"); ramp.className = "ramp";
    if (state.mapMode === "gap") {
      ramp.style.background = `linear-gradient(90deg, ${divColor(-1)}, ${divColor(0)}, ${divColor(1)})`;
      box.append(el("span", "", "Gaining"), ramp, el("span", "", "Losing time vs reference"));
    } else {
      ramp.style.background = `linear-gradient(90deg, ${SEQ.join(",")})`;
      let vmax = 0; state.lap.trace.v.forEach((v) => (vmax = Math.max(vmax, v)));
      box.append(el("span", "", "0"), ramp, el("span", "", `${Math.round(vmax)} km/h`));
    }
    const k = document.createElement("span"); k.className = "key-box mistake"; k.style.marginLeft = "12px";
    box.append(k, el("span", "", "Mistake"), el("span", "", " · numbers = corners"));
  }

  // ---------------------------------------------------------------- uploads
  $("uploadLap").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try {
      const { head, rows } = parseCSV(await f.text());
      for (const need of ["lap_distance", "time", "speed", "throttle", "brake", "steer", "gear"])
        if (!head.includes(need)) throw new Error(`missing column "${need}"`);
      const lap = { file: f.name, name: f.name.replace(/\.csv$/i, "") + " (uploaded)", trace: resampleRows(rows, head), segments: [], segments_source: null };
      state.laps.push(lap); fillLapSelect(); $("lapSelect").value = String(state.laps.length - 1); setLap(state.laps.length - 1);
    } catch (err) { alertMsg(`Couldn't read that CSV: ${err.message}`); }
    e.target.value = "";
  });
  $("uploadJson").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try {
      const j = JSON.parse(await f.text());
      const segs = Array.isArray(j) ? j : j.segments;
      if (!Array.isArray(segs)) throw new Error("expected a list of segments");
      state.lap.segments = segs; state.lap.segments_source = "model";
      setLap(state.laps.indexOf(state.lap));
    } catch (err) { alertMsg(`Couldn't read that JSON: ${err.message}`); }
    e.target.value = "";
  });
  function alertMsg(msg) { $("sourceNote").textContent = msg; }

  // ---------------------------------------------------------------- controls, tabs, theme
  function fillLapSelect() {
    const s = $("lapSelect"); s.replaceChildren();
    state.laps.forEach((l, k) => {
      const o = document.createElement("option"); o.value = String(k);
      o.textContent = `${l.name}${l.is_reference ? " (reference)" : ""}`; s.append(o);
    });
  }
  $("lapSelect").addEventListener("change", (e) => setLap(+e.target.value));
  $("mapMode").addEventListener("change", (e) => { state.mapMode = e.target.value; drawMap(); renderMapLegend(); });
  $("resetZoom").addEventListener("click", () => selectSeg(null));
  document.querySelectorAll("[data-sort]").forEach((b) => b.addEventListener("click", () => {
    state.sort = b.dataset.sort;
    document.querySelectorAll("[data-sort]").forEach((x) => x.classList.toggle("is-active", x === b));
    renderTips();
  }));

  function showView(v) {
    document.querySelectorAll(".tab").forEach((t) => { const on = t.dataset.view === v; t.classList.toggle("is-active", on); t.setAttribute("aria-selected", on); });
    $("view-analysis").hidden = v !== "analysis"; $("view-about").hidden = v !== "about";
    if (v === "analysis") requestAnimationFrame(renderAll);
  }
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => { location.hash = t.dataset.view === "about" ? "about" : ""; showView(t.dataset.view); }));

  $("themeToggle").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("rally-theme", root.dataset.theme); } catch (_) {}
    renderAll();
  });
  try { const t = localStorage.getItem("rally-theme"); if (t) document.documentElement.dataset.theme = t; } catch (_) {}

  // glossary
  const gl = $("glossary");
  for (const k of Object.keys(LABELS)) {
    const tr = document.createElement("tr");
    tr.append(el("td", "", LABELS[k].name), el("td", "", LABELS[k].did), el("td", "", LABELS[k].shows));
    gl.append(tr);
  }

  // ---------------------------------------------------------------- boot
  CHART_SPECS.forEach((s) => charts.push(new Chart(s)));
  // chart on/off toggles
  const tg = $("chartToggles");
  charts.forEach((c) => {
    const b = el("button", "chip" + (c.spec.show ? " is-active" : ""), c.spec.title);
    b.type = "button"; b.setAttribute("aria-pressed", String(c.spec.show));
    b.addEventListener("click", () => {
      const on = c.wrap.hidden; c.wrap.hidden = !on;
      b.classList.toggle("is-active", on); b.setAttribute("aria-pressed", String(on));
      if (on) requestAnimationFrame(() => c.draw());
    });
    tg.append(b);
  });
  showView(location.hash === "#about" ? "about" : "analysis");
  if (!state.laps.length) {
    $("sourceNote").textContent = "No data bundled yet. Run  python web/build_data.py  or load a lap CSV above.";
    return;
  }
  fillLapSelect();
  // open on the first lap that has mistakes to show
  const start = Math.max(0, state.laps.findIndex((l) => !l.is_reference && (l.segments || []).length));
  $("lapSelect").value = String(start);
  setLap(start);
})();
