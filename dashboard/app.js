/* BenchLab dashboard — vanilla JS, no build step, no external deps.
   Fetches the JSON API served by scripts/dashboard_server.py and renders a
   telemetry-console view of each benchmark run plus a cross-machine compare. */

"use strict";

const MACHINE_COLORS = [
  "#a6ff3f", "#45c8f0", "#ffb020", "#b98bff",
  "#ff5c5c", "#4ade80", "#f472b6", "#38bdf8",
];

const state = {
  runs: [],
  colorByMachine: {},
  activePath: null,
  view: "run",
  sort: { key: "gen_tps_mean", dir: -1 },
  sidebarSort: "date", // "date" | "machine"
  lastReport: null,
};

/* --------------------------------------------------------------- helpers */
function el(tag, attrs, children) {
  const n = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k === "style") n.setAttribute("style", v);
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  if (children != null) {
    for (const c of [].concat(children)) {
      if (c == null) continue;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
  }
  return n;
}

function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}

function shortTime(iso) {
  // reports encode timestamps like 2026-05-03T08:58:33+02:00
  const m = String(iso).match(/(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/);
  if (!m) return iso;
  return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}`;
}

function colorFor(machine) {
  return state.colorByMachine[machine] || MACHINE_COLORS[0];
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

function setStatus(cls, text) {
  const s = document.getElementById("status");
  s.className = "status " + cls;
  document.getElementById("status-text").textContent = text;
}

/* ------------------------------------------------------------- data load */
async function loadRuns() {
  setStatus("", "loading…");
  const data = await getJSON("/api/runs");
  state.runs = data.runs || [];
  const machines = [...new Set(state.runs.map((r) => r.machine_label))].sort();
  state.colorByMachine = {};
  machines.forEach((m, i) => (state.colorByMachine[m] = MACHINE_COLORS[i % MACHINE_COLORS.length]));

  document.getElementById("run-count").textContent = state.runs.length;
  renderSidebar();
  setStatus("is-live", `${state.runs.length} runs · ${machines.length} machines`);

  if (state.runs.length) {
    const stillThere = state.runs.find((r) => r.path === state.activePath);
    selectRun((stillThere || state.runs[0]).path);
  } else {
    document.getElementById("view-run").innerHTML =
      '<div class="empty"><div><div class="big">No reports found</div>' +
      'Run <code>python3 scripts/ollama_bench.py</code> to generate one.</div></div>';
  }
}

/* --------------------------------------------------------------- sidebar */
function runItem(run, opts = {}) {
  const color = colorFor(run.machine_label);
  const sub = opts.showMachine
    ? `${run.machine_label} · ${run.engine}`
    : `${run.engine} · ${(run.models || []).length} models`;
  return el("div", {
    class: "run-item" + (run.path === state.activePath ? " is-active" : ""),
    style: "position:relative",
    "data-path": run.path,
    onclick: () => selectRun(run.path),
  }, [
    opts.showMachine ? el("span", { class: "ri-dot", style: `background:${color};color:${color}` }) : null,
    el("div", { class: "ri-main" }, [
      el("div", { class: "ri-time" }, shortTime(run.started)),
      el("div", { class: "ri-sub", title: sub }, sub),
    ]),
    run.is_latest ? el("span", { class: "badge-latest" }, "latest") : null,
  ]);
}

function renderSidebar() {
  const list = document.getElementById("run-list");
  list.innerHTML = "";

  if (state.sidebarSort === "date") {
    // state.runs already arrives newest-first from the API.
    const sorted = [...state.runs].sort((a, b) => String(b.started).localeCompare(String(a.started)));
    for (const run of sorted) list.appendChild(runItem(run, { showMachine: true }));
    return;
  }

  const groups = {};
  for (const run of state.runs) (groups[run.machine_label] ||= []).push(run);
  for (const machine of Object.keys(groups).sort()) {
    const group = el("div", { class: "machine-group" }, [
      el("div", { class: "machine-name", style: `--accent:${colorFor(machine)}` }, machine),
    ]);
    for (const run of groups[machine]) group.appendChild(runItem(run));
    list.appendChild(group);
  }
}

async function selectRun(path) {
  state.activePath = path;
  document.querySelectorAll(".run-item").forEach((n) =>
    n.classList.toggle("is-active", n.getAttribute("data-path") === path));
  const view = document.getElementById("view-run");
  view.innerHTML = '<div class="empty"><div><div class="spinner"></div>reading report…</div></div>';
  try {
    const report = await getJSON("/api/run?path=" + encodeURIComponent(path));
    state.lastReport = report;
    renderRun(report);
  } catch (e) {
    view.innerHTML = `<div class="empty"><div class="big">Failed to load report</div><div>${e.message}</div></div>`;
  }
}

/* ------------------------------------------------------------- run detail */
function renderRun(report) {
  const view = document.getElementById("view-run");
  view.innerHTML = "";
  const models = (report.models || []).filter(Boolean);
  const color = colorFor(report.machine_label);

  // header
  const head = el("div", { class: "runhead reveal" }, [
    el("h1", {}, [report.machine_label, el("span", { class: "cursor" }, "▊")]),
    el("div", { class: "chips" }, [
      chip("engine", report.engine),
      chip("platform", report.platform),
      chip("started", shortTime(report.started)),
      chip("runs/model", (report.runs_per_model || "").replace(/`/g, "")),
      el("span", { class: "chip accent" }, `${models.length} models`),
    ]),
  ]);
  // hardware readout
  const hw = report.hardware || [];
  if (hw.length) {
    const grid = el("div", { class: "hw-readout" });
    for (const cell of hw) grid.appendChild(el("div", { class: "hw-cell" }, [
      el("div", { class: "k" }, cell.key), el("div", { class: "v" }, cell.value || "—"),
    ]));
    if (report.peak_cpu) grid.appendChild(hwCell("peak cpu", report.peak_cpu));
    if (report.peak_ram) grid.appendChild(hwCell("peak ram", report.peak_ram));
    head.appendChild(grid);
  }
  view.appendChild(head);

  if (!models.length) {
    view.appendChild(el("div", { class: "empty" }, el("div", { class: "big" }, "No summary table in this report")));
    return;
  }

  // KPI hero cards
  view.appendChild(sectionLabel("Key metrics"));
  view.appendChild(kpiCards(models));

  // charts
  view.appendChild(sectionLabel("Throughput & efficiency", "higher is better"));
  const panels = el("div", { class: "panels" });
  panels.appendChild(barPanel("Decode throughput", "tok/s", models, "gen_tps_mean", "g"));
  panels.appendChild(barPanel("Memory bandwidth utilization", "% of theoretical", models, "bw_util_mean", "g", { ceiling: 100 }));
  panels.appendChild(barPanel("Throughput per GB", "tok/s per GiB", models, "toks_per_gb_mean", "c"));
  panels.appendChild(barPanel("Time to first token", "ms · lower better", models, "ttft_ms_mean", "a", { lowerBetter: true }));
  view.appendChild(panels);

  // table
  view.appendChild(sectionLabel("Full summary", "click a header to sort"));
  view.appendChild(dataTable(models, color));

  // raw
  view.appendChild(sectionLabel("Raw report"));
  view.appendChild(el("div", { class: "raw" }, el("details", {}, [
    el("summary", {}, `${report.path}`),
    el("pre", {}, report.raw || ""),
  ])));

  requestAnimationFrame(() => animateBars(view));
}

function chip(k, v) { return el("span", { class: "chip" }, [`${k} `, el("b", {}, v || "—")]); }
function hwCell(k, v) { return el("div", { class: "hw-cell" }, [el("div", { class: "k" }, k), el("div", { class: "v" }, v)]); }
function sectionLabel(title, note) {
  return el("div", { class: "section-label" }, [
    el("h2", {}, title), el("span", { class: "rule" }),
    note ? el("span", { class: "note" }, note) : null,
  ]);
}

/* KPI cards */
function best(models, key, dir = 1) {
  let out = null;
  for (const m of models) {
    const v = m[key];
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (!out || (dir > 0 ? v > out.v : v < out.v)) out = { v, model: m.model };
  }
  return out;
}
function kpiCard(k, best, unit, opts = {}) {
  const c = opts.color || "var(--phosphor)";
  const glow = opts.glow || "rgba(166,255,63,0.18)";
  return el("div", { class: "kpi reveal", style: `--kc:${glow};--kc-text:${c}` }, [
    el("div", { class: "kpi-k" }, k),
    el("div", { class: "kpi-v" }, [best ? fmt(best.v, opts.d ?? 1) : "—", el("span", { class: "u" }, unit)]),
    el("div", { class: "kpi-sub" }, best ? (opts.subPrefix || "") + best.model : "no data"),
  ]);
}
function kpiCards(models) {
  const grid = el("div", { class: "kpi-grid" });
  grid.appendChild(kpiCard("Peak decode", best(models, "gen_tps_mean", 1), "tok/s", { d: 1 }));
  const bw = best(models, "bw_util_mean", 1);
  grid.appendChild(kpiCard("Max BW utilization", bw, "%", {
    d: 1, color: bw && bw.v > 100 ? "var(--amber)" : "var(--phosphor)",
    glow: bw && bw.v > 100 ? "rgba(255,176,32,0.18)" : "rgba(166,255,63,0.18)",
    subPrefix: bw && bw.v > 100 ? "sparse · " : "",
  }));
  grid.appendChild(kpiCard("Best per-GB", best(models, "toks_per_gb_mean", 1), "t/s·GiB", { d: 2, color: "var(--cyan)", glow: "rgba(69,200,240,0.18)" }));
  grid.appendChild(kpiCard("Fastest TTFT", best(models, "ttft_ms_mean", -1), "ms", { d: 0, color: "var(--amber)", glow: "rgba(255,176,32,0.16)" }));
  return grid;
}

/* bar panel */
function barPanel(title, unit, models, key, cls, opts = {}) {
  const vals = models.map((m) => m[key]).filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
  let max = vals.length ? Math.max(...vals) : 1;
  if (opts.ceiling) max = Math.max(max, opts.ceiling);
  max = max || 1;

  const rows = el("div", { class: "bars" });
  for (const m of models) {
    const v = m[key];
    const pct = v === null || v === undefined || Number.isNaN(v) ? 0 : Math.max(0, (v / max) * 100);
    let fillCls = cls;
    if (key === "bw_util_mean" && v > 100) fillCls = "a"; // sparse/MoE over the dense ceiling
    const track = el("div", { class: "bar-track" }, [
      el("div", { class: `bar-fill ${fillCls}`, "data-pct": pct.toFixed(2) }),
    ]);
    if (opts.ceiling && opts.ceiling < max) {
      track.appendChild(el("div", {
        class: "bar-ceiling", "data-label": opts.ceiling + "%",
        style: `left:${(opts.ceiling / max) * 100}%`,
      }));
    }
    rows.appendChild(el("div", { class: "bar-row" }, [
      el("div", { class: "bl", title: m.model }, m.model),
      track,
      el("div", { class: "bv" }, fmt(v, opts.d ?? (key === "toks_per_gb_mean" ? 2 : 1))),
    ]));
  }
  return el("div", { class: "panel reveal" }, [
    el("div", { class: "panel-head" }, [el("span", { class: "t" }, title), el("span", { class: "u" }, unit)]),
    rows,
  ]);
}

function animateBars(root) {
  root.querySelectorAll(".bar-fill, .gf").forEach((f) => {
    const pct = f.getAttribute("data-pct");
    if (pct != null) requestAnimationFrame(() => (f.style.width = pct + "%"));
  });
}

/* data table */
const COLUMNS = [
  { key: "model", label: "Model", num: false },
  { key: "ok_total", label: "OK", num: false },
  { key: "gen_tps_mean", label: "Gen tok/s", d: 2 },
  { key: "gen_tps_p90", label: "p90", d: 2 },
  { key: "gen_tps_stdev", label: "σ", d: 2 },
  { key: "prompt_tps_mean", label: "Prompt tok/s", d: 1 },
  { key: "ttft_ms_mean", label: "TTFT ms", d: 1 },
  { key: "eff_bw_mean", label: "Eff BW", d: 1 },
  { key: "bw_util_mean", label: "BW util %", d: 1 },
  { key: "toks_per_gb_mean", label: "Tok/s/GB", d: 2 },
  { key: "wall_s_mean", label: "Wall s", d: 2 },
];

function dataTable(models, color) {
  const rows = [...models].sort((a, b) => {
    const { key, dir } = state.sort;
    const av = a[key], bv = b[key];
    if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
    return String(av).localeCompare(String(bv)) * dir;
  });

  const thead = el("tr", {}, COLUMNS.map((c) => {
    const sorted = state.sort.key === c.key;
    return el("th", {
      class: sorted ? "sorted" : "",
      onclick: () => {
        state.sort = { key: c.key, dir: state.sort.key === c.key ? -state.sort.dir : (c.num === false ? 1 : -1) };
        if (state.lastReport) renderRun(state.lastReport);
      },
    }, [c.label, sorted ? el("span", { class: "arrow" }, state.sort.dir < 0 ? " ▼" : " ▲") : ""]);
  }));

  const tbody = el("tbody");
  for (const m of rows) {
    const tds = COLUMNS.map((c) => {
      if (c.key === "model") {
        return el("td", {}, el("div", { class: "cell-model" }, [
          el("span", { class: "swatch", style: `background:${color}` }), m.model,
        ]));
      }
      if (c.key === "ok_total") return el("td", { class: "dim" }, m.ok_total || "—");
      const v = m[c.key];
      let klass = "";
      if (c.key === "bw_util_mean" && typeof v === "number") klass = v > 100 ? "warn" : "good";
      return el("td", { class: klass }, fmt(v, c.d));
    });
    tbody.appendChild(el("tr", {}, tds));
  }
  return el("div", { class: "table-wrap reveal" }, el("table", { class: "data" }, [el("thead", {}, thead), tbody]));
}

/* ---------------------------------------------------------------- compare */
async function renderCompare() {
  const view = document.getElementById("view-compare");
  view.innerHTML = '<div class="empty"><div><div class="spinner"></div>collecting latest runs…</div></div>';
  let data;
  try { data = await getJSON("/api/compare"); }
  catch (e) { view.innerHTML = `<div class="empty"><div class="big">Compare failed</div><div>${e.message}</div></div>`; return; }

  const reports = (data.reports || []).filter((r) => (r.models || []).length);
  view.innerHTML = "";
  if (reports.length < 2) {
    view.appendChild(el("div", { class: "empty" }, el("div", {}, [
      el("div", { class: "big" }, "Need at least two machines to compare"),
      "Benchmark on another machine, then reload.",
    ])));
    return;
  }

  view.appendChild(el("div", { class: "runhead reveal" }, [
    el("h1", {}, ["Cross-machine ", el("span", { class: "cursor" }, "compare")]),
    el("div", { class: "chips" }, [el("span", { class: "chip accent" }, `${reports.length} machines · latest run each`)]),
  ]));

  const legend = el("div", { class: "legend reveal" });
  for (const r of reports) legend.appendChild(el("span", { class: "item" }, [
    el("span", { class: "sw", style: `background:${colorFor(r.machine_label)}` }), r.machine_label,
  ]));
  view.appendChild(sectionLabel("Machines"));
  view.appendChild(legend);

  view.appendChild(sectionLabel("Decode throughput", "tok/s · higher is better"));
  view.appendChild(groupedPanel(reports, "gen_tps_mean", 1));
  view.appendChild(sectionLabel("Memory bandwidth utilization", "% of theoretical · >100% = sparse/MoE"));
  view.appendChild(groupedPanel(reports, "bw_util_mean", 1));

  requestAnimationFrame(() => animateBars(view));
}

function groupedPanel(reports, key, d) {
  const models = [...new Set(reports.flatMap((r) => r.models.map((m) => m.model)))].sort();
  let max = 0;
  for (const r of reports) for (const m of r.models) {
    const v = m[key]; if (typeof v === "number" && v > max) max = v;
  }
  max = max || 1;

  const panel = el("div", { class: "panel reveal" });
  for (const model of models) {
    const set = el("div", { class: "gbar-set" });
    for (const r of reports) {
      const m = r.models.find((x) => x.model === model);
      const v = m ? m[key] : null;
      const pct = typeof v === "number" ? (v / max) * 100 : 0;
      set.appendChild(el("div", { class: "gbar" }, [
        el("div", { class: "gt" }, el("div", {
          class: "gf", "data-pct": pct.toFixed(2),
          style: `background:linear-gradient(90deg, ${colorFor(r.machine_label)}88, ${colorFor(r.machine_label)})`,
        })),
        el("div", { class: "gv" }, fmt(v, d)),
      ]));
    }
    panel.appendChild(el("div", { class: "gbar-row" }, [el("div", { class: "gl" }, model), set]));
  }
  return panel;
}

/* ------------------------------------------------------------------- wire */
function switchView(v) {
  state.view = v;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.view === v));
  document.getElementById("view-run").classList.toggle("is-active", v === "run");
  document.getElementById("view-compare").classList.toggle("is-active", v === "compare");
  if (v === "compare") renderCompare();
}

document.getElementById("tabs").addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (tab) switchView(tab.dataset.view);
});
document.getElementById("refresh").addEventListener("click", () => loadRuns().catch(showFatal));

document.getElementById("sort-toggle").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-sort]");
  if (!btn || btn.dataset.sort === state.sidebarSort) return;
  state.sidebarSort = btn.dataset.sort;
  document.querySelectorAll("#sort-toggle button").forEach((b) =>
    b.classList.toggle("on", b.dataset.sort === state.sidebarSort));
  renderSidebar();
});

function showFatal(e) { setStatus("is-error", "error: " + e.message); }

/* ----------------------------------------------- live "analysis running" */
let wasRunning = false;
async function pollStatus() {
  let s;
  try { s = await getJSON("/api/status"); } catch { return; }
  // A crashed run could leave a stale heartbeat; treat >3 min old as finished.
  const stale = s.updated && (Date.now() - Date.parse(s.updated)) > 180000;
  const running = !!s.running && !stale;
  const banner = document.getElementById("run-banner");

  if (running) {
    const cur = s.current || "…";
    const phase = s.phase === "warmup" ? ' <span class="ph">(warmup)</span>' : "";
    document.getElementById("rb-text").innerHTML = `Analysis running · <b>${cur}</b>${phase}`;
    const done = s.completed || 0, total = s.total || 0;
    document.getElementById("rb-count").textContent = total ? `${done} / ${total} models` : "";
    document.getElementById("rb-progress-fill").style.width = total ? `${(done / total) * 100}%` : "8%";
    banner.hidden = false;
    setStatus("is-live", `running · ${cur}`);
  } else {
    banner.hidden = true;
  }

  // When a run just finished, refresh the list so the new report shows up.
  if (wasRunning && !running) {
    setStatus("is-live", "run complete — refreshing…");
    loadRuns().catch(() => {});
  }
  wasRunning = running;
}

loadRuns().catch(showFatal);
pollStatus();
setInterval(pollStatus, 4000);
