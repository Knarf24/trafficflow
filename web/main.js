// The page: inputs, tables, charts and the car counter's video loop.
// All the modelling (simulation, forecasting, planning, detection) is Python in worker.js.
"use strict";

const $ = (id) => document.getElementById(id);
const S = { info: null, site: "drive_through", task: "day_ahead", day: null, sim: null, replay: null,
  futureDemand: null, counter: null, models: null, evalData: null };

// ------------------------------------------------------------------ worker bridge
const worker = new Worker("worker.js");
let nextId = 1;
const waiting = new Map();
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") {
    if (m.scope === "vision") $("camStatus").textContent = m.text; else $("loadText").textContent = m.text;
    return;
  }
  if (m.type === "ready") return onReady(m.info);
  if (m.type === "fatal") {
    $("loading").classList.add("failed");
    $("loadText").textContent = "Sorry, Python could not start in this browser: " + m.text;
    $("pyStatus").textContent = "Python failed";
    return;
  }
  const p = waiting.get(m.id);
  if (!p) return;
  waiting.delete(m.id);
  if (m.error) p.reject(new Error(m.error)); else p.resolve(m.ok);
};
function ask(msg, transfer) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    worker.postMessage({ ...msg, id }, transfer || []);
  });
}
const py = (name, args) => ask({ type: "call", name, args });

// ------------------------------------------------------------------ helpers
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : String(kid));
  return e;
}
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 3000);
}
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : (Math.round(v * 10 ** d) / 10 ** d).toFixed(d));
const hhmm = (minute) => `${String(Math.floor(minute / 60)).padStart(2, "0")}:${String(Math.round(minute % 60)).padStart(2, "0")}`;
const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
function setPressed(seg, attr, value) {
  for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset[attr] === value));
}
function table(host, head, rows) {
  host.replaceChildren(el("thead", {}, el("tr", {}, head.map((h) => el("th", { text: h })))),
    el("tbody", {}, rows.map((r) => el("tr", { class: r.cls || null }, r.cells.map((c) => (c && c.nodeType ? el("td", {}, c) : el("td", { text: c })))))));
}
function busy(btn, on, label) {
  btn.disabled = on;
  if (on) { btn.dataset.label = btn.textContent; btn.textContent = label || "Working…"; }
  else if (btn.dataset.label) btn.textContent = btn.dataset.label;
}

// ------------------------------------------------------------------ tabs
for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => showTab(b.dataset.tab));
function showTab(name) {
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === name));
  for (const p of document.querySelectorAll("[data-panel]")) p.hidden = p.dataset.panel !== name;
  try { localStorage.setItem("tf.tab", name); } catch { /* private mode */ }
}

// ================================================================== SIMULATE
const DT_FIELDS = [
  ["order_takers", "Order takers", 1, 3, 1], ["kitchen_crew", "Kitchen crew", 0, 8, 1],
  ["stack_spaces", "Spaces before ordering", 0, 30, 1], ["pickup_spaces", "Spaces before window", 0, 15, 1],
  ["exit_spaces", "Spaces before road", 0, 10, 1], ["order_min", "Minutes to take an order", 0.1, 5, 0.05],
  ["prep_min", "Minutes to make an order", 0.2, 10, 0.1], ["handoff_min", "Minutes at the window", 0.1, 5, 0.05],
  ["balk_prob", "Chance a car leaves if full", 0, 1, 0.05],
];
const PK_FIELDS = [
  ["gate_lanes", "Pay gates open", 0, 4, 1], ["pay_min", "Minutes to pay", 0.05, 3, 0.05],
  ["aisle_spaces", "Spaces in exit aisle", 0, 60, 1], ["exit_spaces", "Spaces before road", 0, 10, 1],
];
const SCEN = {
  drive_through: {
    current: { order_takers: 1, kitchen_crew: 3, stack_spaces: 8, pickup_spaces: 4, exit_spaces: 2, order_min: 0.6, prep_min: 1.5, handoff_min: 0.35, balk_prob: 0.4 },
    proposed: { order_takers: 2, kitchen_crew: 4, stack_spaces: 8, pickup_spaces: 4, exit_spaces: 2, order_min: 0.6, prep_min: 1.5, handoff_min: 0.35, balk_prob: 0.4 },
  },
  parking_exit: {
    current: { gate_lanes: 1, pay_min: 0.4, aisle_spaces: 12, exit_spaces: 2 },
    proposed: { gate_lanes: 2, pay_min: 0.4, aisle_spaces: 12, exit_spaces: 2 },
  },
};
const PRESETS = {
  drive_through: [
    ["lunch", "Weekday lunch rush, 11:00-14:00", 660, [40, 52, 64, 76, 82, 80, 72, 62, 52, 44, 38, 34]],
    ["dinner", "Friday dinner with a promotion, 17:00-20:00", 1020, [50, 62, 80, 96, 104, 100, 92, 80, 66, 56, 48, 42]],
    ["quiet", "Quiet afternoon, 2 hours", 840, [30, 30, 30, 30, 30, 30, 30, 30]],
    ["flat", "Steady rate for 2 hours (type it)", 0, null],
  ],
  parking_exit: [
    ["event", "Event lets out: about 310 cars in 75 minutes", 1260, [300, 500, 300, 100, 40, 0, 0, 0]],
    ["closing", "Store closing: 120 cars/hour for an hour", 1260, [120, 120, 120, 120, 0, 0]],
    ["flat", "Steady rate for 2 hours (type it)", 0, null],
  ],
};

function buildFields() {
  const fields = S.site === "drive_through" ? DT_FIELDS : PK_FIELDS;
  for (const which of ["current", "proposed"]) {
    const host = $(which === "current" ? "curFields" : "proFields");
    host.replaceChildren(...fields.map(([key, label, min, max, step]) =>
      el("label", {}, label, el("input", {
        type: "number", min, max, step, value: SCEN[S.site][which][key], "data-key": key,
        oninput: (e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) SCEN[S.site][which][key] = v; },
      }))));
  }
  const sel = $("demand");
  const keep = sel.value;
  sel.replaceChildren(...PRESETS[S.site].map(([k, label]) => el("option", { value: k, text: label })));
  if (S.counter && S.site === "drive_through") sel.append(el("option", { value: "counter", text: `From the car counter: ${fmt(S.counter.rate_per_hour, 0)} cars/hour for 1 hour` }));
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
  $("flatWrap").hidden = sel.value !== "flat";
}
$("demand").addEventListener("change", () => { $("flatWrap").hidden = $("demand").value !== "flat"; });
$("siteSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  S.site = b.dataset.site;
  setPressed($("siteSeg"), "site", S.site);
  buildFields();
});
$("copyCur").addEventListener("click", () => {
  SCEN[S.site].proposed = { ...SCEN[S.site].current };
  buildFields();
  toast("Proposed now matches current. Change one thing and run.");
});

function demandRates() {
  const v = $("demand").value;
  if (v === "flat") return { start: 0, rates: Array(8).fill(Math.max(0, +$("flatRate").value || 0)) };
  if (v === "counter") return { start: 0, rates: Array(4).fill(S.counter.rate_per_hour) };
  const p = PRESETS[S.site].find((x) => x[0] === v);
  return { start: p[2], rates: p[3] };
}

async function updateGap() {
  const flow = Math.max(0, +$("roadFlow").value || 0);
  const g = await py("gap", { vehicles_per_hour: flow });
  S.gap = g.gap_wait_min;
  $("gapNote").textContent = `With ${flow} vehicles/hour on the road, a driver waits about ${fmt(g.gap_wait_min * 60, 0)} s on average for a safe gap (5-second gap needed).`;
}
$("roadFlow").addEventListener("change", updateGap);

const METRICS = [
  ["mean_wait", "Average time on site (minutes)", -1, 1],
  ["p95_wait", "Slowest 5% of cars: at least (minutes)", -1, 1],
  ["lost", "Cars that drove away", -1, 1],
  ["street_car_minutes", "Car-minutes queued on the street", -1, 0],
  ["peak_on_site", "Most cars on site at once", 0, 0],
  ["served_in_hours", "Cars served within the period", 1, 0],
  ["clear_after_close", "Minutes to clear the line afterwards", -1, 1],
  ["unfinished", "Cars still stuck when the run stopped", -1, 0],
];

function params(which) {
  return { site: S.site, ...SCEN[S.site][which], gap_wait_min: S.gap ?? 0.05 };
}

$("runSim").addEventListener("click", async () => {
  const btn = $("runSim");
  const { start, rates } = demandRates();
  if (!rates.some((r) => r > 0) && !(+$("initQueue").value > 0)) return toast("Add some demand first.");
  busy(btn, true, "Simulating…");
  try {
    await updateGap();
    const reps = +$("reps").value;
    const args = { current: params("current"), proposed: params("proposed"), rates, reps, seed: 1, initial_queue: +$("initQueue").value || 0 };
    const t0 = performance.now();
    const out = await py("compare", args);
    S.sim = { out, start, rates, args };
    $("simStatus").textContent = `${reps * 2} runs in ${fmt((performance.now() - t0) / 1000, 1)} s`;
    showSim();
    await loadReplay();
  } catch (err) {
    toast("Simulation failed: " + err.message);
  } finally {
    busy(btn, false);
  }
});

function showSim() {
  const { out, start } = S.sim;
  $("simOut").hidden = false;
  const s = out.summary;
  const rows = METRICS.map(([k, label, dir, d]) => {
    const m = s[k];
    const allZero = m.current.mean === 0 && m.proposed.mean === 0;
    let verdict = "–";
    if (!allZero && dir !== 0) {
      const share = dir < 0 ? m.better_share : m.higher_share;
      verdict = `${Math.round(share * out.reps)} of ${out.reps}`;
    }
    const range = (b) => `${fmt(b.mean, d)}  (${fmt(b.lo, d)} to ${fmt(b.hi, d)})`;
    const diff = m.diff.mean;
    const good = dir === 0 || Math.abs(diff) < 1e-9 ? "" : (diff * dir > 0 ? "good" : "bad");
    return { cells: [label, range(m.current), range(m.proposed), el("span", { class: good, text: (diff > 0 ? "+" : "") + fmt(diff, d) }), verdict] };
  });
  table($("simTable"), ["", "Current: average (90% of runs)", "Proposed", "Change", "Proposed better in"], rows);
  const w = s.mean_wait;
  $("simSummary").textContent = `Over ${out.reps} runs with about ${fmt(out.arrived.mean, 0)} cars each, the proposed setup changes the average time on site by ${fmt(w.diff.mean, 1)} minutes (95% confidence ±${fmt(w.diff.ci, 1)}). These are model results from the inputs above, not measurements.`;
  const pts = (c) => c.map((p) => [start + p[0], p[2]]);
  const band = (c) => c.map((p) => [start + p[0], p[1], p[3]]);
  const ticks = [];
  const end = start + out.horizon;
  for (let m = Math.ceil(start / 30) * 30; m <= end; m += 30) ticks.push([m, hhmm(m)]);
  lineChart($("simChart"), [
    { name: "Current", color: "var(--bad)", points: pts(out.curves.current), band: band(out.curves.current) },
    { name: "Proposed", color: "var(--teal)", points: pts(out.curves.proposed), band: band(out.curves.proposed) },
  ], { xticks: ticks, ylabel: "cars in line (incl. street)", label: "Cars in line over time" });
}

// ------------------------------------------------------------------ replay of one run
async function loadReplay() {
  const which = document.querySelector("#replaySeg [aria-pressed=true]").dataset.which;
  const { args, rates, start } = S.sim;
  const res = await py("replay", { params: args[which], rates, seed: 1, initial_queue: args.initial_queue });
  const p = { ...(S.site === "drive_through" ? {} : {}), ...args[which] };
  S.replay = { res, start, p, which, end: Math.max(15 * rates.length, ...res.cars.map((c) => (c.leave ?? 0))) };
  $("replayT").max = S.replay.end;
  $("replayT").value = Math.min(+$("replayT").value, S.replay.end);
  drawLane();
}
$("replaySeg").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if (!b || !S.sim) return;
  setPressed($("replaySeg"), "which", b.dataset.which);
  await loadReplay();
});
$("replayT").addEventListener("input", drawLane);
let playing = null;
$("playBtn").addEventListener("click", () => {
  if (playing) { cancelAnimationFrame(playing); playing = null; $("playBtn").textContent = "▶ Play"; return; }
  $("playBtn").textContent = "❚❚ Pause";
  let last = performance.now();
  const tick = (now) => {
    const t = +$("replayT").value + ((now - last) / 1000) * 2; // 2 simulated minutes per second
    last = now;
    $("replayT").value = Math.min(t, S.replay.end);
    drawLane();
    if (t >= S.replay.end) { playing = null; $("playBtn").textContent = "▶ Play"; return; }
    playing = requestAnimationFrame(tick);
  };
  playing = requestAnimationFrame(tick);
});

function drawLane() {
  if (!S.replay) return;
  const t = +$("replayT").value;
  const { res, start, p } = S.replay;
  $("replayClock").textContent = hhmm(start + t);
  const dt = S.site === "drive_through";
  const st = dt
    ? [["Waiting to order", p.stack_spaces, "Order point", p.order_takers], ["Waiting for window", p.pickup_spaces, "Window", 1], ["Waiting for a gap", p.exit_spaces, "Road", 1]]
    : [["Exit aisle", p.aisle_spaces, "Pay gate", p.gate_lanes], ["Waiting for a gap", p.exit_spaces, "Road", 1]];
  const queues = st.map(() => []);
  const servers = st.map(() => []);
  const overflow = [];
  let gone = 0, lost = 0;
  for (const c of res.cars) {
    if (c.arrive > t) continue;
    if (c.balked) { lost++; continue; }
    if (c.leave != null && c.leave <= t) { gone++; continue; }
    const waited = t - c.arrive;
    if (c.into[0] == null || c.into[0] > t) { overflow.push(waited); continue; }
    for (let s = st.length - 1; s >= 0; s--) {
      if (c.into[s] != null && c.into[s] <= t) {
        if (c.start[s] == null || c.start[s] > t) queues[s].push([c.into[s], waited]);
        else servers[s].push(waited);
        break;
      }
    }
  }
  const car = (w) => el("span", { class: "car", style: `--h:${Math.max(0, 140 - w * 14)}`, title: `waiting ${fmt(w, 1)} min` });
  const parts = [];
  parts.push(el("div", { class: "seg-box street" }, el("div", { class: "lbl", text: dt ? "On the street" : "Still in parking spaces" }),
    el("div", { class: "cars" }, overflow.slice(0, 24).map(car), overflow.length > 24 ? el("b", { text: `+${overflow.length - 24}` }) : null)));
  st.forEach(([qLabel, cap, sLabel, n], s) => {
    const q = queues[s].sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    const slots = [];
    for (let i = 0; i < Math.max(cap, q.length); i++) slots.push(el("span", { class: "slot" }, q[i] != null ? car(q[i]) : null));
    parts.push(el("div", { class: "seg-box" }, el("div", { class: "lbl", text: `${qLabel} (${cap})` }), el("div", { class: "cars" }, slots)));
    const serv = [];
    for (let i = 0; i < Math.max(n, servers[s].length); i++) serv.push(el("span", { class: "slot server" }, servers[s][i] != null ? car(servers[s][i]) : null));
    parts.push(el("div", { class: "seg-box srv" }, el("div", { class: "lbl", text: sLabel }), el("div", { class: "cars" }, serv)));
  });
  $("lane").replaceChildren(...parts);
  $("laneNote").textContent = `${gone} cars have left · ${lost} drove away · colour shows how long each car has waited (green short, red long). A car that finished but has no room ahead stays in its spot and blocks it.`;
}

// ================================================================== FORECAST
const FEATURE_NAMES = {
  minute_of_day: "Time of day", weekday: "Day of week", weekend: "Weekend", holiday: "Holiday", promo_day: "Promotion day",
  school_day: "School day", temp_c: "Temperature", rain_mm: "Rain", tod_s1: "Daily cycle (sin 1)", tod_c1: "Daily cycle (cos 1)",
  tod_s2: "Daily cycle (sin 2)", tod_c2: "Daily cycle (cos 2)", tod_s3: "Daily cycle (sin 3)", tod_c3: "Daily cycle (cos 3)",
  lag1: "Cars in the last 15 min", lag2: "Cars 15-30 min ago", lag3: "Cars 30-45 min ago", lag4: "Cars 45-60 min ago",
  yesterday: "Same time yesterday", last_week: "Same time last week", last_week_next: "Next slot last week",
  today_vs_last_week: "Today so far vs last week", two_weeks: "Same time 2 weeks ago", mean4: "Average of last 4 weeks",
  last_week_smooth: "Last week, smoothed",
};
const MODEL_NAMES = { ridge: "Linear regression (Ridge)", random_forest: "Random forest", gradient_boosting: "Gradient boosting" };

function modelName(task, key) {
  return MODEL_NAMES[key] || "Simple guess: " + (S.models.baseline_names[task][key] || key);
}

function showScores() {
  const task = S.task;
  const info = S.models.tasks[task];
  const rows = Object.entries(info.test).sort((a, b) => a[1].mae - b[1].mae).map(([k, v]) => ({
    cls: k === info.chosen ? "chosen" : (MODEL_NAMES[k] ? "" : "base"),
    cells: [modelName(task, k) + (k === info.chosen ? "  ✓ chosen" : ""), fmt(info.validation[k]?.mae, 2), fmt(v.mae, 2), fmt(v.rmse, 2), (v.bias > 0 ? "+" : "") + fmt(v.bias, 2)],
  }));
  table($("scoreTable"), ["Model", "Validation error", "Test error (MAE, cars)", "Test RMSE", "Bias"], rows);
  const best = info.best_baseline;
  $("scoreNote").textContent = `The model was chosen on the validation weeks before looking at the test weeks. On the test weeks, ${MODEL_NAMES[info.chosen]} is off by ${fmt(info.test[info.chosen].mae, 2)} cars per 15 minutes on average (about ${fmt(info.test_mean_arrivals, 1)} cars arrive per slot), ${fmt(info.improvement_vs_best_baseline * 100, 0)}% less than the best simple guess (${S.models.baseline_names[task][best].toLowerCase()}). Its low-high range held the real count ${fmt(info.interval_coverage * 100, 0)}% of the time (aim: 80%). Remember: this is synthetic data, so these scores do not say how it would do at a real restaurant.`;
  barChart($("impChart"), info.importance.slice(0, 10).map((r) => ({ label: FEATURE_NAMES[r.feature] || r.feature, value: Math.max(0, r.mae_increase), text: `+${fmt(r.mae_increase, 3)} cars error` })));
  const groups = Object.entries(info.by_period);
  const cols = Object.keys(groups[0][1]).filter((k) => k !== "slots");
  table($("periodTable"), ["Slots", "Count", ...cols.map((c) => MODEL_NAMES[c] || S.models.baseline_names[task][c])],
    groups.map(([g, v]) => ({ cells: [g, String(v.slots), ...cols.map((c) => fmt(v[c], 2))] })));
}

$("taskSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  S.task = b.dataset.task;
  setPressed($("taskSeg"), "task", S.task);
  showScores();
  drawDay();
});

async function loadDay(di) {
  S.day = await py("day", { day: di });
  drawDay();
  const sel = $("n15Slot");
  sel.replaceChildren(...S.day.minute.map((m, j) => el("option", { value: j, text: `${hhmm(m)} to ${hhmm(m + 15)}` })));
  sel.value = 24; // 12:00
  runNext15();
}
$("testDay").addEventListener("change", () => loadDay(+$("testDay").value));

function drawDay() {
  if (!S.day) return;
  const d = S.day, f = d[S.task];
  const x = d.minute.map((m) => m + 7.5);
  const base = Object.entries(f.baselines).sort((a, b) => f.baseline_mae[a[0]] - f.baseline_mae[b[0]])[0];
  const ticks = [];
  for (let m = 360; m <= 1380; m += 120) ticks.push([m, hhmm(m)]);
  lineChart($("fcChart"), [
    { name: "What happened", color: "var(--muted)", kind: "bars", points: x.map((v, j) => [v, d.actual[j]]) },
    { name: `${MODEL_NAMES[S.models.tasks[S.task].chosen]} (range shaded)`, color: "var(--accent)", points: x.map((v, j) => [v, f.point[j]]), band: x.map((v, j) => [v, f.low[j], f.high[j]]) },
    { name: "Simple guess: " + S.models.baseline_names[S.task][base[0]].toLowerCase(), color: "var(--teal)", dash: "5 4", points: x.map((v, j) => [v, base[1][j]]) },
  ], { xmin: 360, xmax: 1380, xticks: ticks, ylabel: "cars per 15 min", label: "Forecast vs actual" });
  const flags = [d.promo_day ? "promotion day" : "", d.holiday ? "holiday" : "", d.rain_mm.some((r) => r > 0) ? "rain" : ""].filter(Boolean).join(", ");
  $("fcDayNote").textContent = `${DAYS[d.weekday]} ${d.date}${flags ? " (" + flags + ")" : ""}: ${d.actual.reduce((a, b) => a + b, 0)} cars. Model error ${fmt(f.mae, 2)} vs simple guess ${fmt(base[1] ? f.baseline_mae[base[0]] : 0, 2)} cars per slot.`;
}

async function runNext15() {
  if (!S.day) return;
  const raw = $("n15Last").value;
  const r = await py("next15", { day: +$("testDay").value, slot: +$("n15Slot").value, last15: raw === "" ? null : +raw });
  const j = +$("n15Slot").value;
  $("n15Out").replaceChildren(
    "Forecast for ", el("b", { text: `${hhmm(S.day.minute[j])}-${hhmm(S.day.minute[j] + 15)}` }), ": ",
    el("b", { class: "accent", text: `${fmt(r.point, 1)} cars` }), ` (likely ${fmt(Math.min(r.low, r.point), 0)} to ${fmt(Math.max(r.high, r.point), 0)}). `,
    el("span", { class: "small", text: `Using ${raw === "" ? r.lag1_actual + " cars in the last 15 minutes (from the data)" : raw + " cars in the last 15 minutes (your number)"}. What really happened: ${r.actual} cars.` }));
}
$("n15Slot").addEventListener("change", runNext15);
$("n15Last").addEventListener("change", runNext15);

// ------------------------------------------------------------------ new day + live weather
$("fdWeather").addEventListener("change", () => { $("cityWrap").hidden = $("fdWeather").value !== "live"; });

async function liveWeather(city) {
  const g = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`);
  if (!g.ok) throw new Error("place search failed");
  const place = (await g.json()).results?.[0];
  if (!place) throw new Error(`no place called "${city}"`);
  const w = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}&hourly=temperature_2m,precipitation&forecast_days=2&timezone=auto`);
  if (!w.ok) throw new Error("weather request failed");
  const data = await w.json();
  const times = data.hourly.time;
  const tomorrow = times[24].slice(0, 10);
  const temp = [], rain = [];
  for (let h = 6; h < 23; h++) {
    const i = times.indexOf(`${tomorrow}T${String(h).padStart(2, "0")}:00`);
    for (let q = 0; q < 4; q++) { temp.push(data.hourly.temperature_2m[i]); rain.push(data.hourly.precipitation[i] / 4); }
  }
  const weekday = (new Date(tomorrow + "T12:00:00").getDay() + 6) % 7;
  return { temp, rain, place: `${place.name}${place.country ? ", " + place.country : ""}`, date: tomorrow, weekday };
}

$("fdRun").addEventListener("click", async () => {
  const btn = $("fdRun");
  busy(btn, true, "Forecasting…");
  const args = { weekday: +$("fdWeekday").value, promo_day: +$("fdPromo").value, holiday: +$("fdHoliday").value };
  let note = "Typical weather for the season, dry.";
  try {
    const mode = $("fdWeather").value;
    if (mode === "rain") { args.rain_mm = Array(68).fill(1.0); note = "Rain all day (4 mm per hour)."; }
    if (mode === "live") {
      try {
        const w = await liveWeather($("fdCity").value.trim() || "Pittsburgh");
        args.temp_c = w.temp;
        args.rain_mm = w.rain;
        args.weekday = w.weekday;
        $("fdWeekday").value = w.weekday;
        const rainTotal = w.rain.reduce((a, b) => a + b, 0);
        note = `Live forecast for ${w.place} on ${w.date} (${DAYS[w.weekday]}): ${fmt(Math.min(...w.temp), 0)} to ${fmt(Math.max(...w.temp), 0)} °C, ${fmt(rainTotal, 1)} mm of rain between 06:00 and 23:00, applied to the practice site.`;
      } catch (err) {
        note = `Could not get live weather (${err.message}), so typical dry weather is used instead.`;
      }
    }
    const f = await py("future_day", args);
    S.futureDemand = { ...f, label: `New ${DAYS[args.weekday]} (forecast)` };
    const x = f.minute.map((m) => m + 7.5);
    const ticks = [];
    for (let m = 360; m <= 1380; m += 120) ticks.push([m, hhmm(m)]);
    $("fdChart").hidden = false;
    lineChart($("fdChart"), [{ name: "Forecast (range shaded)", color: "var(--accent)", points: x.map((v, j) => [v, f.point[j]]), band: x.map((v, j) => [v, f.low[j], f.high[j]]) }],
      { xmin: 360, xmax: 1380, xticks: ticks, ylabel: "cars per 15 min" });
    $("fdStatus").textContent = `${note} About ${fmt(f.point.reduce((a, b) => a + b, 0), 0)} cars expected.`;
    $("fdPlanRow").hidden = false;
    fillPlanDays();
  } catch (err) {
    toast("Forecast failed: " + err.message);
  } finally {
    busy(btn, false);
  }
});
$("fdToPlan").addEventListener("click", () => { $("plDay").value = "future"; showTab("plan"); });

// ================================================================== PLANNER
function fillPlanDays() {
  const sel = $("plDay");
  const keep = sel.value;
  const [a, b] = S.info.test_days;
  const opts = [];
  if (S.futureDemand) opts.push(el("option", { value: "future", text: S.futureDemand.label }));
  for (let di = a; di < b; di++) {
    const [date, wd] = S.info.dates[di];
    opts.push(el("option", { value: di, text: `${DAYS[wd]} ${date} (test day)` }));
  }
  sel.replaceChildren(...opts);
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
}

$("plRun").addEventListener("click", async () => {
  const btn = $("plRun");
  busy(btn, true, "Planning… (about 10 s)");
  try {
    const choice = $("plDay").value;
    let demand, actual = null, label;
    if (choice === "future") { demand = S.futureDemand.point; label = S.futureDemand.label; }
    else {
      const d = await py("day", { day: +choice });
      demand = d.day_ahead.point;
      actual = d.actual;
      label = `${DAYS[d.weekday]} ${d.date}`;
    }
    const costs = { wage_per_hour: +$("plWage").value, wait_value_per_min: +$("plWait").value, lost_car_value: +$("plLost").value, street_value_per_min: +$("plStreet").value };
    const t0 = performance.now();
    const r = await py("plan", { demand, actual, max_staff: +$("plMax").value, costs, reps: 6 });
    $("plStatus").textContent = `Planned in ${fmt((performance.now() - t0) / 1000, 1)} s`;
    $("plOut").hidden = false;
    const hours = [[6, 10.5], [10.5, 14], [14, 17], [17, 20.5], [20.5, 23]];
    table($("plTable"), ["Shift", "Time", "Order takers", "Kitchen crew", "Forecast cars"],
      r.plan.map(([t, c], k) => {
        const a = (hours[k][0] - 6) * 4, b = (hours[k][1] - 6) * 4;
        const cars = demand.slice(a, b).reduce((x, y) => x + y, 0);
        return { cells: [r.shifts[k], `${hhmm(hours[k][0] * 60)}-${hhmm(hours[k][1] * 60)}`, String(t), String(c), fmt(cars, 0)] };
      }));
    const P = r.check.plan, R = r.check.rota;
    table($("plCheck"), ["", "Staff-hours", "Wages", "Avg time on site", "Slowest 5%", "Cars lost", "Street car-min", "Total cost"],
      [["Recommended plan", P], ["Fixed rota (1 + 4 all day)", R]].map(([n, v]) => ({ cells: [n, fmt(v.staff_hours, 1), "$" + fmt(v.labour_cost, 0), fmt(v.mean_wait, 1) + " min", fmt(v.p95_wait, 1) + " min", fmt(v.lost, 1), fmt(v.street_car_minutes, 0), "$" + fmt(v.total_cost, 0)] })));
    $("plNote").textContent = actual
      ? `${label}: the plan was made from the ML forecast, then checked against what actually arrived that day (synthetic data), with fresh random service times.`
      : `${label}: checked against the forecast itself, because this day has not happened. Real staffing also needs breaks, minimum shift lengths and people's availability.`;
  } catch (err) {
    toast("Planning failed: " + err.message);
  } finally {
    busy(btn, false);
  }
});

function showEval() {
  const e = S.evalData;
  if (!e) return;
  const names = { fixed_rota: "Fixed rota, 1 + 4 all day", last_week_plan: "Planned from last week's counts", ml_plan: "Planned from the ML forecast", hindsight_plan: "Planned with hindsight (best case)" };
  table($("evalTable"), ["Plan", "Staff-hours", "Wages", "Avg time on site", "Cars lost", "Total cost per day"],
    Object.entries(names).map(([k, n]) => {
      const v = e.average_per_day[k];
      return { cls: k === "ml_plan" ? "chosen" : "", cells: [n, fmt(v.staff_hours, 1), "$" + fmt(v.labour_cost, 0), fmt(v.mean_wait, 2) + " min", fmt(v.lost, 1), "$" + fmt(v.total_cost, 0)] };
    }));
  const a = e.average_per_day;
  const saving = 1 - a.ml_plan.total_cost / a.fixed_rota.total_cost;
  const vsLw = a.ml_plan.total_cost - a.last_week_plan.total_cost;
  $("evalNote").textContent = `Averaged over the ${e.n_days} test days (synthetic data). Each plan is scored on the cars that actually arrived. Planning from a forecast cuts the total cost by about ${fmt(saving * 100, 0)}% compared with a flat rota, mostly through fewer staff-hours in quiet periods, with slightly longer waits. The ML forecast's planning cost is ${vsLw <= 0 ? "$" + fmt(-vsLw, 0) + " lower" : "$" + fmt(vsLw, 0) + " higher"} per day than planning from last week's counts (cheaper on ${e.ml_beats_last_week_days} of ${e.n_days} days), so its better accuracy has not yet turned into cheaper staffing. Hindsight shows about $${fmt(a.ml_plan.total_cost - a.hindsight_plan.total_cost, 0)} per day is still on the table.`;
}

// ================================================================== CAR COUNTER
const cam = { kind: null, stop: false, img: null, lastBoxes: [] };
const canvas = $("camCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const video = $("camVid");

async function ensureVision() {
  $("camStatus").textContent = "Loading the detector…";
  await ask({ type: "vision-init" });
  $("camStatus").textContent = "Detector ready.";
}
function settings() {
  const zone = +$("camZone").value;
  return { line: +$("camLine").value, direction: $("camDir").value, zone: [zone, 1.0] };
}
function drawOverlay(boxes) {
  const w = canvas.width, h = canvas.height;
  const s = settings();
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#ff4d4d";
  ctx.beginPath();
  if (s.direction === "vertical") { ctx.moveTo(s.line * w, s.zone[0] * h); ctx.lineTo(s.line * w, h); }
  else { ctx.moveTo(s.zone[0] * w, s.line * h); ctx.lineTo(w, s.line * h); }
  ctx.stroke();
  if (s.zone[0] > 0) {
    ctx.fillStyle = "rgba(0,0,0,.35)";
    if (s.direction === "vertical") ctx.fillRect(0, 0, w, s.zone[0] * h); else ctx.fillRect(0, 0, s.zone[0] * w, h);
  }
  ctx.font = "bold 14px system-ui";
  for (const b of boxes || []) {
    const [x, y, bw, bh] = b.box;
    ctx.strokeStyle = b.counted ? "#3fd6c6" : "#ffb547";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, bw, bh);
    const tag = `${b.kind}${b.id ? " #" + b.id : ""} ${Math.round(b.score * 100)}%`;
    ctx.fillStyle = "rgba(0,0,0,.6)";
    ctx.fillRect(x, y - 18, ctx.measureText(tag).width + 8, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, x + 4, y - 4);
  }
  ctx.restore();
}
function redraw() {
  if (cam.kind === "photo" && cam.img) ctx.drawImage(cam.img, 0, 0, canvas.width, canvas.height);
  else if (cam.kind === "video") ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  drawOverlay(cam.lastBoxes);
}
for (const id of ["camDir", "camLine", "camZone"]) $(id).addEventListener("input", redraw);

function sizeCanvas(w, h) {
  const scale = Math.min(1, 854 / w);
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
}
function loadVideo(src) {
  return new Promise((resolve, reject) => {
    video.onloadeddata = () => { sizeCanvas(video.videoWidth, video.videoHeight); resolve(); };
    video.onerror = () => reject(new Error("this browser cannot play that video"));
    video.src = src;
    video.load();
  });
}
function seek(t) {
  return new Promise((resolve) => {
    video.onseeked = () => resolve();
    video.currentTime = t;
  });
}
async function openVideo(src, name) {
  cam.kind = "video";
  cam.lastBoxes = [];
  $("camUse").hidden = true;
  $("camOut").textContent = "";
  try {
    await loadVideo(src);
    await seek(Math.min(1, video.duration / 2));
    redraw();
    $("camRun").disabled = false;
    $("camStatus").textContent = `${name}: ${fmt(video.duration, 0)} s. Set the red line across the cars' path, then count.`;
  } catch (err) {
    $("camStatus").textContent = "Could not open the video: " + err.message;
  }
}
$("camSample").addEventListener("click", () => {
  const webm = video.canPlayType('video/webm; codecs="vp9"');
  $("camDir").value = "vertical"; $("camLine").value = 0.5; $("camZone").value = 0.3;
  const path = webm ? "sample/intersection.webm" : "sample/intersection.mp4";
  $("camStatus").textContent = "Loading the sample clip…";
  // load it whole, so seeking frame by frame works on any web server
  fetch(path).then((r) => { if (!r.ok) throw new Error(r.status); return r.blob(); })
    .then((blob) => openVideo(URL.createObjectURL(blob), "Sample clip"))
    .catch((err) => { $("camStatus").textContent = "Could not load the sample clip: " + err.message; });
});
$("camVideo").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) openVideo(URL.createObjectURL(f), f.name);
});
$("camPhoto").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = async () => {
    cam.kind = "photo";
    cam.img = img;
    cam.lastBoxes = [];
    sizeCanvas(img.naturalWidth, img.naturalHeight);
    redraw();
    $("camRun").disabled = true;
    try {
      await ensureVision();
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const clean = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const r = await ask({ type: "photo", buffer: clean.data.buffer, width: canvas.width, height: canvas.height }, [clean.data.buffer]);
      cam.lastBoxes = r.boxes;
      redraw();
      S.photoCount = r.visible;
      $("camOut").replaceChildren("Vehicles found in the photo: ", el("b", { class: "accent", text: String(r.visible) }));
      $("camUse").hidden = false;
      for (const id of ["useArrivals", "useRoad", "useLast15"]) $(id).hidden = true;
      $("useQueue").hidden = false;
    } catch (err) {
      $("camStatus").textContent = "Detection failed: " + err.message;
    }
  };
  img.src = URL.createObjectURL(f);
});

$("camRun").addEventListener("click", async () => {
  if (cam.kind !== "video") return;
  $("camRun").disabled = true;
  $("camStop").disabled = false;
  cam.stop = false;
  try {
    await ensureVision();
    await ask({ type: "configure", settings: settings() });
    await ask({ type: "reset" });
    const step = 0.25;
    const dur = video.duration;
    let t = 0, count = 0;
    const t0 = performance.now();
    for (; t < dur && !cam.stop; t += step) {
      await seek(t);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const r = await ask({ type: "frame", buffer: img.data.buffer, width: canvas.width, height: canvas.height, t }, [img.data.buffer]);
      cam.lastBoxes = r.boxes;
      count = r.count;
      redraw();
      $("camOut").replaceChildren("Counted so far: ", el("b", { class: "accent", text: String(count) }), el("span", { class: "small", text: `  · ${fmt(t, 1)} of ${fmt(dur, 0)} s · ${fmt((performance.now() - t0) / 1000, 0)} s elapsed` }));
    }
    const sum = await ask({ type: "summary", duration: Math.min(t, dur) });
    S.counter = sum;
    $("camOut").replaceChildren("Counted ", el("b", { class: "accent", text: `${sum.count} vehicles` }), ` crossing the line in ${fmt(sum.duration_s, 0)} s, about `,
      el("b", { text: `${fmt(sum.rate_per_hour, 0)} per hour` }), el("span", { class: "small", text: " (a short clip makes this rate rough)." }));
    $("camUse").hidden = false;
    for (const id of ["useArrivals", "useRoad", "useLast15"]) $(id).hidden = false;
    $("useQueue").hidden = true;
    buildFields();
  } catch (err) {
    $("camStatus").textContent = "Counting failed: " + err.message;
  } finally {
    $("camRun").disabled = false;
    $("camStop").disabled = true;
  }
});
$("camStop").addEventListener("click", () => { cam.stop = true; });

$("useArrivals").addEventListener("click", () => {
  if (S.site !== "drive_through") { S.site = "drive_through"; setPressed($("siteSeg"), "site", S.site); }
  buildFields();
  $("demand").value = "counter";
  $("flatWrap").hidden = true;
  showTab("sim");
  toast(`Demand set to ${fmt(S.counter.rate_per_hour, 0)} cars/hour. Press Run comparison.`);
});
$("useRoad").addEventListener("click", () => {
  $("roadFlow").value = Math.round(S.counter.rate_per_hour);
  updateGap();
  showTab("sim");
  toast("Road traffic set from the count. Cars now wait longer for a gap when leaving.");
});
$("useLast15").addEventListener("click", () => {
  $("n15Last").value = Math.round(S.counter.rate_per_hour / 4);
  showTab("fc");
  runNext15();
  $("n15Out").scrollIntoView({ behavior: "smooth", block: "center" });
});
$("useQueue").addEventListener("click", () => {
  $("initQueue").value = S.photoCount;
  showTab("sim");
  toast(`${S.photoCount} cars set as already in line.`);
});

// ================================================================== ABOUT
function aboutText() {
  const m = S.models.tasks.day_ahead;
  $("aboutText").innerHTML = `
<h2>What TrafficFlow does</h2>
<p>Queues of cars form on business property: at a drive-through, a parking-lot exit, a pickup lane. When the waiting spaces run out, cars block each other and spill onto the public road. TrafficFlow helps a manager see where the queue forms, what the next hours will bring, and how many people to put on each shift.</p>
<h3>1. Car-by-car simulation</h3>
<p>Each car is followed individually: it arrives at random (a Poisson process whose rate changes every 15 minutes), waits in a fixed number of spaces, is served for a random time, and moves on only if there is room ahead. If there isn't, it stays put and blocks its spot ("blocking"). At a drive-through the kitchen crew prepares orders in parallel and a car waits at the window for its food; when the stack is full a car either drives away or waits on the street. Leaving the property needs a safe gap in road traffic, using a standard gap-acceptance formula. The current and proposed setups always get exactly the same customers, and the comparison is repeated so you see a range.</p>
<h3>2. Machine-learning forecast</h3>
<p>Two tasks: the number of cars in the next 15 minutes, and every 15-minute slot of tomorrow. Inputs are time of day, day of week, holidays, promotions, school days, temperature, rain and recent or past counts. Linear regression, a random forest and gradient boosting (scikit-learn) are compared with simple guesses on weeks they never saw; quantile gradient boosting gives a low and high estimate. The trained models are exported to JSON and run here in plain Python. Chosen for tomorrow's forecast: ${MODEL_NAMES[m.chosen]}.</p>
<h3>3. Staff planner</h3>
<p>For each shift, every allowed mix of order takers and kitchen crew is simulated against the forecast and the cheapest total (wages + waiting time + lost cars + street queue) is kept, shift by shift. It was tested on 14 held-out days against a flat rota, planning from last week, and hindsight.</p>
<h3>4. Car counter</h3>
<p>A pretrained NanoDet-Plus detector (OpenCV Model Zoo, Apache 2.0) finds cars, buses, trucks and motorcycles in video frames; a simple tracker counts each one once when it crosses the line. Counts can become the arrival rate, the road traffic past the exit, or the "last 15 minutes" input of the forecast. A photo count can become the cars already in line.</p>
<h3>Honest limits</h3>
<ul>
<li><b>The data is synthetic.</b> The 14 weeks of drive-through arrivals were generated by a documented made-up process (rush hours, weekdays, rain, heat, promotions, a holiday, school starting, noise). The scores show the method works on held-out data; they say nothing about accuracy at a real restaurant.</li>
<li>Service times and site sizes are assumptions you can change, not measurements. Nothing here has been calibrated against a real site.</li>
<li>The weather inputs in the practice data are the actual (made-up) weather, so the forecast assumes a perfect weather forecast.</li>
<li>The detector was not trained or checked on this footage. Overhead views, night, rain and cars hidden behind others cause missed or double counts. A 30-second clip gives only a rough hourly rate.</li>
<li>The planner searches shift by shift (greedy), ignores breaks and minimum shift lengths, and assumes extra crew scale linearly.</li>
</ul>
<h3>Built from</h3>
<p>Ideas from GeeksforGeeks Python projects, rebuilt and combined: <a href="https://www.geeksforgeeks.org/python/opencv-python-program-vehicle-detection-video-frame/" target="_blank" rel="noopener">Vehicle Detection in a Video frame</a> and <a href="https://www.geeksforgeeks.org/python/scraping-weather-data-using-python-to-get-umbrella-reminder-on-email/" target="_blank" rel="noopener">Scraping Weather Data</a>, together with queue simulation and scikit-learn regression. Weather by <a href="https://open-meteo.com/" target="_blank" rel="noopener">Open-Meteo</a> (CC BY 4.0). Sample clip from Wikimedia Commons (CC BY 4.0). Python runs in the browser with <a href="https://pyodide.org/" target="_blank" rel="noopener">Pyodide</a>.</p>`;
}

// ================================================================== start
async function onReady(info) {
  S.info = info;
  try {
    const [models, ev] = await Promise.all([
      fetch("data/models.json").then((r) => r.json()),
      fetch("data/planner_eval.json").then((r) => r.json()),
    ]);
    S.models = models;
    S.evalData = ev;
  } catch (err) {
    $("loadText").textContent = "Could not load the model files: " + err.message;
    return;
  }
  $("loading").hidden = true;
  $("app").inert = false;
  $("app").classList.remove("warming");
  $("pyStatus").textContent = "Python ready";
  buildFields();
  await updateGap();
  const [a, b] = info.test_days;
  $("testDay").replaceChildren(...info.dates.slice(a, b).map(([date, wd, hol, promo], k) =>
    el("option", { value: a + k, text: `${DAYS[wd]} ${date}${promo ? " · promotion" : ""}${hol ? " · holiday" : ""}` })));
  $("testDay").value = a + 4;
  $("fdWeekday").replaceChildren(...DAYS.map((d, i) => el("option", { value: i, text: d })));
  $("fdWeekday").value = (info.dates[info.dates.length - 1][1] + 1) % 7;
  showScores();
  await loadDay(a + 4);
  fillPlanDays();
  showEval();
  aboutText();
  let tab = "sim";
  try { tab = localStorage.getItem("tf.tab") || "sim"; } catch { /* private mode */ }
  if (location.hash.length > 1) tab = location.hash.slice(1);
  if (document.querySelector(`[data-panel="${tab}"]`)) showTab(tab);
  document.documentElement.dataset.py = "ready";
  // Keeps Python saved on this device so the next visit starts without downloading it (see sw.js).
  if ("serviceWorker" in navigator && isSecureContext) navigator.serviceWorker.register("sw.js").catch(() => {});
}

// The page shows right away and can be read while Python starts in the background; the controls
// switch on once it is ready (about half a second on a repeat visit).
{
  let tab = "sim";
  try { tab = localStorage.getItem("tf.tab") || "sim"; } catch { /* private mode */ }
  if (location.hash.length > 1) tab = location.hash.slice(1);
  if (document.querySelector(`[data-panel="${tab}"]`)) showTab(tab);
  document.documentElement.dataset.ui = "ready";
}
