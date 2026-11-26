// Small hand-written SVG charts (no chart library needed).
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v - 1e-9) return m * p;
  return 10 * p;
}

// series: [{name, color, points:[[x,y]], band:[[x,lo,hi]], dash, kind:"line"|"bars"}]
function lineChart(host, series, opts = {}) {
  host.replaceChildren();
  const W = 860, H = opts.height || 290, L = 44, R = 12, T = 28, B = 34;
  const xs = series.flatMap((s) => (s.points || s.band || []).map((p) => p[0]));
  const ys = series.flatMap((s) => [...(s.points || []).map((p) => p[1]), ...(s.band || []).map((p) => p[2])]);
  const x0 = opts.xmin ?? Math.min(...xs), x1 = opts.xmax ?? Math.max(...xs);
  const y1 = niceMax(opts.ymax ?? Math.max(1, ...ys));
  const sx = (x) => L + ((x - x0) / Math.max(1e-9, x1 - x0)) * (W - L - R);
  const sy = (y) => T + (1 - y / y1) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "chart" });
  for (let i = 0; i <= 5; i++) {
    const y = (y1 * i) / 5;
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: sy(y), y2: sy(y), class: "grid" }));
    svg.append(svgEl("text", { x: L - 6, y: sy(y) + 4, class: "axis", "text-anchor": "end" }, +y.toFixed(1)));
  }
  const ticks = opts.xticks || [];
  for (const [x, label] of ticks) {
    svg.append(svgEl("text", { x: sx(x), y: H - 12, class: "axis", "text-anchor": "middle" }, label));
  }
  if (opts.ylabel) svg.append(svgEl("text", { x: 4, y: 12, class: "axis" }, opts.ylabel));
  for (const s of series) {
    if (s.band) {
      const top = s.band.map((p) => `${sx(p[0])},${sy(p[2])}`);
      const bot = s.band.slice().reverse().map((p) => `${sx(p[0])},${sy(p[1])}`);
      svg.append(svgEl("polygon", { points: [...top, ...bot].join(" "), fill: s.color, opacity: 0.18 }));
    }
    if (s.kind === "bars" && s.points) {
      const bw = Math.max(1, ((W - L - R) / s.points.length) * 0.8);
      for (const [x, y] of s.points) {
        svg.append(svgEl("rect", { x: sx(x) - bw / 2, y: sy(y), width: bw, height: Math.max(0, sy(0) - sy(y)), fill: s.color, opacity: 0.55 }));
      }
    } else if (s.points) {
      svg.append(svgEl("polyline", {
        points: s.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" "), fill: "none", stroke: s.color,
        "stroke-width": s.width || 2.2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round",
      }));
    }
  }
  host.append(svg);
  const legend = document.createElement("div");
  legend.className = "legend";
  for (const s of series) {
    if (!s.name) continue;
    const item = document.createElement("span");
    const sw = document.createElement("i");
    sw.style.background = s.color;
    if (s.dash) sw.className = "dash";
    item.append(sw, s.name);
    legend.append(item);
  }
  host.append(legend);
}

// rows: [{label, value, sd}]
function barChart(host, rows, opts = {}) {
  host.replaceChildren();
  const W = 860, rowH = 24, L = 190, R = 70, H = rows.length * rowH + 10;
  const vmax = niceMax(Math.max(1e-6, ...rows.map((r) => r.value)));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "bar chart" });
  rows.forEach((r, i) => {
    const y = 5 + i * rowH;
    const w = Math.max(0, (r.value / vmax) * (W - L - R));
    svg.append(svgEl("text", { x: L - 8, y: y + 15, class: "axis", "text-anchor": "end" }, r.label));
    svg.append(svgEl("rect", { x: L, y: y + 3, width: w, height: rowH - 8, rx: 3, fill: opts.color || "var(--accent)" }));
    svg.append(svgEl("text", { x: L + w + 6, y: y + 15, class: "axis" }, r.text ?? r.value.toFixed(2)));
  });
  host.append(svg);
}
