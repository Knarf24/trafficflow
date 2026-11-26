// Runs all the Python (the trafficflow package) in a background thread so the page stays smooth.
// Messages in:  {id, type: "call", name, args} | {id, type: "vision-init"} | {id, type: "frame"|"photo", ...}
// Messages out: {id, ok} | {id, error} | {type: "status", text} | {type: "progress", ...}
importScripts("pyodide/pyodide.js");

const base = new URL("./", self.location.href).href;
let py = null;
let api = null;
let counter = null;
const PY_FILES = ["__init__.py", "sim.py", "forecast.py", "planner.py", "api.py", "vision.py"];

async function fetchText(path) {
  const r = await fetch(base + path);
  if (!r.ok) throw new Error("Could not load " + path);
  return r.text();
}

async function init() {
  self.postMessage({ type: "status", text: "Starting Python…" });
  py = await loadPyodide({ indexURL: base + "pyodide/" });
  py.FS.mkdirTree("/home/pyodide/trafficflow");
  for (const f of PY_FILES) py.FS.writeFile("/home/pyodide/trafficflow/" + f, await fetchText("trafficflow/" + f));
  self.postMessage({ type: "status", text: "Loading the practice data and the trained models…" });
  const [csv, models] = await Promise.all([fetchText("data/synthetic_drive_through.csv"), fetchText("data/models.json")]);
  py.runPython("from trafficflow import api");
  api = py.globals.get("api");
  const res = JSON.parse(api.call("load", JSON.stringify({ csv, models })));
  if (res.error) throw new Error(res.error);
  return res.ok;
}

const ready = init();
ready.then(
  (info) => self.postMessage({ type: "ready", info }),
  (e) => self.postMessage({ type: "fatal", text: String((e && e.message) || e) })
);

async function visionInit() {
  if (counter) return true;
  self.postMessage({ type: "status", scope: "vision", text: "Loading OpenCV and NumPy (about 13 MB, once)…" });
  await py.loadPackage(["numpy", "opencv-python"]);
  self.postMessage({ type: "status", scope: "vision", text: "Loading the vehicle detector…" });
  py.FS.mkdirTree("/home/pyodide/models");
  const r = await fetch(base + "models/object_detection_nanodet_2022nov.onnx");
  if (!r.ok) throw new Error("Could not load the detector model");
  py.FS.writeFile("/home/pyodide/models/object_detection_nanodet_2022nov.onnx", new Uint8Array(await r.arrayBuffer()));
  py.runPython("from trafficflow.vision import Counter\n_counter = Counter('/home/pyodide/models/object_detection_nanodet_2022nov.onnx')\n_counter.detector()");
  counter = py.globals.get("_counter");
  return true;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    await ready;
    let out;
    if (m.type === "call") {
      const res = JSON.parse(api.call(m.name, JSON.stringify(m.args || {})));
      if (res.error) throw new Error(res.error);
      out = res.ok;
    } else if (m.type === "vision-init") {
      out = await visionInit();
    } else if (m.type === "configure") {
      out = JSON.parse(counter.configure(JSON.stringify(m.settings)));
    } else if (m.type === "reset") {
      counter.reset();
      out = true;
    } else if (m.type === "frame") {
      out = JSON.parse(counter.process_rgba(new Uint8Array(m.buffer), m.width, m.height, m.t));
    } else if (m.type === "photo") {
      out = JSON.parse(counter.photo_rgba(new Uint8Array(m.buffer), m.width, m.height));
    } else if (m.type === "summary") {
      out = JSON.parse(counter.summary(m.duration));
    }
    self.postMessage({ id: m.id, ok: out });
  } catch (err) {
    self.postMessage({ id: m.id, error: String((err && err.message) || err) });
  }
};
