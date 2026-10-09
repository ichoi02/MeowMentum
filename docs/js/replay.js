/* 3D replay of hardware drops and simulation rollouts.
   The pose is rebuilt exactly as in the MuJoCo model (model/cat.xml):
   front body -> Rx(q_r1) -> Ry(q_p) -> Rx(q_r2) -> rear body -> Ry(q_t) -> tail,
   with the front-body orientation taken from the logged IMU quaternion. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const DATA = window.PAGE_DATA;
const MM = window.MM || (window.MM = {});
const CONDS = MM.CONDS;
const G = 9.81;
const T_EVAL = Math.sqrt(2 * 2.5 / G);           // whole-cat score instant: 2.5 m of fall
const RANGE = { roll: 7.28, pitch: 1.57, tail: 1.9199 };
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (s) => document.querySelector(s);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const stageEl = $("#stage3d");
if (!DATA || !DATA.trials || !stageEl || !CONDS) {
  if (MM.replayFallback) MM.replayFallback("Replay data could not be loaded.");
  throw new Error("replay: missing data");
}

/* ---------------- data ---------------- */
const hw = DATA.trials;
const sims = DATA.sim || [];
const condKey = (roll, pitch) => (CONDS.find((c) => c.roll === roll && c.pitch === pitch) || {}).key;
const condByKey = (k) => CONDS.find((c) => c.key === k);
function trialsFor(morph, cond) {
  return hw
    .filter((t) => t.morph === morph && t.roll === cond.roll && t.pitch === cond.pitch)
    .sort((a, b) => (a.date + String(a.rep).padStart(2, "0")).localeCompare(b.date + String(b.rep).padStart(2, "0")));
}
function representative(list) {
  // the drop whose score sits closest to its condition's median: typical, not cherry-picked
  const scored = list.filter((t) => t.included && t.score != null);
  if (!scored.length) return list[0];
  const s = scored.map((t) => t.score).sort((a, b) => a - b);
  const med = s[Math.floor((s.length - 1) / 2)];
  return scored.reduce((best, t) => (Math.abs(t.score - med) < Math.abs(best.score - med) ? t : best), scored[0]);
}

const state = { source: "hw", morph: "tail", cond: "p45", id: null, t: 0, playing: !reduceMotion, speed: 0.25, hold: 0, rewind: 0 };
let series = null;      // { t[], q[], j[], tilt[], impact_t? }
let qs = [];            // THREE.Quaternion per sample (heading-normalised)

/* ---------------- renderer ---------------- */
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
} catch (e) {
  if (MM.replayFallback) MM.replayFallback("Your browser could not start WebGL, which the 3D replay needs.");
  throw e;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stageEl.prepend(renderer.domElement);
renderer.domElement.setAttribute("role", "img");
renderer.domElement.setAttribute("aria-label", "3D view of the robot. Drag sideways to orbit.");

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 20);
camera.position.set(0.68, 0.35, 1.16);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enableZoom = false;
controls.enablePan = false;
controls.rotateSpeed = 0.7;
controls.target.set(0, 0, 0);
renderer.domElement.style.touchAction = "pan-y";   // vertical swipes keep scrolling the page
controls.addEventListener("start", () => { $("#hint").style.opacity = "0"; });

const hemi = new THREE.HemisphereLight(0xffffff, 0x7d838c, 0.55);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(0.5, 2.2, 0.7);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
// the shadow camera must cover the whole ground disc, or the uncovered part renders as shadow
Object.assign(key.shadow.camera, { left: -1.3, right: 1.3, top: 1.3, bottom: -1.3, near: 0.5, far: 4 });
key.shadow.bias = -0.0004;
scene.add(key);
const rim = new THREE.DirectionalLight(0xcfe0ff, 0.8);
rim.position.set(-1.2, 0.4, -1.0);
scene.add(rim);

// Ground below, so "down" always reads. Contact is irrelevant: the replay hangs at the origin.
const GROUND_Y = -0.34;
const ground = new THREE.Mesh(new THREE.CircleGeometry(1.2, 72), new THREE.ShadowMaterial({ opacity: 0.18 }));
ground.rotation.x = -Math.PI / 2;
ground.position.y = GROUND_Y;
ground.receiveShadow = true;
scene.add(ground);
const ringMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.55 });
const rings = new THREE.Group();
[0.16, 0.32, 0.48].forEach((r) => {
  const pts = [];
  for (let i = 0; i <= 96; i++) { const a = (i / 96) * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r)); }
  rings.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
});
rings.position.y = GROUND_Y + 0.0005;
scene.add(rings);
const plumb = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -0.16, 0), new THREE.Vector3(0, GROUND_Y, 0)]),
  new THREE.LineDashedMaterial({ dashSize: 0.012, gapSize: 0.01, transparent: true, opacity: 0.5 })
);
plumb.computeLineDistances();
scene.add(plumb);

/* ---------------- robot ---------------- */
const mats = {
  shell: new THREE.MeshStandardMaterial({ roughness: 0.42, metalness: 0.12 }),
  metal: new THREE.MeshStandardMaterial({ color: 0xb8bec6, roughness: 0.3, metalness: 0.8 }),
  housing: new THREE.MeshStandardMaterial({ color: 0x8f969f, roughness: 0.38, metalness: 0.6 }),
  foot: new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.0 }),
  roll: new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.1, emissiveIntensity: 0.3 }),
  pitch: new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.1, emissiveIntensity: 0.3 }),
  tail: new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.1, emissiveIntensity: 0.3 }),
};
const ghostMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.3, depthWrite: false });

function shellGeometry(dir) {
  // lathe profile: flat inner face, cylindrical side, domed outer end. dir = -1 dome at -x, +1 dome at +x
  const R = 0.056, inner = 0.048, side = 0.045, dome = 0.042, pts = [];
  pts.push(new THREE.Vector2(0.0001, -inner));
  pts.push(new THREE.Vector2(R - 0.004, -inner));
  pts.push(new THREE.Vector2(R, -inner + 0.004));
  pts.push(new THREE.Vector2(R, side));
  for (let i = 1; i <= 14; i++) {
    const a = (i / 14) * (Math.PI / 2);
    pts.push(new THREE.Vector2(R * Math.cos(a) + 0.0001, side + dome * Math.sin(a)));
  }
  const g = new THREE.LatheGeometry(pts, 48);
  g.rotateZ(dir < 0 ? Math.PI / 2 : -Math.PI / 2); // lathe axis y -> x, dome toward dir
  g.computeVertexNormals();
  return g;
}
const GEO = {
  front: shellGeometry(-1),
  rear: shellGeometry(1),
  shaft: new THREE.CylinderGeometry(0.013, 0.013, 0.052, 20).rotateZ(Math.PI / 2),
  housing: new THREE.BoxGeometry(0.046, 0.042, 0.05),
  ringX: new THREE.TorusGeometry(0.026, 0.0062, 12, 48).rotateY(Math.PI / 2),
  ringY: new THREE.TorusGeometry(0.033, 0.0062, 12, 48).rotateX(Math.PI / 2),
  ringT: new THREE.TorusGeometry(0.022, 0.0058, 12, 48).rotateX(Math.PI / 2),
  foot: new THREE.SphereGeometry(0.016, 20, 14).scale(1, 1, 0.75),
  rod: new THREE.CylinderGeometry(0.0055, 0.0055, 0.2, 14).rotateZ(-Math.PI / 2).translate(0.1, 0, 0),
  tip: new THREE.CylinderGeometry(0.012, 0.012, 0.036, 18).rotateZ(-Math.PI / 2).translate(0.182, 0, 0),
};
const EDGES = new Map();
function edges(geo) {
  if (!EDGES.has(geo)) EDGES.set(geo, new THREE.EdgesGeometry(geo, 35));
  return EDGES.get(geo);
}

function part(geo, mat, parent, pos, ghost) {
  const m = ghost ? new THREE.LineSegments(edges(geo), ghostMat) : new THREE.Mesh(geo, mat);
  if (pos) m.position.set(pos[0], pos[1], pos[2]);
  if (!ghost) { m.castShadow = true; m.receiveShadow = true; }
  parent.add(m);
  return m;
}
function buildRobot(ghost) {
  const front = new THREE.Group();
  part(GEO.front, mats.shell, front, [0, 0, 0], ghost);
  if (!ghost) {
    part(GEO.foot, mats.foot, front, [-0.034, 0.03, -0.052]);
    part(GEO.foot, mats.foot, front, [-0.034, -0.03, -0.052]);
  }
  const s1 = new THREE.Group(); s1.position.set(0.1, 0, 0); front.add(s1);          // rot1 about x
  if (!ghost) {
    part(GEO.shaft, mats.metal, s1, [-0.026, 0, 0]);
    part(GEO.ringX, mats.roll, s1, [-0.036, 0, 0]);
  }
  const s2 = new THREE.Group(); s1.add(s2);                                           // pitch about y
  if (!ghost) {
    part(GEO.housing, mats.housing, s2, [0, 0, 0]);
    part(GEO.ringY, mats.pitch, s2, [0, 0, 0]);
    part(GEO.shaft, mats.metal, s2, [0.026, 0, 0]);
  }
  const rear = new THREE.Group(); rear.position.set(0.1, 0, 0); s2.add(rear);       // rot2 about x
  part(GEO.rear, mats.shell, rear, [0, 0, 0], ghost);
  if (!ghost) {
    part(GEO.ringX, mats.roll, rear, [-0.064, 0, 0]);
    part(GEO.foot, mats.foot, rear, [0.034, 0.03, -0.052]);
    part(GEO.foot, mats.foot, rear, [0.034, -0.03, -0.052]);
  }
  const tail = new THREE.Group(); tail.position.set(0.05, 0, 0); rear.add(tail);    // tail about y
  if (!ghost) part(GEO.ringT, mats.tail, tail, [0.03, 0, 0]);
  part(GEO.rod, mats.metal, tail, [0, 0, 0], ghost);
  part(GEO.tip, mats.metal, tail, [0, 0, 0], ghost);
  return { front, s1, s2, rear, tail };
}

// world: data is z-up; three.js is y-up
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);
const robot = buildRobot(false);
const ghost = buildRobot(true);
world.add(robot.front);
world.add(ghost.front);

const SPINE = new THREE.Vector3(0.1, 0, 0);
const _v = new THREE.Vector3();
function pose(r, q, j) {
  r.front.quaternion.copy(q);
  _v.copy(SPINE).applyQuaternion(q).negate();   // keep the spine joint at the origin
  r.front.position.copy(_v);
  r.s1.rotation.set(j[0], 0, 0);
  r.s2.rotation.set(0, j[1], 0);
  r.rear.rotation.set(j[2], 0, 0);
  r.tail.rotation.set(0, j[3], 0);
}

function isDark() {
  return (document.documentElement.getAttribute("data-theme") ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
}
function applyTheme() {
  const dark = isDark();
  mats.shell.color.set(dark ? 0x30353d : 0x2a2f37);
  mats.foot.color.set(css("--foot") || "#d8463b");
  for (const k of ["roll", "pitch", "tail"]) {
    const c = css(k === "tail" ? "--tailj" : "--" + k);
    mats[k].color.set(c);
    mats[k].emissive.set(c);
  }
  ghostMat.color.set(css("--muted") || "#5f6876");
  ghostMat.opacity = dark ? 0.35 : 0.3;
  ringMat.color.set(css("--line-strong") || "#c9d0d8");
  plumb.material.color.set(css("--muted") || "#5f6876");
  ground.material.opacity = dark ? 0.4 : 0.16;
  hemi.intensity = dark ? 0.5 : 0.55;
  drawMini();
  requestRender();
}
window.addEventListener("mm-theme", applyTheme);

/* ---------------- series selection ---------------- */
function headingFix(q0) {
  // heading is unobservable from gravity; turn each replay so the body axis starts along +x
  const ax = new THREE.Vector3(1, 0, 0).applyQuaternion(q0);
  if (Math.hypot(ax.x, ax.y) < 0.2) return new THREE.Quaternion();
  return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.atan2(ax.y, ax.x));
}
// Hardware replays end exactly at impact (the accelerometer peak in the blanket, itself a
// logged sample); simulated rollouts have no landing and run the full episode.
const endT = () => (state.source === "hw" && series.impact_t != null ? series.impact_t : series.t[series.t.length - 1]);

function load() {
  const cond = condByKey(state.cond);
  if (state.source === "sim") {
    series = sims.find((s) => s.morph === state.morph && s.roll === cond.roll && s.pitch === cond.pitch) || null;
  } else {
    const list = trialsFor(state.morph, cond);
    if (!list.find((t) => t.id === state.id)) state.id = (representative(list) || {}).id;
    series = list.find((t) => t.id === state.id) || null;
  }
  if (!series) return;
  const raw = series.q.map((w) => new THREE.Quaternion(w[1], w[2], w[3], w[0]).normalize());
  const fix = headingFix(raw[0]);
  qs = raw.map((q) => fix.clone().multiply(q));
  for (let i = 1; i < qs.length; i++) if (qs[i].dot(qs[i - 1]) < 0) qs[i].set(-qs[i].x, -qs[i].y, -qs[i].z, -qs[i].w);
  pose(ghost, qs[0], series.j[0]);
  robot.tail.visible = ghost.tail.visible = state.morph === "tail";
  state.t = 0;
  state.hold = 0;
  state.rewind = 0;
  syncUI();
  drawMini();
  frame(0);
  requestRender();
}

/* ---------------- UI ---------------- */
const elConds = $("#r-conds"), elTrials = $("#r-trials"), elTrialsWrap = $("#r-trials-wrap");
CONDS.forEach((c) => {
  const b = document.createElement("button");
  b.type = "button";
  b.dataset.k = c.key;
  b.innerHTML = `Roll ${c.roll}°<span>pitch ${c.pitch}°</span>`;
  b.addEventListener("click", () => { state.cond = c.key; state.id = null; load(); });
  elConds.appendChild(b);
});
function seg(id, key, onChange) {
  document.querySelectorAll(`#${id} button`).forEach((b) => {
    b.addEventListener("click", () => {
      state[key] = key === "speed" ? parseFloat(b.dataset.v) : b.dataset.v;
      document.querySelectorAll(`#${id} button`).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      if (onChange) onChange();
    });
  });
}
// The module loads lazily, so honour any choice made in the panel before it arrived.
for (const [id, key] of [["r-source", "source"], ["r-morph", "morph"], ["r-speed", "speed"]]) {
  const b = document.querySelector(`#${id} button[aria-pressed="true"]`);
  if (b) state[key] = key === "speed" ? parseFloat(b.dataset.v) : b.dataset.v;
}
seg("r-source", "source", () => { state.id = null; load(); });
seg("r-morph", "morph", () => { state.id = null; load(); });
seg("r-speed", "speed");
if (!sims.length) document.querySelector('#r-source button[data-v="sim"]').disabled = true;

const HW_NOTE = "Hardware log: front-body orientation from the IMU and four joint encoders at 50 Hz; the rear segment is rebuilt from the joint chain. Heading is not observable from gravity, so each replay is turned to start facing the same way. Each replay ends at impact, the peak deceleration in the blanket, 0.70–0.85 s after free-fall detection; the samples logged after it are not shown. Distance is the ballistic estimate ½gt² from free-fall detection, so near impact it can exceed the nominal 2.8 m drop. Scores are taken at 2.5 m of fall, before impact.";
const SIM_NOTE = "Simulation: one MuJoCo rollout of this configuration's student policy with nominal physics and clean sensing, released at rest with a straight spine and run for the full one-second episode, further than the 2.8 m hardware drop. Simulated rollouts are not scored.";

function syncUI() {
  const cond = condByKey(state.cond);
  const hwMode = state.source === "hw";
  document.querySelectorAll("#r-source button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === state.source)));
  document.querySelectorAll("#r-morph button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === state.morph)));
  elConds.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.k === state.cond)));
  elTrialsWrap.hidden = !hwMode;
  elTrials.innerHTML = "";
  if (hwMode) {
    trialsFor(state.morph, cond).forEach((t, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = String(i + 1);
      b.title = `${t.id}${t.included ? `, whole-cat score ${t.score.toFixed(3)}` : ", excluded: impact before 2.5 m of fall"}`;
      b.setAttribute("aria-label", `Drop ${i + 1}, ${b.title}`);
      b.setAttribute("aria-pressed", String(t.id === state.id));
      if (!t.included) b.classList.add("excluded");
      b.addEventListener("click", () => { state.id = t.id; load(); });
      elTrials.appendChild(b);
    });
  }
  const morphTxt = state.morph === "tail" ? "with tail" : "no tail";
  const condTxt = `roll ${cond.roll}°, pitch ${cond.pitch}°`;
  $("#hud-left").textContent = hwMode ? `Hardware log · ${morphTxt} · ${condTxt}` : `Simulation · ${morphTxt} student · ${condTxt}`;
  $("#ro-score").textContent = !hwMode ? "not scored" : series && series.included && series.score != null ? series.score.toFixed(3) : "excluded";
  $("#ro-policy-lbl").textContent = hwMode ? "Policy that flew, identified by replaying the log" : "Policy";
  $("#ro-policy").textContent = hwMode
    ? (series && series.policy === "tail" ? "with-tail policy" : "no-tail policy")
    : (state.morph === "tail" ? "with-tail" : "no-tail") + " student";
  $("#jr-tail").classList.toggle("off", state.morph !== "tail");
  let note = hwMode ? HW_NOTE : SIM_NOTE;
  if (!hwMode && state.morph === "notail" && cond.roll === 180) note += " From the 180° roll releases the simulated no-tail student is not feet-down (mean tilt above 45°) at 2.5 m of fall; from the pitched ones it does not get there within the one-second episode.";
  $("#r-note").textContent = note;
  $("#r-chart-note").textContent = hwMode
    ? "Line: this drop, up to impact. Band: hardware mean ± SD for this release, first 0.64 s. Dashed: 2.5 m scoring point."
    : "Line: the simulated rollout. Band: hardware drops of this configuration and release (mean ± SD, first 0.64 s)" +
      (state.morph === "notail" && cond.roll === 180 && cond.pitch > 0 ? ", all flown with the with-tail policy" : "") +
      ". Dashed: 2.5 m of fall, where hardware drops are scored.";
  $("#hud-left").dataset.short = (hwMode ? "Log" : "Sim") + " · " + (state.morph === "tail" ? "tail" : "no tail");
  drawMarks();
}

function drawMarks() {
  const marks = $("#r-marks");
  marks.innerHTML = "";
  if (!series) return;
  const T = endT();
  const W = marks.clientWidth || 300;
  const list = [[Math.sqrt(2 / G), "1 m", ""], [Math.sqrt(4 / G), "2 m", ""], [T_EVAL, "2.5 m", ""]];
  // place labels by priority so the scoring point wins collisions; ticks always show
  const prio = { "2.5 m": 1, "1 m": 2, "2 m": 3 };
  const placed = [];
  list.filter((m) => m[0] <= T).sort((a, b) => prio[a[1]] - prio[b[1]]).forEach(([tt, label, cls]) => {
    const px = (tt / T) * W, half = label.length * 3.2 + 3, x = (tt / T) * 100;
    const show = placed.every((p) => Math.abs(p.px - px) > p.half + half + 4);
    if (show) placed.push({ px, half });
    marks.insertAdjacentHTML("beforeend", `${show ? `<span class="${cls}" style="left:${x}%">${label}</span>` : ""}<i style="left:${x}%"></i>`);
  });
}

const playBtn = $("#r-play");
function setPlayIcon() {
  // SVG elements have no .hidden property, so toggle the attribute itself
  playBtn.querySelector(".i-pause").toggleAttribute("hidden", !state.playing);
  playBtn.querySelector(".i-play").toggleAttribute("hidden", state.playing);
  playBtn.setAttribute("aria-label", state.playing ? "Pause" : "Play");
}
playBtn.addEventListener("click", () => {
  if (!state.playing && series && state.t >= endT()) state.t = 0;
  state.playing = !state.playing;
  state.hold = 0;
  state.rewind = 0;
  setPlayIcon();
  requestRender();
});
const scrub = $("#r-scrub");
scrub.addEventListener("input", () => {
  if (!series) return;
  state.playing = false;
  state.rewind = 0;
  setPlayIcon();
  state.t = (scrub.value / 1000) * endT();
  frame(0);
  requestRender();
});

/* ---------------- mini chart ---------------- */
const mini = $("#r-chart");
let miniHead = null, miniDot = null, mx = null, my = null;
function drawMini() {
  mini.innerHTML = "";
  if (!series) return;
  const svg = MM.svg;
  const MW = Math.round(Math.max(240, mini.clientWidth || 300)), MH = Math.round(Math.min(MW * 0.36, 130)) + 20;
  const mm = { l: 30, r: 6, t: 8, b: 18 };
  const T = endT();
  mx = (v) => mm.l + (v / T) * (MW - mm.l - mm.r);
  my = (v) => MH - mm.b - (Math.min(225, Math.max(0, v)) / 225) * (MH - mm.t - mm.b);
  const s = svg("svg", { viewBox: `0 0 ${MW} ${MH}`, width: MW, height: MH, role: "img", "aria-label": "Mean body tilt of this drop over time" }, mini);
  const ax = svg("g", { class: "axis" }, s);
  [0, 90, 180].forEach((v) => {
    svg("line", { x1: mm.l, x2: MW - mm.r, y1: my(v), y2: my(v), class: "gridline" }, ax);
    svg("text", { x: mm.l - 5, y: my(v) + 3, "text-anchor": "end", text: v + "°", style: "font-size:10px" }, ax);
  });
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
    if (v > T + 1e-6) return;
    svg("text", { x: mx(v), y: MH - 4, "text-anchor": "middle", text: v + " s", style: "font-size:10px" }, ax);
  });
  const cond = condByKey(state.cond);
  const ref = (DATA.tilt || []).find((c) => c.morph === state.morph && c.roll === cond.roll && c.pitch === cond.pitch);
  const col = `var(--c-${cond.key})`;
  if (ref) {
    let d = "";
    ref.t.forEach((t, i) => { d += (i ? "L" : "M") + mx(t).toFixed(1) + "," + my(ref.mean[i] + ref.sd[i]).toFixed(1); });
    for (let i = ref.t.length - 1; i >= 0; i--) d += "L" + mx(ref.t[i]).toFixed(1) + "," + my(ref.mean[i] - ref.sd[i]).toFixed(1);
    svg("path", { d: d + "Z", style: `fill:${col};fill-opacity:.16;stroke:none` }, s);
  }
  if (T_EVAL <= T) svg("line", { x1: mx(T_EVAL), x2: mx(T_EVAL), y1: mm.t, y2: MH - mm.b, class: "guide" }, s);
  let line = "";
  series.t.forEach((t, i) => {
    if (t <= T + 1e-9) line += (line ? "L" : "M") + mx(t).toFixed(1) + "," + my(series.tilt[i]).toFixed(1);
  });
  svg("path", { d: line, style: `fill:none;stroke:${col};stroke-width:2;stroke-linejoin:round` }, s);
  if (state.source === "hw" && series.impact_t != null) {
    svg("text", { x: MW - mm.r, y: mm.t + 2, "text-anchor": "end", text: "impact", style: "font-size:10px;fill:var(--pitch)" }, ax);
  }
  miniHead = svg("line", { x1: mm.l, x2: mm.l, y1: mm.t, y2: MH - mm.b, style: "stroke:var(--ink);stroke-width:1" }, s);
  miniDot = svg("circle", { r: 3.6, cx: mm.l, cy: my(series.tilt[0]), style: "fill:var(--ink);stroke:var(--surface);stroke-width:1.5" }, s);
  frame(0);
}

/* ---------------- frame ---------------- */
const _q = new THREE.Quaternion();
const jv = [0, 0, 0, 0];
function sample(t) {
  const ts = series.t, n = ts.length;
  if (t <= ts[0]) return { i: 0, u: 0 };
  if (t >= ts[n - 1]) return { i: n - 2, u: 1 };
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ts[mid] <= t) lo = mid; else hi = mid; }
  return { i: lo, u: (t - ts[lo]) / (ts[lo + 1] - ts[lo]) };
}
function setBar(id, v, range) {
  const el = $("#jb-" + id), f = Math.max(-1, Math.min(1, v / range)) * 50;
  el.style.left = (f < 0 ? 50 + f : 50) + "%";
  el.style.width = Math.abs(f) + "%";
}
const REWIND = 0.45;   // s, eased return to the release pose between loops
function frame(dt) {
  if (!series) return;
  const T = endT();
  if (state.playing && dt > 0) {
    if (state.t >= T) {
      state.hold += dt;
      if (state.hold > 0.9) state.rewind += dt;
      if (state.rewind >= REWIND) { state.t = 0; state.hold = 0; state.rewind = 0; }
    } else {
      state.t = Math.min(T, state.t + dt * state.speed);
    }
  }
  const { i, u } = sample(state.t);
  _q.slerpQuaternions(qs[i], qs[i + 1], u);
  for (let k = 0; k < 4; k++) jv[k] = series.j[i][k] + (series.j[i + 1][k] - series.j[i][k]) * u;
  if (state.rewind > 0) {
    const e = Math.min(1, state.rewind / REWIND), w = e * e * (3 - 2 * e);
    _q.slerp(qs[0], w);
    for (let k = 0; k < 4; k++) jv[k] += (series.j[0][k] - jv[k]) * w;
  }
  if (state.morph !== "tail") jv[3] = 0;
  pose(robot, _q, jv);
  const tilt = series.tilt[i] + (series.tilt[i + 1] - series.tilt[i]) * u;
  const atImpact = state.source === "hw" && series.impact_t != null && state.t >= T - 1e-9;
  const d = 0.5 * G * state.t ** 2;
  $("#hud-right").innerHTML = atImpact
    ? `<b>${state.t.toFixed(3)} s</b> · impact`
    : `<b>${state.t.toFixed(3)} s</b> · ${d.toFixed(2)} m fallen`;
  $("#ro-tilt").textContent = tilt.toFixed(1) + "°";
  $("#jv-roll").textContent = MM.fmt(jv[0], 2) + " rad";
  $("#jv-pitch").textContent = MM.fmt(jv[1], 2) + " rad";
  $("#jv-tail").textContent = state.morph === "tail" ? MM.fmt(jv[3], 2) + " rad" : "–";
  setBar("roll", jv[0], RANGE.roll);
  setBar("pitch", jv[1], RANGE.pitch);
  setBar("tail", state.morph === "tail" ? jv[3] : 0, RANGE.tail);
  if (document.activeElement !== scrub) scrub.value = String(Math.round((state.t / T) * 1000));
  scrub.setAttribute("aria-valuetext", atImpact ? `${state.t.toFixed(2)} s, impact` : `${state.t.toFixed(2)} s, ${d.toFixed(2)} m fallen`);
  if (miniHead) {
    const x = mx(state.t);
    miniHead.setAttribute("x1", x); miniHead.setAttribute("x2", x);
    miniDot.setAttribute("cx", x); miniDot.setAttribute("cy", my(tilt));
  }
}

/* ---------------- loop ---------------- */
let visible = false, last = 0, rafId = 0;
function requestRender() { if (!rafId && visible) rafId = requestAnimationFrame(loop); }
function loop(now) {
  rafId = 0;
  const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
  last = now;
  if (state.playing) frame(dt);
  const moved = controls.update();
  renderer.render(scene, camera);
  if (visible && (state.playing || moved)) rafId = requestAnimationFrame(loop);
  else last = 0;
}
controls.addEventListener("change", requestRender);
let lastMiniW = 0;
function resize() {
  const w = stageEl.clientWidth, h = stageEl.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.fov = w / h < 0.95 ? 36 : 30;   // keep the whole robot in frame on narrow screens
  camera.updateProjectionMatrix();
  if (Math.abs(mini.clientWidth - lastMiniW) > 16) { lastMiniW = mini.clientWidth; drawMini(); drawMarks(); }
  requestRender();
}
new ResizeObserver(resize).observe(stageEl);
new ResizeObserver(resize).observe(mini);
new IntersectionObserver((es) => {
  es.forEach((e) => {
    visible = e.isIntersecting;
    if (visible) { last = 0; requestRender(); }
  });
}, { threshold: 0.05 }).observe(stageEl);
document.addEventListener("visibilitychange", () => { if (!document.hidden) { last = 0; requestRender(); } });

/* ---------------- public API (Fig. 5 points link here) ---------------- */
MM.replay = {
  select(opts, scroll, focus) {
    if (opts.id) {
      const t = hw.find((x) => x.id === opts.id);
      if (!t) return;
      state.source = "hw"; state.morph = t.morph; state.cond = condKey(t.roll, t.pitch); state.id = t.id;
    }
    state.playing = !reduceMotion;
    setPlayIcon();
    load();
    if (scroll) document.getElementById("replay-app").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    if (focus) {
      const b = document.querySelector('#r-trials button[aria-pressed="true"]');
      if (b) b.focus({ preventScroll: true });
    }
    requestRender();
  },
};

applyTheme();
setPlayIcon();
load();
resize();
// A deep link below the replay may have scrolled before this module grew the panel.
{
  const h = location.hash && document.querySelector(location.hash);
  if (h && (stageEl.compareDocumentPosition(h) & Node.DOCUMENT_POSITION_FOLLOWING)) h.scrollIntoView();
}
