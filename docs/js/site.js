/* MeowMentum project page: everything except the 3D replay (js/replay.js). */
(function () {
  "use strict";

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var DATA = window.PAGE_DATA || null;
  var G = 9.81;
  var SVGNS = "http://www.w3.org/2000/svg";

  // Release conditions, in the order the paper uses.
  var CONDS = [
    { key: "r45", roll: 45, pitch: 0, label: "Roll 45°", dash: true },
    { key: "r90", roll: 90, pitch: 0, label: "Roll 90°", dash: true },
    { key: "r180", roll: 180, pitch: 0, label: "Roll 180°", dash: true },
    { key: "p15", roll: 180, pitch: 15, label: "Roll 180° · pitch 15°", dash: false },
    { key: "p30", roll: 180, pitch: 30, label: "Roll 180° · pitch 30°", dash: false },
    { key: "p45", roll: 180, pitch: 45, label: "Roll 180° · pitch 45°", dash: false }
  ];
  function condOf(roll, pitch) {
    for (var i = 0; i < CONDS.length; i++) if (CONDS[i].roll === roll && CONDS[i].pitch === pitch) return CONDS[i];
    return null;
  }
  var MM = window.MM = window.MM || {};
  MM.CONDS = CONDS;
  MM.condOf = condOf;

  function svg(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) {
      if (k === "style") e.style.cssText = attrs[k];
      else if (k === "text") e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }
  MM.svg = svg;
  function scale(d0, d1, r0, r1) { return function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); }; }
  function fmt(v, n) { return (v < 0 ? "−" : "") + Math.abs(v).toFixed(n); }
  MM.fmt = fmt;
  function playSafe(v) { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
  function setPlayIcon(btn, playing, what) {
    // SVG elements have no .hidden property, so toggle the attribute itself
    $(".i-pause", btn).toggleAttribute("hidden", !playing);
    $(".i-play", btn).toggleAttribute("hidden", playing);
    btn.setAttribute("aria-label", (playing ? "Pause " : "Play ") + what);
  }

  /* ---------------- abstract: collapsed until asked for ----------------
     Open in the markup, so it stays readable without the script. */
  (function abstractToggle() {
    var box = $("#abstract-box");
    if (!box) return;
    var btn = $(".abstract-toggle", box), inner = $(".abstract-inner", box);
    function set(open) {
      box.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", String(open));
      inner.inert = !open;
    }
    set(false);
    btn.addEventListener("click", function () { set(!box.classList.contains("open")); });
  })();

  /* ---------------- nav ---------------- */
  var nav = $("#nav");
  var links = $$(".nav-links a");
  var sections = links.map(function (a) { return $(a.getAttribute("href")); });
  var navLinks = $(".nav-links"), lastActive = -1;
  function navCue() {
    navLinks.classList.toggle("fade-r", navLinks.scrollLeft + navLinks.clientWidth < navLinks.scrollWidth - 2);
    navLinks.classList.toggle("fade-l", navLinks.scrollLeft > 2);
  }
  function onScroll() {
    nav.classList.toggle("scrolled", window.scrollY > 8);
    var y = window.scrollY + window.innerHeight * 0.35, active = -1;
    sections.forEach(function (s, i) { if (s && s.offsetTop <= y) active = i; });
    links.forEach(function (a, i) {
      var on = i === active;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current");
    });
    if (active !== lastActive && active >= 0 && navLinks.scrollWidth > navLinks.clientWidth) {
      var a = links[active];
      if (a.offsetLeft < navLinks.scrollLeft || a.offsetLeft + a.offsetWidth > navLinks.scrollLeft + navLinks.clientWidth) {
        navLinks.scrollTo({ left: a.offsetLeft - 24, behavior: reduceMotion ? "auto" : "smooth" });
      }
    }
    lastActive = active;
    if (tip.classList.contains("on")) hideTip();
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  navLinks.addEventListener("scroll", navCue, { passive: true });
  window.addEventListener("resize", navCue);

  /* ---------------- reveal ----------------
     Elements are visible by default. Only those below the fold at load are tucked away,
     and they come back as they scroll in. */
  var revealIO = "IntersectionObserver" in window ? new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add("in"); e.target.classList.remove("pre"); revealIO.unobserve(e.target); }
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }) : null;
  function initReveal() {
    $$("[data-reveal]").forEach(function (el) {
      if (!revealIO || reduceMotion) { el.classList.add("in"); return; }
      if (el.getBoundingClientRect().top < window.innerHeight * 0.95) {
        requestAnimationFrame(function () { el.classList.add("in"); });
      } else {
        el.classList.add("pre");
        revealIO.observe(el);
      }
    });
  }

  function visibleWatch(el, cb) {
    if (!("IntersectionObserver" in window)) { cb(true); return; }
    new IntersectionObserver(function (es) { es.forEach(function (e) { cb(e.isIntersecting); }); }, { threshold: 0.2 }).observe(el);
  }

  /* ---------------- hero video ----------------
     Starts once its frame has finished revealing, so the release is never missed. */
  (function heroVideo() {
    var v = $("#hero-video"), btn = $("#hero-toggle"), frame = $(".hero-frame");
    if (!v) return;
    var paused = reduceMotion, inView = true, ready = reduceMotion;
    function icon() { setPlayIcon(btn, !v.paused || (!paused && (!inView || !ready)), "video"); }
    function update() { if (!paused && inView && ready) playSafe(v); else v.pause(); icon(); }
    v.addEventListener("play", icon);
    v.addEventListener("pause", icon);
    var go = function () { if (ready) return; ready = true; v.currentTime = 0; update(); };
    frame.addEventListener("animationend", go, { once: true });
    setTimeout(go, 1400);
    btn.addEventListener("click", function () { paused = !paused; ready = true; update(); });
    visibleWatch(v, function (on) { inView = on; update(); });
    update();
  })();

  /* ---------------- synced comparison clips ---------------- */
  (function clips() {
    var vt = $("#clip-tail"), vn = $("#clip-notail");
    if (!vt || !vn) return;
    var vids = [vn, vt];
    var current = null, userPaused = reduceMotion, inView = false, raf = 0, first = true;
    var playBtn = $("#clip-play"), restart = $("#clip-restart"), bar = $("#clip-progress");

    function icons() { setPlayIcon(playBtn, !vt.paused, "both videos"); }
    vt.addEventListener("play", icons);
    vt.addEventListener("pause", icons);
    function tick() {
      var D = vt.duration;
      if (!vt.paused && D) {
        // The clips are matched frame for frame by robot height, so keep them within a
        // frame: nudge the playback rate for small drift, seek only after a big jump.
        // Loop-aware: one clip may wrap a frame before the other.
        var d = vn.currentTime - vt.currentTime;
        if (d > D / 2) d -= D; else if (d < -D / 2) d += D;
        if (Math.abs(d) > 0.25) { vn.currentTime = vt.currentTime; vn.playbackRate = 1; }
        else vn.playbackRate = Math.max(0.85, Math.min(1.15, 1 - d * 3));
        bar.style.width = (vt.currentTime / D * 100) + "%";
      }
      raf = requestAnimationFrame(tick);
    }
    function play() { vids.forEach(playSafe); cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); }
    function pause() { vids.forEach(function (v) { v.pause(); }); vn.playbackRate = 1; cancelAnimationFrame(raf); icons(); }
    function load(name) {
      if (name === current) return;
      current = name;
      vids.forEach(function (v) {
        var morph = v === vt ? "tail" : "notail";
        // fade through the incoming poster rather than an empty card
        v.parentNode.style.backgroundImage = "url(assets/video/" + name + "_" + morph + ".jpg)";
        v.classList.add("fading");
      });
      setTimeout(function () {
        var ready = 0;
        vids.forEach(function (v) {
          var morph = v === vt ? "tail" : "notail";
          v.poster = "assets/video/" + name + "_" + morph + ".jpg";
          v.src = "assets/video/" + name + "_" + morph + ".mp4";
          var done = function () {
            v.removeEventListener("loadeddata", done);
            if (++ready === 2) {
              vids.forEach(function (x) { x.currentTime = 0; x.classList.remove("fading"); });
              bar.style.width = "0%";
              if (inView && !userPaused) play(); else icons();
            }
          };
          v.addEventListener("loadeddata", done);
          v.load();
        });
      }, first ? 0 : 300);
      first = false;
    }
    $$("#clip-conds button").forEach(function (b) {
      b.addEventListener("click", function () {
        $$("#clip-conds button").forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        load(b.dataset.clip);
      });
    });
    playBtn.addEventListener("click", function () {
      if (vt.paused) { userPaused = false; play(); } else { userPaused = true; pause(); }
    });
    restart.addEventListener("click", function () {
      vids.forEach(function (v) { v.currentTime = 0; });
      userPaused = false; play();
    });
    visibleWatch(vt, function (on) {
      inView = on;
      if (on && !current) load($("#clip-conds button[aria-pressed=true]").dataset.clip);
      if (on && !userPaused && current) play();
      if (!on) pause();
    });
    icons();
  })();

  /* ---------------- tooltip ---------------- */
  var tip = $("#tip");
  function showTip(html, x, y) {
    tip.innerHTML = html;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 14));
    var top = y - h - 12 < 8 ? y + 16 : y - h - 12;
    tip.style.left = left + "px"; tip.style.top = top + "px";
    tip.classList.add("on");
  }
  function hideTip() { tip.classList.remove("on"); }
  MM.showTip = showTip; MM.hideTip = hideTip;

  // Charts are drawn with one SVG unit per CSS pixel, so text stays legible at every width.
  function hostWidth(host) { return Math.round(Math.max(300, host.clientWidth || 520)); }

  /* ---------------- Fig. 4: mean body tilt ---------------- */
  var locked = null;
  function tiltCharts() {
    if (!DATA || !DATA.tilt) return;
    var tmax = 0;
    DATA.tilt.forEach(function (c) { tmax = Math.max(tmax, c.t[c.t.length - 1]); });
    ["tail", "notail"].forEach(function (morph) {
      var host = $("#tilt-" + morph);
      host.innerHTML = "";
      var W = hostWidth(host), H = Math.round(Math.min(W * 0.56, 340)) + 40;
      var m = { l: 40, r: 10, t: 28, b: 38 };
      var x = scale(0, tmax, m.l, W - m.r), y = scale(0, 225, H - m.b, m.t);
      var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Mean body tilt over time, " + (morph === "tail" ? "with tail" : "no tail") }, host);
      var ax = svg("g", { class: "axis" }, s);
      [0, 45, 90, 135, 180, 225].forEach(function (v) {
        svg("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "gridline" }, ax);
        svg("text", { x: m.l - 7, y: y(v) + 3.5, "text-anchor": "end", text: v + "°" }, ax);
      });
      svg("text", { x: 0, y: 11, "text-anchor": "start", class: "axis-title", text: "mean body tilt" }, ax);
      [0, 0.2, 0.4, 0.6].forEach(function (v) {
        if (v > tmax + 1e-6) return;
        svg("text", { x: x(v), y: H - m.b + 16, "text-anchor": "middle", text: v.toFixed(1) }, ax);
      });
      svg("text", { x: W - m.r, y: H - 4, "text-anchor": "end", class: "axis-title", text: "time since free-fall detection (s)" }, ax);
      [1, 2].forEach(function (d) {
        var tt = Math.sqrt(2 * d / G);
        if (tt > tmax) return;
        svg("line", { x1: x(tt), x2: x(tt), y1: m.t - 4, y2: H - m.b, class: "guide" }, s);
        var gx = x(tt), nearEdge = gx > W - m.r - 36;
        svg("text", { x: nearEdge ? W - m.r : gx, y: m.t - 8, "text-anchor": nearEdge ? "end" : "middle", class: "guide-label", text: d + " m fallen" }, s);
      });
      var gBands = svg("g", {}, s), gLines = svg("g", {}, s);
      DATA.tilt.filter(function (c) { return c.morph === morph; }).forEach(function (c) {
        var cond = condOf(c.roll, c.pitch); if (!cond) return;
        var col = "var(--c-" + cond.key + ")";
        var up = "", dn = "", line = "";
        for (var i = 0; i < c.t.length; i++) {
          up += (i ? "L" : "M") + x(c.t[i]).toFixed(1) + "," + y(Math.min(225, c.mean[i] + c.sd[i])).toFixed(1);
          line += (i ? "L" : "M") + x(c.t[i]).toFixed(1) + "," + y(c.mean[i]).toFixed(1);
        }
        for (var j = c.t.length - 1; j >= 0; j--) dn += "L" + x(c.t[j]).toFixed(1) + "," + y(Math.max(0, c.mean[j] - c.sd[j])).toFixed(1);
        svg("path", { d: up + dn + "Z", class: "band fade-band", "data-k": cond.key, style: "fill:" + col + ";stroke:none" }, gBands);
        var p = svg("path", { d: line, class: "series " + (cond.dash ? "fade-band" : "draw"), "data-k": cond.key,
          style: "stroke:" + col + (cond.dash ? ";stroke-dasharray:5 4" : "") }, gLines);
        p.style.setProperty("--i", CONDS.indexOf(cond));
        if (!cond.dash) p.style.setProperty("--len", Math.ceil(p.getTotalLength()) + 1);
        svg("title", { text: cond.label + " · n = " + c.n }, p);
      });
    });
    highlight(locked);
  }
  function highlight(k) {
    $$("#tilt-charts svg").forEach(function (s) {
      s.classList.toggle("dim", !!k);
      $$("[data-k]", s).forEach(function (p) { p.classList.toggle("hot", p.getAttribute("data-k") === k); });
    });
  }
  function tiltLegend() {
    var legend = $("#tilt-legend"), wrap = $("#tilt-charts");
    if (!legend || !DATA || !DATA.tilt) return;
    CONDS.forEach(function (c) {
      var b = document.createElement("button");
      b.type = "button";
      b.style.color = "var(--c-" + c.key + ")";
      b.setAttribute("aria-pressed", "false");
      b.innerHTML = '<i class="' + (c.dash ? "dash" : "") + '"></i><span>' + c.label + "</span>";
      var on = function () { wrap.classList.add("settled"); highlight(c.key); }, off = function () { highlight(locked); };
      b.addEventListener("mouseenter", on); b.addEventListener("focus", on);
      b.addEventListener("mouseleave", off); b.addEventListener("blur", off);
      b.addEventListener("click", function () {
        locked = locked === c.key ? null : c.key;
        $$("button", legend).forEach(function (x, i) { x.setAttribute("aria-pressed", String(CONDS[i].key === locked)); });
        wrap.classList.add("settled");
        highlight(locked);
      });
      legend.appendChild(b);
    });
  }

  /* ---------------- Fig. 5: whole-cat scores ---------------- */
  function quantile(a, p) {
    var pos = (a.length - 1) * p, lo = Math.floor(pos), hi = Math.ceil(pos);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  }
  function boxStats(vals) {
    var a = vals.slice().sort(function (p, q) { return p - q; });
    var q1 = quantile(a, 0.25), q2 = quantile(a, 0.5), q3 = quantile(a, 0.75), iqr = q3 - q1;
    var lo = a[0], hi = a[a.length - 1];
    for (var i = 0; i < a.length; i++) if (a[i] >= q1 - 1.5 * iqr) { lo = a[i]; break; }
    for (var j = a.length - 1; j >= 0; j--) if (a[j] <= q3 + 1.5 * iqr) { hi = a[j]; break; }
    return { q1: q1, q2: q2, q3: q3, lo: lo, hi: hi };
  }
  function replayDrop(id, focus) {
    hideTip();
    var go = function () { if (MM.replay) MM.replay.select({ source: "hw", id: id }, true, focus); };
    if (MM.replay) go();
    else if (MM.loadReplay) MM.loadReplay().then(function () { setTimeout(go, 0); }, function () {});
  }
  function scoreCharts() {
    if (!DATA || !DATA.trials) return;
    var scored = DATA.trials.filter(function (t) { return t.included && t.score != null; });
    var min = Math.min.apply(null, scored.map(function (t) { return t.score; }));
    var y0 = Math.min(0.4, Math.floor((min - 0.03) * 10) / 10);
    var panels = [
      { host: "#score-roll", conds: [CONDS[0], CONDS[1], CONDS[2]], tick: function (c) { return c.roll + "°"; }, axis: "release roll" },
      { host: "#score-pitch", conds: [CONDS[2], CONDS[3], CONDS[4], CONDS[5]], tick: function (c) { return c.pitch + "°"; }, axis: "release pitch, nose down" }
    ];
    panels.forEach(function (P, pi) {
      var host = $(P.host);
      host.innerHTML = "";
      var W = hostWidth(host), H = Math.round(Math.min(W * 0.5, 300)) + 50;
      var m = { l: 36, r: 8, t: 26, b: 40 };
      var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
      var defs = svg("defs", {}, s);
      var pat = svg("pattern", { id: "hatch" + pi, width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
      svg("rect", { width: 5, height: 5, style: "fill:var(--accent-weak)" }, pat);
      svg("line", { x1: 0, y1: 0, x2: 0, y2: 5, style: "stroke:var(--accent);stroke-width:1.4;opacity:.55" }, pat);
      var y = scale(y0, 1.0, H - m.b, m.t);
      var ax = svg("g", { class: "axis", "aria-hidden": "true" }, s);
      for (var v = Math.round(y0 * 10); v <= 10; v++) {
        var val = v / 10;
        svg("line", { x1: m.l, x2: W - m.r, y1: y(val), y2: y(val), class: "gridline" }, ax);
        svg("text", { x: m.l - 7, y: y(val) + 3.5, "text-anchor": "end", text: val.toFixed(1) }, ax);
      }
      svg("text", { x: 0, y: 11, "text-anchor": "start", class: "axis-title", text: "whole-cat score" }, ax);
      if (y0 > 0) {
        var by = H - m.b + 1;
        svg("path", { d: "M" + (m.l - 5) + "," + (by + 3) + "l6,-4M" + (m.l - 5) + "," + (by + 7) + "l6,-4", style: "stroke:var(--muted);fill:none" }, ax);
      }
      var gw = (W - m.l - m.r) / P.conds.length;
      var bw = Math.max(18, Math.min(30, gw * 0.3));
      var dotsG = svg("g", { role: "group", "aria-label": "Whole-cat scores by " + P.axis + ". Use the arrow keys to move between drops and Enter to replay one." });
      var allDots = [];
      P.conds.forEach(function (c, ci) {
        var cx = m.l + gw * (ci + 0.5);
        svg("text", { x: cx, y: H - m.b + 17, "text-anchor": "middle", text: P.tick(c) }, ax);
        ["notail", "tail"].forEach(function (morph, mi) {
          var vals = scored.filter(function (t) { return t.morph === morph && t.roll === c.roll && t.pitch === c.pitch; });
          if (!vals.length) return;
          var st = boxStats(vals.map(function (t) { return t.score; }));
          var bx = cx + (mi ? 4 : -4 - bw);
          var g = svg("g", { "aria-hidden": "true" }, s);
          var stroke = morph === "tail" ? "var(--accent)" : "var(--grey)";
          svg("line", { x1: bx + bw / 2, x2: bx + bw / 2, y1: y(st.lo), y2: y(st.q1), style: "stroke:" + stroke }, g);
          svg("line", { x1: bx + bw / 2, x2: bx + bw / 2, y1: y(st.q3), y2: y(st.hi), style: "stroke:" + stroke }, g);
          svg("line", { x1: bx + bw * 0.28, x2: bx + bw * 0.72, y1: y(st.lo), y2: y(st.lo), style: "stroke:" + stroke }, g);
          svg("line", { x1: bx + bw * 0.28, x2: bx + bw * 0.72, y1: y(st.hi), y2: y(st.hi), style: "stroke:" + stroke }, g);
          svg("rect", { x: bx, y: y(st.q3), width: bw, height: Math.max(1, y(st.q1) - y(st.q3)), rx: 2,
            style: (morph === "tail" ? "fill:url(#hatch" + pi + ")" : "fill:var(--grey-weak)") + ";stroke:" + stroke }, g);
          svg("line", { x1: bx, x2: bx + bw, y1: y(st.q2), y2: y(st.q2), style: "stroke-width:2;stroke:" + (morph === "tail" ? "var(--accent)" : "var(--ink-2)") }, g);
          vals.forEach(function (t, k) {
            // deterministic jitter so the layout is stable between visits
            var j = (((k * 7919 + ci * 104729 + mi * 31) % 97) / 97 - 0.5) * (bw - 8);
            var px = bx + bw / 2 + j, py = y(t.score);
            var label = (morph === "tail" ? "With tail" : "No tail") + ", " + c.label + ", score " + t.score.toFixed(3);
            var dg = svg("g", { class: "dot-g", tabindex: "-1", role: "button", "aria-label": label + ". Replay this drop." }, dotsG);
            svg("circle", { cx: px, cy: py, r: 10, class: "hit", style: "fill:transparent" }, dg);
            svg("circle", { cx: px, cy: py, r: 3.4, class: "dot pop",
              style: "fill:" + (morph === "tail" ? "var(--accent)" : "var(--grey)") + ";stroke:var(--surface);stroke-width:1.2;--i:" + (k + ci * 12) }, dg);
            var html = "<b>" + (morph === "tail" ? "With tail" : "No tail") + "</b> · " + c.label + "<br><span>whole-cat score</span> " + t.score.toFixed(3) + "<br><span>" + t.id + "</span><br><span>select to replay</span>";
            dg.addEventListener("mousemove", function (e) { showTip(html, e.clientX, e.clientY); });
            dg.addEventListener("mouseleave", hideTip);
            dg.addEventListener("focus", function () { var r = dg.getBoundingClientRect(); showTip(html, r.right, r.top); });
            dg.addEventListener("blur", hideTip);
            dg.addEventListener("click", function () { replayDrop(t.id, false); });
            dg.addEventListener("keydown", function (e) {
              var i = allDots.indexOf(dg);
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); replayDrop(t.id, true); }
              else if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); move(i + 1); }
              else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); move(i - 1); }
              else if (e.key === "Home") { e.preventDefault(); move(0); }
              else if (e.key === "End") { e.preventDefault(); move(allDots.length - 1); }
            });
            allDots.push(dg);
          });
        });
      });
      s.appendChild(dotsG);   // dots above the boxes
      // roving tab stop: one per panel, arrows move within it
      function move(i) {
        if (i < 0 || i >= allDots.length) return;
        allDots.forEach(function (d) { d.setAttribute("tabindex", "-1"); });
        allDots[i].setAttribute("tabindex", "0");
        allDots[i].focus();
      }
      if (allDots.length) allDots[0].setAttribute("tabindex", "0");
      svg("text", { x: W - m.r, y: H - 4, "text-anchor": "end", class: "axis-title", text: P.axis }, ax);
    });
  }

  // Redraw the charts when their width changes enough to matter.
  function watchChartWidth() {
    if (!("ResizeObserver" in window) || !$("#tilt-tail")) return;
    var lastW = $("#tilt-tail").clientWidth, timer = 0;
    new ResizeObserver(function () {
      clearTimeout(timer);
      timer = setTimeout(function () {
        var w = $("#tilt-tail").clientWidth;
        if (Math.abs(w - lastW) < 24) return;
        lastW = w;
        var f = document.activeElement && document.activeElement.closest && document.activeElement.closest(".dot-g");
        var key = f && f.getAttribute("aria-label");
        tiltCharts();
        scoreCharts();
        if (key) {
          var n = $$("#score-charts .dot-g").filter(function (d) { return d.getAttribute("aria-label") === key; })[0];
          if (n) {
            $$(".dot-g", n.closest("svg")).forEach(function (d) { d.setAttribute("tabindex", "-1"); });
            n.setAttribute("tabindex", "0");
            n.focus({ preventScroll: true });
          }
        }
      }, 150);
    }).observe($("#tilt-charts"));
  }

  /* ---------------- results table scroll cue ---------------- */
  function tableCue() {
    $$(".table-wrap, .pipeline-wrap").forEach(function (w) {
      var set = function () { w.classList.toggle("scrolls", w.scrollWidth > w.clientWidth + 2 && w.scrollLeft + w.clientWidth < w.scrollWidth - 2); };
      w.addEventListener("scroll", set, { passive: true });
      window.addEventListener("resize", set);
      set();
    });
  }

  /* ---------------- BibTeX ---------------- */
  $("#bib-copy").addEventListener("click", function () {
    var b = this, text = $("#bibtex").textContent;
    var done = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1600); };
    var fallback = function () {
      var r = document.createRange(); r.selectNodeContents($("#bibtex"));
      var sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
      b.textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? "Press ⌘C" : "Press Ctrl+C";
      setTimeout(function () { b.textContent = "Copy"; }, 2500);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback);
    else fallback();
  });

  /* ---------------- replay fallback ---------------- */
  function replayFallback(msg) {
    var st = $("#stage3d");
    if (st && !$(".fallback", st)) st.insertAdjacentHTML("beforeend", '<div class="fallback">' + msg + "</div>");
  }
  MM.replayFallback = replayFallback;
  if (location.protocol === "file:") {
    replayFallback("The 3D replay needs the page served over http(s), for example with <code>python3 -m http.server</code> in <code>docs/</code>.");
  }

  /* ---------------- boot ---------------- */
  tiltLegend();
  tiltCharts();
  scoreCharts();
  watchChartWidth();
  tableCue();
  initReveal();
  onScroll();
  navCue();
})();
