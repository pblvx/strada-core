/* STRADA-CORE · track_canvas.js — WebSocket bus + circuito Silverstone + estela térmica.
   Expone window.STRADA (bus de eventos) para que telemetry_hud.js comparta UNA sola conexión WS. */
(function () {
  "use strict";

  // ---------- Bus de eventos compartido ----------
  const STRADA = {
    _packetCbs: [],
    _statusCbs: [],
    onPacket(cb) { this._packetCbs.push(cb); },
    onStatus(cb) { this._statusCbs.push(cb); },
    _emitPacket(p) { for (const cb of this._packetCbs) { try { cb(p); } catch (e) {} } },
    _emitStatus(s) { for (const cb of this._statusCbs) { try { cb(s); } catch (e) {} } },
  };
  window.STRADA = STRADA;

  // ---------- Conexión WebSocket con reconexión (backoff) ----------
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const WS_URL = `${proto}//${window.location.host}/ws/telemetry`;
  let ws = null;
  let retry = 0;
  let closedByUser = false;

  function setStatus(kind, text) { STRADA._emitStatus({ kind, text }); }

  function connect() {
    setStatus("down", "CONNECTING");
    try {
      ws = new WebSocket(WS_URL);
    } catch (e) {
      scheduleReconnect();
      return;
    }
    ws.onopen = () => { retry = 0; setStatus("live", "LIVE"); };
    ws.onmessage = (evt) => {
      try { STRADA._emitPacket(JSON.parse(evt.data)); } catch (e) {}
    };
    ws.onerror = () => { setStatus("down", "ERROR"); };
    ws.onclose = () => {
      if (closedByUser) { return; }
      setStatus("down", "RECONNECTING");
      scheduleReconnect();
    };
  }
  function scheduleReconnect() {
    retry = Math.min(retry + 1, 5);
    const delay = Math.min(1000 * retry, 5000);
    setTimeout(() => { if (!closedByUser) { connect(); } }, delay);
  }

  // ---------- Canvas: circuito + estela térmica ----------
  const canvas = document.getElementById("track");
  const ctx = canvas.getContext("2d");
  let dpr = window.devicePixelRatio || 1;

  const PAD = 40;
  let trackPts = [];      // historial completo (x,y) para trazar el circuito
  let trail = [];         // últimos 40 puntos {x,y,temp} para la estela
  let bounds = { minX: 0, minY: 0, maxX: 1000, maxY: 1000 };
  let car = { x: 500, y: 500, prev: { x: 500, y: 500 }, temp: 95, shown: false };
  let circuit = [];        // trazado completo de Silverstone desde /api/circuit
  let circuitLoaded = false;
  let boundsFixed = false; // true cuando el bounding box proviene del circuito

  function resize() {
    const wrap = document.getElementById("canvas-wrap");
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);

  // Mapea coordenadas [0,1000] -> pixels del viewport con auto-escala + padding
  function toScreen(x, y) {
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const bw = Math.max(1e-6, bounds.maxX - bounds.minX);
    const bh = Math.max(1e-6, bounds.maxY - bounds.minY);
    const areaW = w - PAD * 2;
    const areaH = h - PAD * 2;
    // escala uniforme: no deforma el circuito
    const scale = Math.min(areaW / bw, areaH / bh);
    // centrado dentro del area disponible
    const offX = PAD + (areaW - bw * scale) / 2;
    const offY = PAD + (areaH - bh * scale) / 2;
    // eje Y invertido: en cartesianas Y crece hacia arriba, en canvas hacia abajo
    return {
      px: offX + (x - bounds.minX) * scale,
      py: offY + (bounds.maxY - y) * scale,
    };
  }

  // Color por temperatura: <80 cian, 80-105 verde óptimo, >115 rojo (rampa lineal)
  function tempColor(t) {
    const COLD = [0, 229, 255], OPT = [0, 255, 136], HOT = [255, 51, 102];
    let c;
    if (t <= 80) { c = COLD; }
    else if (t <= 105) {
      const u = (t - 80) / 25; c = lerp3(COLD, OPT, u);
    } else if (t <= 115) {
      const u = (t - 105) / 10; c = lerp3(OPT, HOT, u);
    } else { c = HOT; }
    return `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;
  }
  function lerp3(a, b, u) {
    u = Math.max(0, Math.min(1, u));
    return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
  }

  // ---------- Trazado completo del circuito desde /api/circuit ----------
  async function loadCircuit() {
    try {
      const res = await fetch("/api/circuit");
      if (!res.ok) { throw new Error("HTTP " + res.status); }
      circuit = await res.json();
      circuitLoaded = true;
      if (circuit.length >= 2) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const p of circuit) {
          if (p.x < minX) { minX = p.x; }
          if (p.x > maxX) { maxX = p.x; }
          if (p.y < minY) { minY = p.y; }
          if (p.y > maxY) { maxY = p.y; }
        }
        // margen de padding proporcional para centrar el circuito en el canvas
        const mx = (maxX - minX) * 0.04;
        const my = (maxY - minY) * 0.04;
        bounds = { minX: minX - mx, minY: minY - my, maxX: maxX + mx, maxY: maxY + my };
        boundsFixed = true;
      }
    } catch (e) {
      // fallback: seguir con auto-escala acumulada por paquetes
      circuitLoaded = true;
    }
  }

  // ---------- Recepción de paquetes ----------
  STRADA.onPacket((p) => {
    if (typeof p.x !== "number" || typeof p.y !== "number") { return; }
    // solo auto-escala mientras no tengamos el circuito fijo del endpoint
    if (!boundsFixed) {
      bounds.minX = Math.min(bounds.minX, p.x);
      bounds.maxX = Math.max(bounds.maxX, p.x);
      bounds.minY = Math.min(bounds.minY, p.y);
      bounds.maxY = Math.max(bounds.maxY, p.y);
    }

    trackPts.push({ x: p.x, y: p.y });
    if (trackPts.length > 6000) { trackPts.shift(); }

    const temp = (typeof p.pinn_tyre_temp_c === "number") ? p.pinn_tyre_temp_c : car.temp;
    car.prev = { x: car.x, y: car.y };
    car.x = p.x;
    car.y = p.y;
    car.temp = temp;
    car.shown = true;

    trail.push({ x: p.x, y: p.y, temp: temp });
    if (trail.length > 40) { trail.shift(); }

    const lapEl = document.getElementById("lap-num");
    if (lapEl && typeof p.lap_number === "number") { lapEl.textContent = p.lap_number; }
  });

  // ---------- Render loop (60 FPS) ----------
  // Traza el circuito con beginPath/moveTo/lineTo (SIN closePath). Si hay un salto
  // puntos consecutivos (p.e. reinicio de posicion), rompe el trazo con moveTo para
  // NO dibujar una linea diagonal que cruce el centro de la pista.
  const MAX_JUMP = 60; // unidades normalizadas [0,1000] entre puntos consecutivos
  function traceCircuit() {
    ctx.beginPath();
    let prev = circuit[0];
    const p0 = toScreen(prev.x, prev.y);
    ctx.moveTo(p0.px, p0.py);
    for (let i = 1; i < circuit.length; i++) {
      const cur = circuit[i];
      const dx = cur.x - prev.x;
      const dy = cur.y - prev.y;
      const s = toScreen(cur.x, cur.y);
      if (dx * dx + dy * dy > MAX_JUMP * MAX_JUMP) {
        ctx.moveTo(s.px, s.py); // salto: no unir con linea
      } else {
        ctx.lineTo(s.px, s.py);
      }
      prev = cur;
    }
  }

  function drawCircuit() {
    if (circuit.length < 2) { return; }
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    // 1) asfalto: linea gruesa gris azulada
    ctx.lineWidth = 9 * dpr;
    ctx.strokeStyle = "#1e293b";
    traceCircuit();
    ctx.stroke();
    // 2) linea central tenue (trazo de trazada)
    ctx.lineWidth = 1 * dpr;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    traceCircuit();
    ctx.stroke();
  }

  function drawTrail() {
    if (trail.length < 2) { return; }
    ctx.lineCap = "round";
    for (let i = 1; i < trail.length; i++) {
      const a = toScreen(trail[i - 1].x, trail[i - 1].y);
      const b = toScreen(trail[i].x, trail[i].y);
      const alpha = i / trail.length;          // se desvanece hacia la cola
      const col = tempColor(trail[i].temp);
      ctx.globalAlpha = 0.15 + 0.85 * alpha;
      ctx.strokeStyle = col;
      ctx.lineWidth = (1 + 4 * alpha) * dpr;
      ctx.shadowColor = col;
      ctx.shadowBlur = 8 * alpha * dpr;
      ctx.beginPath();
      ctx.moveTo(a.px, a.py);
      ctx.lineTo(b.px, b.py);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
  }

  function drawCar() {
    if (!car.shown) { return; }
    const s = toScreen(car.x, car.y);
    const col = tempColor(car.temp);
    ctx.shadowColor = col;
    ctx.shadowBlur = 18 * dpr;
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(s.px, s.py, 5 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    // núcleo blanco del monoplaza
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(s.px, s.py, 2 * dpr, 0, Math.PI * 2);
    ctx.fill();
  }

  function frame() {
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    ctx.clearRect(0, 0, w, h);
    // rejilla tenue de fondo
    ctx.fillStyle = "#0a0c10";
    ctx.fillRect(0, 0, w, h);
    drawCircuit();
    drawTrail();
    drawCar();
    requestAnimationFrame(frame);
  }

  // ---------- Arranque ----------
  resize();
  requestAnimationFrame(frame);
  connect();
  loadCircuit();

  window.addEventListener("beforeunload", () => {
    closedByUser = true;
    if (ws) { try { ws.close(); } catch (e) {} }
  });
})();

