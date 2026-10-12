/* STRADA-CORE · telemetry_hud.js — HUD numérico, panel térmico, tarjeta táctica MCTS.
   Comparte la conexión WS creada por track_canvas.js a través de window.STRADA. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  // ---------- Estado de enlace (LIVE / RECONNECTING) ----------
  if (window.STRADA && STRADA.onStatus) {
    STRADA.onStatus((s) => {
      const box = $("link-status");
      const txt = $("link-text");
      if (!box || !txt) { return; }
      box.classList.toggle("is-live", s.kind === "live");
      box.classList.toggle("is-down", s.kind !== "live");
      txt.textContent = s.text;
    });
  }

  // ---------- Helpers ----------
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const RPM_MAX = 13000;

  function tempColor(t) {
    if (t < 80) return "#00e5ff";
    if (t <= 105) return "#00ff88";
    if (t <= 115) return "#ff9f1c";
    return "#ff3366";
  }
  function tempState(t, kind) {
    if (kind === "brake") {
      if (t < 250) return "COLD";
      if (t <= 600) return "OPTIMAL";
      return "OVERHEAT";
    }
    if (t < 80) return "COLD";
    if (t <= 105) return "OPTIMAL";
    return "OVERHEAT";
  }
  function setBar(el, pct) {
    if (el) { el.style.width = clamp(pct, 0, 100).toFixed(1) + "%"; }
  }
  function compoundColor(c) {
    const k = (c || "").toUpperCase();
    if (k === "SOFT") return "#ff3366";
    if (k === "MEDIUM") return "#ffd400";
    if (k === "HARD") return "#e8e8e8";
    return "#6b7a8d";
  }

  // ---------- HUD de telemetría ----------
  function updateTelemetry(p) {
    // velocímetro
    if (typeof p.speed_kmh === "number") { $("speed").textContent = Math.round(p.speed_kmh); }
    // marcha (1-8, N si 0)
    if (typeof p.gear === "number") { $("gear").textContent = p.gear <= 0 ? "N" : p.gear; }
    // rpm
    if (typeof p.rpm === "number") {
      $("rpm").textContent = Math.round(p.rpm);
      const pct = (p.rpm / RPM_MAX) * 100;
      setBar($("rpm-fill"), pct);
      $("rpm-fill").classList.toggle("redline", pct > 88);
    }
    // acelerador / freno (transición CSS suave)
    if (typeof p.throttle_pct === "number") {
      setBar($("throttle-bar"), p.throttle_pct);
      $("throttle-pct").textContent = Math.round(p.throttle_pct) + "%";
    }
    if (typeof p.brake_pct === "number") {
      setBar($("brake-bar-pedal"), p.brake_pct);
      $("brake-pct").textContent = Math.round(p.brake_pct) + "%";
    }
    // DRS
    if (typeof p.drs_active === "boolean") {
      $("drs-badge").classList.toggle("on", p.drs_active);
    }

    // ---------- Panel térmico PINN ----------
    if (typeof p.pinn_tyre_temp_c === "number") {
      const t = p.pinn_tyre_temp_c;
      $("tyre-temp").textContent = t.toFixed(1);
      $("tyre-temp").style.color = tempColor(t);
      setBar($("tyre-bar"), clamp((t / 150) * 100, 0, 100));
      $("tyre-bar").style.background = tempColor(t);
      $("tyre-state").textContent = tempState(t, "tyre");
      $("tyre-state").style.color = tempColor(t);
    }
    if (typeof p.pinn_brake_temp_c === "number") {
      const t = p.pinn_brake_temp_c;
      $("brake-temp").textContent = t.toFixed(1);
      $("brake-temp").style.color = tempColor(t);
      setBar($("brake-bar"), clamp((t / 1000) * 100, 0, 100));
      $("brake-bar").style.background = tempColor(t);
      $("brake-state").textContent = tempState(t, "brake");
      $("brake-state").style.color = tempColor(t);
    }
    if (p.tyre_compound) {
      const chip = $("compound-chip");
      chip.textContent = p.tyre_compound;
      chip.style.color = compoundColor(p.tyre_compound);
      chip.style.borderColor = compoundColor(p.tyre_compound);
    }
    if (typeof p.tyre_age_laps === "number") { $("tyre-age").textContent = p.tyre_age_laps; }
  }

  if (window.STRADA && STRADA.onPacket) {
    STRADA.onPacket(updateTelemetry);
  }

  // ---------- Tarjeta táctica MCTS ----------
  function renderDecision(d) {
    const actionEl = $("tac-action");
    actionEl.textContent = d.action;
    actionEl.classList.toggle("act-box", d.action === "BOX_NOW");
    actionEl.classList.toggle("act-stay", d.action === "STAY_OUT");

    $("tac-compound").textContent = d.recommended_compound;
    $("tac-compound").style.color = compoundColor(d.recommended_compound);
    $("tac-pitlap").textContent = "L" + d.recommended_pit_lap;
    $("tac-traffic").textContent = (d.traffic_risk_score * 100).toFixed(0) + "%";
    $("tac-undercut").textContent = (d.undercut_success_probability * 100).toFixed(0) + "%";

    const delta = $("tac-delta");
    const dt = d.expected_race_time_delta_s;
    delta.textContent = (dt >= 0 ? "+" : "") + dt.toFixed(2) + "s";
    delta.classList.toggle("pos", dt >= 0);
    delta.classList.toggle("neg", dt < 0);

    // alerta visual de parada si ventana abierta o BOX_NOW
    const card = $("tactical-card");
    card.classList.toggle("is-pit", d.pit_window_open === true || d.action === "BOX_NOW");
  }

  async function refreshStrategy() {
    const btn = $("refresh-btn");
    try {
      if (btn) { btn.style.opacity = "0.5"; }
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 8000);
      const res = await fetch("/api/strategy/decision", { signal: ctrl.signal });
      clearTimeout(to);
      if (!res.ok) { throw new Error("HTTP " + res.status); }
      renderDecision(await res.json());
    } catch (e) {
      // No tumba la UI si el MCTS tarda o falla (regla: desconexiones limpias)
      const card = $("tactical-card");
      if (card) { card.classList.remove("is-pit"); }
    } finally {
      if (btn) { btn.style.opacity = "1"; }
    }
  }

  // botón manual + auto-refresh cada 5 s
  const btn = $("refresh-btn");
  if (btn) { btn.addEventListener("click", refreshStrategy); }
  setInterval(refreshStrategy, 5000);
  refreshStrategy();
})();

