"""
Pestaña de Streaming Continuo en Tiempo Real (WebSocket) para la interfaz AuraVoice.

Arquitectura cliente-servidor:
  Navegador (micrófono -> AudioWorklet -> remuestreo a 16 kHz -> WebSocket)
      <-> src/websocket_server.py (búfer circular + emotion2vec+ Large en CUDA)

Todo el panel en vivo (control, veredicto, barras de 9 clases, plano de Russell e
historial) vive dentro de un único componente HTML, porque los resultados llegan por
WebSocket directamente al navegador y Streamlit no necesita re-ejecutarse para pintarlos.
"""

import json
import socket

import streamlit as st

from src.result_parser import EMOTION_COLORS

WS_PORT = 8765


def check_ws_server(host: str = "127.0.0.1", port: int = WS_PORT) -> dict:
    """
    Comprueba si el servidor WebSocket está activo sin lanzar nunca una excepción.
    Primero intenta un handshake WebSocket real (PING/PONG); si la librería no está
    disponible, cae a una comprobación TCP simple del puerto.
    """
    try:
        from websockets.sync.client import connect

        with connect(f"ws://{host}:{port}", open_timeout=0.8, close_timeout=0.3) as ws:
            ws.recv(timeout=1.0)            # Mensaje READY inicial del servidor
            ws.send("PING")
            pong = json.loads(ws.recv(timeout=1.0))
            return {"online": True, "device": pong.get("device", "?")}
    except ImportError:
        pass
    except Exception:
        return {"online": False, "device": None}

    try:
        with socket.create_connection((host, port), timeout=0.4):
            return {"online": True, "device": "?"}
    except OSError:
        return {"online": False, "device": None}


def _client_html(port: int) -> str:
    classes = json.dumps([{"name": k, "color": v} for k, v in EMOTION_COLORS.items()], ensure_ascii=False)
    return CLIENT_TEMPLATE.replace("__WS_PORT__", str(port)).replace("__CLASSES__", classes)


def render_streaming_tab(render_html):
    """Dibuja la pestaña completa de streaming en tiempo real vía WebSocket."""
    status = check_ws_server()
    c_status, c_retry = st.columns([5, 1])
    with c_status:
        if status["online"]:
            st.success(f"**Servidor WebSocket Activo** en `ws://localhost:{WS_PORT}` · Dispositivo de cómputo en silicio: `{status['device']}`")
        else:
            st.warning(
                "**Servidor WebSocket apagado o inaccesible.** Para transmitir micrófono en vivo:\n\n"
                "1. Abre otra terminal en la carpeta del proyecto y ejecuta: `uv run python src/websocket_server.py`\n"
                "2. Espera el mensaje *Servidor listo*\n"
                "3. Pulsa **Comprobar de nuevo**."
            )
    with c_retry:
        st.button("Comprobar de nuevo", key="ws_recheck", use_container_width=True)

    try:
        st.components.v1.html(_client_html(WS_PORT), height=740, scrolling=False)
    except Exception as exc:  # El panel nunca debe tumbar la app
        st.info(f"No se pudo cargar el panel de streaming en este navegador: {exc}")

    st.caption(
        "Si el navegador bloquea el micrófono, haz clic en el icono del candado junto a la URL "
        "y permite el acceso al micrófono para localhost. El análisis es acústico (sin transcripción) "
        "y es una estimación probabilística, no un diagnóstico."
    )




# ------------------------------------------------------------------------------
# Cliente HTML/JS (se inyecta dentro del iframe del componente).
# El iframe no hereda el CSS de app.py, así que se replican aquí las mismas variables
# y clases del Sistema de Diseño AuraVoice (section-card, hero-emotion-banner,
# prob-item, spec-box, botón primario) para que la pestaña se vea igual al resto.
# ------------------------------------------------------------------------------
CLIENT_TEMPLATE = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
    --primary: #EA580C; --secondary: #F43F5E; --tertiary: #0D9488;
    --neutral-dark: #382E2B; --neutral-body: #574A43; --neutral-muted: #8F7D74;
    --bg-page: #FBF8F5; --bg-card: #FFFFFF; --bg-subtle: #F6EFE9;
    --border-card: #EFE4DC; --border-light: #E7DDD4;
    --shadow-card: 0 4px 20px -2px rgba(87, 74, 67, 0.06);
    --radius-card: 18px; --radius-pill: 9999px;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 2px; background: transparent;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; color: var(--neutral-body); }
.grid { display: grid; grid-template-columns: 1fr 1.25fr; gap: 16px; align-items: stretch; }

/* --- Clases copiadas del Sistema de Diseño AuraVoice (app.py) --- */
.section-card { background: #FFFFFF; border: 1px solid var(--border-card); border-radius: var(--radius-card);
    padding: 22px; box-shadow: var(--shadow-card); display: flex; flex-direction: column; }
.card-title-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.card-title { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.85rem; font-weight: 800; letter-spacing: 0.5px;
    color: var(--neutral-muted); text-transform: uppercase; display: flex; align-items: center; gap: 6px; }
.confidence-pill { background: #CCFBF1; color: #0F766E; font-size: 0.76rem; font-weight: 700; padding: 4px 10px;
    border-radius: var(--radius-pill); border: 1px solid #99F6E4; }
.nav-badge { background: #FEF3C7; color: #92400E; font-size: 0.78rem; font-weight: 700; padding: 4px 12px;
    border-radius: var(--radius-pill); display: inline-flex; align-items: center; gap: 6px; border: 1px solid #FDE68A; }
.nav-badge.live { background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }
.nav-badge.error { background: #FFF1F2; color: #9F1239; border-color: #FECDD3; }
.nav-badge-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.hero-emotion-banner { background: linear-gradient(135deg, #FFF7ED 0%, #FFF1F2 100%); border: 1px solid #FED7AA;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; }
.emotion-name-large { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.85rem; font-weight: 800; color: #261E1A;
    line-height: 1.1; margin-bottom: 4px; transition: color .3s; }
.circumplex-vector { font-size: 0.78rem; font-weight: 600; color: #9A3412; font-family: 'JetBrains Mono', monospace; }
.emotion-state-tag { background: var(--primary); color: white; font-size: 0.78rem; font-weight: 700; padding: 6px 14px;
    border-radius: var(--radius-pill); box-shadow: 0 2px 8px rgba(234, 88, 12, 0.25); white-space: nowrap; }
.prob-item { margin-bottom: 7px; }
.prob-meta { display: flex; justify-content: space-between; font-size: 0.80rem; font-weight: 600; margin-bottom: 3px; }
.prob-name { color: #382E2B; }
.prob-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #1F2937; }
.prob-track { background: #F3ECE6; height: 6px; border-radius: 9999px; overflow: hidden; }
.prob-fill { height: 100%; width: 0; border-radius: 9999px; transition: width 0.4s ease; }
.model-specs-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 14px;
    padding-top: 12px; border-top: 1px solid var(--border-light); }
.model-specs-grid.two { grid-template-columns: repeat(2, 1fr); margin-top: 12px; }
.spec-box { background: #FBF8F5; border: 1px solid #EFE5DD; border-radius: 10px; padding: 8px 6px; text-align: center; }
.spec-label { font-size: 0.68rem; font-weight: 600; color: var(--neutral-muted); text-transform: uppercase; margin-bottom: 2px; }
.spec-value { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.85rem; font-weight: 800; color: #261E1A; }
.spec-sub { font-size: 0.66rem; color: #0D9488; font-weight: 600; }
.section-label { font-size: 0.74rem; font-weight: 700; color: #574A43; text-transform: uppercase; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* Botón primario idéntico a div.stButton > button[kind="primary"] de app.py */
.btn-primary { width: 100%; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.90rem; border-radius: 10px;
    padding: 10px 22px; cursor: pointer; transition: all 0.2s ease; background-color: #0F172A; color: #FFFFFF;
    border: 1px solid #0F172A; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15); }
.btn-primary:hover { background-color: #1E293B; border-color: #1E293B; transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25); }
.btn-primary.stop { background-color: #FFFFFF; color: #1E293B; border: 1px solid #CBD5E1; box-shadow: none; }
.btn-primary:disabled { opacity: .6; cursor: wait; transform: none; }

#msg { font-size: 0.80rem; color: var(--neutral-body); margin: 8px 0 10px; min-height: 28px; }
.level-track { background: #FDFBFA; border: 1px solid #F5ECE5; border-radius: 12px; height: 50px; padding: 7px;
    display: flex; align-items: center; gap: 3px; }
.level-track > div { flex: 1; min-width: 3px; height: 6%; border-radius: 9999px;
    background: linear-gradient(180deg, #EA580C 0%, #F43F5E 55%, #9F1239 100%);
    transition: height .12s ease, background .4s ease, box-shadow .4s ease; }
#timeline { display: flex; gap: 3px; height: 20px; margin-top: 6px; }
#timeline > div { flex: 1; border-radius: 4px; background: #F3ECE6; }

/* AuraTerminal Live en WebSocket */
.ws-terminal {
    background: #140F0D;
    border: 1px solid #382E2B;
    border-radius: 10px;
    padding: 8px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    color: #E7DDD4;
    height: 125px;
    overflow-y: auto;
    line-height: 1.45;
    margin-top: 6px;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.4);
}
.ws-term-line { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ws-term-line.dim { color: #8F7D74; }
.ws-term-line .t-win { color: #F59E0B; font-weight: 600; }
.ws-term-line .t-emo { font-weight: 700; }
.ws-term-line .t-lat { color: #10B981; }
.ws-term-line .t-dev { color: #38BDF8; }
</style></head>
<body>
<div class="grid">
  <!-- ======================= CONTROL DE TRANSMISIÓN ======================= -->
  <div class="section-card">
    <div class="card-title-row">
      <div class="card-title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EA580C" stroke-width="2.5"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
        Control de Transmisión
      </div>
      <span id="badge" class="nav-badge"><span class="nav-badge-dot"></span><span id="badge-txt">En espera</span></span>
    </div>

    <button id="btn" class="btn-primary">Iniciar Streaming en Vivo</button>
    <p id="msg">El navegador pedirá permiso para usar el micrófono.</p>

    <div class="section-label" style="margin-bottom:6px;">Señal del micrófono</div>
    <div class="level-track" id="level"></div>

    <div class="model-specs-grid two">
      <div class="spec-box"><div class="spec-label">Ventanas</div><div class="spec-value mono" id="s-win">0</div><div class="spec-sub">3.0 s c/u · paso 1.0 s</div></div>
      <div class="spec-box"><div class="spec-label">Latencia media</div><div class="spec-value mono" id="s-avg">—</div><div class="spec-sub" id="s-last">última: —</div></div>
      <div class="spec-box"><div class="spec-label">Audio enviado</div><div class="spec-value mono" id="s-sent">0.0 s</div><div class="spec-sub">PCM Float32</div></div>
      <div class="spec-box"><div class="spec-label">Muestreo</div><div class="spec-value mono" id="s-sr">—</div><div class="spec-sub">remuestreo a 16 kHz</div></div>
    </div>

    <div class="section-label" style="margin-top:12px;">Historial (últimas 24 ventanas)</div>
    <div id="timeline"></div>

    <div class="section-label" style="margin-top:12px; display:flex; justify-content:space-between; align-items:center;">
      <span>AuraTerminal Live (WebSocket Stream)</span>
      <span id="term-status" style="font-size:0.67rem; color:#8F7D74; font-weight:700;">EN ESPERA</span>
    </div>
    <div class="ws-terminal" id="ws-term">
      <div class="ws-term-line dim">[SISTEMA] Pulsa Iniciar Streaming para observar el crono-análisis en tiempo real...</div>
    </div>
  </div>

  <!-- ======================= ESTADO EMOCIONAL EN VIVO ======================= -->
  <div class="section-card">
    <div class="card-title-row">
      <div class="card-title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EA580C" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
        Estado Emocional en Vivo
      </div>
      <div class="confidence-pill" id="conf">Confianza: —</div>
    </div>

    <div class="hero-emotion-banner">
      <div>
        <div class="emotion-name-large" id="pred">Esperando audio…</div>
        <div class="circumplex-vector" id="va">Valencia: — · Activación: —</div>
      </div>
      <div class="emotion-state-tag" id="tag">● Sin señal</div>
    </div>

    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <span class="section-label">Desglose de Probabilidades (Softmax Distribution)</span>
      <span style="font-size:0.72rem; color:#8F7D74; font-weight:600;" id="ts">9 Clases Oficiales</span>
    </div>
    <div id="bars"></div>

    <div class="model-specs-grid">
      <div class="spec-box"><div class="spec-label">Backbone</div><div class="spec-value">emotion2vec+</div><div class="spec-sub" id="d-params">— params</div></div>
      <div class="spec-box"><div class="spec-label">Inferencia</div><div class="spec-value mono" id="d-lat">--- ms</div><div class="spec-sub" id="d-fwd">por ventana</div></div>
      <div class="spec-box"><div class="spec-label">VRAM</div><div class="spec-value mono" id="d-vram">—</div><div class="spec-sub">torch.cuda</div></div>
      <div class="spec-box"><div class="spec-label">Dispositivo</div><div class="spec-value mono" id="d-dev">—</div><div class="spec-sub" id="d-gpu">—</div></div>
    </div>
  </div>
</div>

<script>
(() => {
  const PORT = __WS_PORT__;
  const CLASSES = __CLASSES__;
  const TARGET_SR = 16000;
  const SEND_SAMPLES = 4000;          // 250 ms a 16 kHz por trama
  const $ = (id) => document.getElementById(id);

  let host = "localhost";
  try { host = window.parent.location.hostname || "localhost"; } catch (e) {}
  const WS_URL = `ws://${host}:${PORT}`;

  let ws = null, ctx = null, stream = null, node = null, src = null, mute = null;
  let streaming = false, pending = [], pendingLen = 0, sentSamples = 0;
  let latencies = [], history = [];

  // ---------- Barras de probabilidad (mismo marcado que la pestaña Sesión en Vivo) ----------
  const bars = {};
  CLASSES.forEach(c => {
    const item = document.createElement("div");
    item.className = "prob-item";
    item.innerHTML = `<div class="prob-meta"><span class="prob-name">${c.name}</span><span class="prob-val">0.0%</span></div>` +
                     `<div class="prob-track"><div class="prob-fill" style="background-color:${c.color}"></div></div>`;
    $("bars").appendChild(item);
    bars[c.name] = { fill: item.querySelector(".prob-fill"), val: item.querySelector(".prob-val") };
  });
  for (let i = 0; i < 24; i++) $("timeline").appendChild(document.createElement("div"));
  const LEVEL_BARS = 40;
  for (let i = 0; i < LEVEL_BARS; i++) $("level").appendChild(document.createElement("div"));
  let levelHist = new Array(LEVEL_BARS).fill(0);

  function setBadge(text, kind) { $("badge").className = "nav-badge" + (kind ? " " + kind : ""); $("badge-txt").textContent = text; }
  function setMsg(t) { $("msg").textContent = t; }

  // ---------- Remuestreo a 16 kHz (promedio por bloques = filtro paso-bajo simple) ----------
  // Se guardan entre tramas las muestras sobrantes (tail) y la fase fraccionaria (frac),
  // para que el remuestreo sea continuo y sin deriva con tasas no enteras (p. ej. 44.1 kHz).
  let tail = new Float32Array(0), frac = 0;
  function downsample(input, inRate) {
    if (inRate === TARGET_SR) return new Float32Array(input);
    const ratio = inRate / TARGET_SR;
    const data = new Float32Array(tail.length + input.length);
    data.set(tail, 0); data.set(input, tail.length);
    const outLen = Math.max(0, Math.floor((data.length - frac) / ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const start = Math.floor(frac + i * ratio);
      const end = Math.min(data.length, Math.floor(frac + (i + 1) * ratio));
      let sum = 0, n = 0;
      for (let j = start; j < end; j++) { sum += data[j]; n++; }
      out[i] = n ? sum / n : data[Math.min(start, data.length - 1)];
    }
    const advance = frac + outLen * ratio;
    const consumed = Math.floor(advance);
    frac = advance - consumed;
    tail = data.slice(consumed);
    return out;
  }

  let levelAcc = 0, levelN = 0;
  function onFrame(frame) {
    // Medidor de nivel estilo forma de onda (RMS acumulado cada ~50 ms)
    for (let i = 0; i < frame.length; i++) levelAcc += frame[i] * frame[i];
    levelN += frame.length;
    if (levelN >= 2400) {
      levelHist.push(Math.min(1, Math.sqrt(levelAcc / levelN) * 6)); levelHist.shift();
      const cells = $("level").children;
      for (let i = 0; i < LEVEL_BARS; i++) cells[i].style.height = Math.max(6, levelHist[i] * 100) + "%";
      levelAcc = 0; levelN = 0;
    }

    const pcm = downsample(frame, ctx.sampleRate);
    pending.push(pcm); pendingLen += pcm.length;
    if (pendingLen >= SEND_SAMPLES && ws && ws.readyState === WebSocket.OPEN) {
      const packet = new Float32Array(pendingLen);
      let off = 0; pending.forEach(p => { packet.set(p, off); off += p.length; });
      pending = []; pendingLen = 0;
      ws.send(packet.buffer);
      sentSamples += packet.length;
      $("s-sent").textContent = (sentSamples / TARGET_SR).toFixed(1) + " s";
    }
  }

  // ---------- Captura de audio: AudioWorklet con respaldo a ScriptProcessor ----------
  async function startAudio() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("INSECURE");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true }
    });
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    $("s-sr").textContent = (ctx.sampleRate / 1000).toFixed(1) + " kHz";
    src = ctx.createMediaStreamSource(stream);
    mute = ctx.createGain(); mute.gain.value = 0;   // Evita oír el propio micrófono
    tail = new Float32Array(0); frac = 0;

    try {
      const code = `class Tap extends AudioWorkletProcessor {
        process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
      } registerProcessor('aura-tap', Tap);`;
      const url = URL.createObjectURL(new Blob([code], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      node = new AudioWorkletNode(ctx, "aura-tap");
      node.port.onmessage = (e) => onFrame(e.data);
    } catch (e) {
      node = ctx.createScriptProcessor(4096, 1, 1);   // Navegadores sin AudioWorklet
      node.onaudioprocess = (e) => onFrame(e.inputBuffer.getChannelData(0));
    }
    src.connect(node); node.connect(mute); mute.connect(ctx.destination);
  }

  function stopAudio() {
    try { if (src) src.disconnect(); if (node) node.disconnect(); } catch (e) {}
    try { if (stream) stream.getTracks().forEach(t => t.stop()); } catch (e) {}
    try { if (ctx && ctx.state !== "closed") ctx.close(); } catch (e) {}
    src = node = stream = ctx = null; pending = []; pendingLen = 0;
    levelHist = new Array(LEVEL_BARS).fill(0);
    Array.from($("level").children).forEach(c => {
      c.style.height = "6%";
      c.style.background = "linear-gradient(180deg, #EA580C 0%, #F43F5E 55%, #9F1239 100%)";
      c.style.boxShadow = "none";
    });
  }

  // ---------- Renderizado de resultados ----------
  const EMOTION_GRADIENTS = {
    "Alegría": "linear-gradient(180deg, #EA580C 0%, #F59E0B 55%, #C2410C 100%)",
    "Tristeza": "linear-gradient(180deg, #0D9488 0%, #14B8A6 55%, #0F766E 100%)",
    "Enojo": "linear-gradient(180deg, #EF4444 0%, #F43F5E 55%, #991B1B 100%)",
    "Miedo": "linear-gradient(180deg, #E11D48 0%, #BE123C 55%, #881337 100%)",
    "Disgusto": "linear-gradient(180deg, #D97706 0%, #B45309 55%, #78350F 100%)",
    "Neutral": "linear-gradient(180deg, #574A43 0%, #786C65 55%, #382E2B 100%)",
    "Sorpresa": "linear-gradient(180deg, #8B5CF6 0%, #7C3AED 55%, #5B21B6 100%)",
    "Otro": "linear-gradient(180deg, #64748B 0%, #475569 55%, #334155 100%)",
    "Neutro / Desc.": "linear-gradient(180deg, #475569 0%, #334155 55%, #1E293B 100%)"
  };

  const sign = (x) => (x >= 0 ? "+" : "") + x.toFixed(2);
  function render(d) {
    $("pred").textContent = d.predominant; $("pred").style.color = d.color;
    $("conf").textContent = `Confianza: ${d.confidence}%`;
    $("va").textContent = `Valencia: ${sign(d.valence)} · Activación: ${sign(d.arousal)}`;
    $("tag").textContent = "● " + (d.circumplex_title || d.predominant);
    $("tag").style.background = d.color;
    $("ts").textContent = `Ventana #${d.window} · ${d.timestamp}`;

    // Actualizar color de la onda acústica según el estado emocional capturado
    const activeGrad = EMOTION_GRADIENTS[d.predominant] || `linear-gradient(180deg, ${d.color} 0%, #292524 100%)`;
    const levelCells = $("level").children;
    for (let i = 0; i < LEVEL_BARS; i++) {
      levelCells[i].style.background = activeGrad;
      levelCells[i].style.boxShadow = `0 0 6px ${d.color}60`;
    }

    const byName = {}; d.breakdown.forEach(b => byName[b.name] = b.percent);
    CLASSES.forEach(c => {
      const p = byName[c.name] || 0;
      bars[c.name].fill.style.width = p + "%"; bars[c.name].val.textContent = p.toFixed(1) + "%";
    });

    latencies.push(d.latency_ms); if (latencies.length > 30) latencies.shift();
    const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    $("s-win").textContent = d.window;
    $("s-avg").textContent = avg.toFixed(0) + " ms";
    $("s-last").textContent = `última: ${d.latency_ms} ms`;
    $("d-lat").textContent = d.latency_ms.toFixed(0) + " ms";
    if (d.model_forward_ms != null) $("d-fwd").textContent = `modelo: ${d.model_forward_ms.toFixed(0)} ms`;
    $("d-dev").textContent = d.device === "cpu" ? "CPU" : d.device;
    $("d-gpu").textContent = d.device === "cpu" ? "Procesador" : d.gpu_name.replace("NVIDIA GeForce ", "");
    $("d-vram").textContent = d.device === "cpu" ? "n/a" : d.vram_mb.toFixed(0) + " MiB";
    pushHistory(d.color, `${d.timestamp} ${d.predominant} ${d.confidence}%`);
    appendTermLine(d);
  }

  function appendTermLine(d) {
    const term = $("ws-term");
    if (!term) return;
    const line = document.createElement("div");
    line.className = "ws-term-line";
    const ts = d.timestamp || new Date().toTimeString().split(" ")[0];
    const dev = (d.device === "cpu" ? "CPU" : (d.device || "CUDA:0")).toUpperCase();
    line.innerHTML = `<span style="color:#8F7D74;">[${ts}]</span> ` +
      `<span class="t-dev">[${dev}]</span> ` +
      `<span class="t-win">win #${d.window}</span>: ` +
      `<span class="t-emo" style="color:${d.color};">${d.predominant} (${d.confidence}%)</span> | ` +
      `<span class="t-lat">${d.latency_ms.toFixed(1)} ms</span> | ` +
      `<span style="color:#A8A29E;">RMS: ${(d.rms || 0).toFixed(4)}</span>`;
    term.appendChild(line);
    while (term.children.length > 50) term.removeChild(term.firstChild);
    term.scrollTop = term.scrollHeight;
  }

  function appendTermText(txt, color) {
    const term = $("ws-term");
    if (!term) return;
    const line = document.createElement("div");
    line.className = "ws-term-line";
    if (color) line.style.color = color;
    line.textContent = txt;
    term.appendChild(line);
    while (term.children.length > 50) term.removeChild(term.firstChild);
    term.scrollTop = term.scrollHeight;
  }

  function pushHistory(color, title) {
    history.push({ color, title }); if (history.length > 24) history.shift();
    const cells = $("timeline").children;
    for (let i = 0; i < 24; i++) {
      const h = history[i - (24 - history.length)];
      cells[i].style.background = h ? h.color : "#F3ECE6"; cells[i].title = h ? h.title : "";
    }
  }

  // ---------- Conexión WebSocket ----------
  function connect() {
    return new Promise((resolve, reject) => {
      let opened = false;
      try { ws = new WebSocket(WS_URL); } catch (e) { reject(e); return; }
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        opened = true;
        if ($("term-status")) { $("term-status").textContent = "ACTIVO"; $("term-status").style.color = "#10B981"; }
        appendTermText(`[WS] Handshake WebSocket OK: conectado a ${WS_URL}`, "#10B981");
        resolve();
      };
      ws.onmessage = (ev) => {
        let d; try { d = JSON.parse(ev.data); } catch (e) { return; }
        if (d.status === "OK") render(d);
        else if (d.status === "READY") {
          $("d-dev").textContent = d.device;
          if (d.params_str) $("d-params").textContent = `${d.params_str} · ${d.dtype}`;
          appendTermText(`[READY] Modelo emotion2vec+ Large listo en ${d.device} | ${d.params_str || "?"} | ${d.dtype || "?"}`, "#38BDF8");
        } else if (d.status === "SILENCE") {
          $("ts").textContent = `Silencio detectado · ventana #${d.window}`;
          pushHistory("#E7DDD4", `${d.timestamp} silencio`);
          appendTermText(`[${d.timestamp}] [VAD] Silencio detectado en ventana #${d.window} (RMS < 0.004)`, "#8F7D74");
        } else if (d.status === "ERROR") {
          setMsg("Error de inferencia en el servidor: " + d.message);
          appendTermText(`[ERROR] Inferencia fallida: ${d.message}`, "#EF4444");
        }
      };
      ws.onerror = () => {
        if (!opened) reject(new Error("WS"));
        if ($("term-status")) { $("term-status").textContent = "ERROR"; $("term-status").style.color = "#EF4444"; }
      };
      ws.onclose = () => {
        if ($("term-status")) { $("term-status").textContent = "DESCONECTADO"; $("term-status").style.color = "#8F7D74"; }
        if (streaming) {
          stop(false);
          setBadge("Conexión perdida", "error");
          setMsg("Se perdió la conexión con el servidor. Verifica que python src/websocket_server.py siga activo y vuelve a iniciar.");
        }
      };
    });
  }

  async function start() {
    $("btn").disabled = true; setMsg("Conectando con el servidor y solicitando el micrófono…");
    appendTermText("[SISTEMA] Iniciando handshake WebSocket y captura de audio...", "#F59E0B");
    try {
      await connect();
    } catch (e) {
      $("btn").disabled = false;
      setBadge("Servidor no disponible", "error");
      setMsg(`No se pudo conectar a ${WS_URL}. Inicia el servidor con: uv run python src/websocket_server.py`);
      appendTermText(`[ERROR] No se pudo conectar a ${WS_URL}`, "#EF4444");
      return;
    }
    try {
      await startAudio();
    } catch (e) {
      try { ws.close(); } catch (_) {}
      stopAudio(); $("btn").disabled = false;
      setBadge("Micrófono bloqueado", "error");
      if (e.message === "INSECURE") setMsg("El navegador solo permite el micrófono en https o en http://localhost. Abre la app como http://localhost:8501.");
      else if (e.name === "NotAllowedError") setMsg("Permiso de micrófono denegado. Haz clic en el candado de la barra de direcciones y permite el micrófono.");
      else if (e.name === "NotFoundError") setMsg("No se encontró ningún micrófono conectado.");
      else setMsg("No se pudo iniciar el micrófono: " + (e.message || e.name));
      appendTermText(`[MIC:ERROR] ${e.message || e.name}`, "#EF4444");
      return;
    }
    streaming = true; sentSamples = 0; latencies = [];
    try { ws.send("RESET"); } catch (e) {}
    $("btn").disabled = false; $("btn").textContent = "Detener Streaming"; $("btn").classList.add("stop");
    setBadge("Transmitiendo en vivo", "live");
    setMsg("Enviando audio a 16 kHz. El primer veredicto aparece tras 3 s de audio.");
    appendTermText("[STREAM] Captura activa: tramas de 250ms PCM Float32 enviadas", "#10B981");
  }

  function stop(userAction = true) {
    streaming = false; stopAudio();
    try { if (ws && ws.readyState <= 1) ws.close(); } catch (e) {}
    $("btn").textContent = "Iniciar Streaming en Vivo"; $("btn").classList.remove("stop");
    if (userAction) {
      setBadge("Detenido", "");
      setMsg("Transmisión finalizada.");
      appendTermText("[SISTEMA] Transmisión detenida por el usuario.", "#8F7D74");
      if ($("term-status")) { $("term-status").textContent = "DETENIDO"; $("term-status").style.color = "#8F7D74"; }
    }
  }

  $("btn").onclick = () => (streaming ? stop(true) : start());
  window.addEventListener("beforeunload", () => stop(false));
})();
</script>
</body></html>
"""
