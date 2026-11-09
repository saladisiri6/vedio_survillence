// ============================================================
// SENTINEL dashboard client logic
// ============================================================

const state = {
  mode: "upload",
  scores: [],          // rolling score history for the waveform
  threshold: null,
  webcamStream: null,
  webcamTimer: null,
  ws: null,
  rtspPolling: null,
};

// ---------------- clock ----------------
function tickClock() {
  const el = document.getElementById("clock");
  el.textContent = new Date().toTimeString().split(" ")[0];
}
setInterval(tickClock, 1000);
tickClock();

// ---------------- health check ----------------
async function checkHealth() {
  const statusEl = document.getElementById("sysStatusText");
  const recDot = document.getElementById("recDot");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "ok") {
      statusEl.textContent = "MODEL ONLINE";
      recDot.classList.add("live");
      state.threshold = data.threshold;
      document.getElementById("thresholdValue").textContent = data.threshold.toFixed(4);
    } else {
      statusEl.textContent = "MODEL UNAVAILABLE";
    }
  } catch (e) {
    statusEl.textContent = "SERVER UNREACHABLE";
  }
}
checkHealth();

// ---------------- tabs ----------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const mode = btn.dataset.mode;
    document.querySelector(`.tab-panel[data-panel="${mode}"]`).classList.add("active");
    switchMode(mode);
  });
});

function switchMode(mode) {
  // Tear down whatever the previous mode was doing.
  stopWebcam();
  stopRtspPolling();
  state.mode = mode;

  const label = document.getElementById("sourceLabel");
  const vfContent = document.getElementById("vfContent");
  document.getElementById("webcamVideo").style.display = "none";

  if (mode === "upload") {
    label.textContent = "FILE UPLOAD";
    vfContent.innerHTML = '<p class="vf-placeholder">SELECT A VIDEO FILE BELOW</p>';
  } else if (mode === "webcam") {
    label.textContent = "WEBCAM // STANDBY";
    vfContent.innerHTML = "";
  } else if (mode === "rtsp") {
    label.textContent = "RTSP // DISCONNECTED";
    vfContent.innerHTML = '<p class="vf-placeholder">ENTER STREAM URL AND CONNECT</p>';
  }
  resetStatus();
}

function resetStatus() {
  state.scores = [];
  drawWaveform();
  setStatus("standby");
  document.getElementById("scoreValue").textContent = "--";
}

// ---------------- status badge ----------------
function setStatus(kind, score, extraLabel) {
  const badge = document.getElementById("statusBadge");
  const text = document.getElementById("statusText");
  const flash = document.getElementById("alertFlash");
  badge.className = "status-badge";
  flash.classList.remove("active");

  if (kind === "standby") {
    text.textContent = "STANDBY";
  } else if (kind === "buffering") {
    badge.classList.add("buffering");
    text.textContent = "BUFFERING...";
  } else if (kind === "normal") {
    badge.classList.add("normal");
    text.textContent = "NORMAL";
  } else if (kind === "anomaly") {
    badge.classList.add("anomaly");
    text.textContent = extraLabel || "ANOMALY DETECTED";
    flash.classList.add("active");
  }

  if (score !== undefined) {
    document.getElementById("scoreValue").textContent = score.toFixed(5);
  }
}

// ---------------- waveform ----------------
function drawWaveform() {
  const canvas = document.getElementById("waveform");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "#1a2b1f";
  ctx.lineWidth = 1;
  for (let gx = 0; gx < w; gx += 20) {
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
  }

  if (state.scores.length < 2) return;

  const maxScore = Math.max(...state.scores, state.threshold || 0.001) * 1.2;
  const stepX = w / 59;

  // threshold line
  if (state.threshold) {
    const ty = h - (state.threshold / maxScore) * h;
    ctx.strokeStyle = "#FFB238";
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(w, ty); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.beginPath();
  ctx.strokeStyle = "#3CFF7A";
  ctx.lineWidth = 1.5;
  ctx.shadowColor = "#3CFF7A";
  ctx.shadowBlur = 4;
  state.scores.forEach((s, i) => {
    const x = i * stepX;
    const y = h - (s / maxScore) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function pushScore(score) {
  state.scores.push(score);
  if (state.scores.length > 60) state.scores.shift();
  drawWaveform();
}

// ============================================================
// UPLOAD MODE
// ============================================================
const fileInput = document.getElementById("fileInput");
const fileDropText = document.getElementById("fileDropText");
const fileDrop = document.querySelector(".file-drop");
const analyzeBtn = document.getElementById("analyzeBtn");
let selectedFile = null;

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    selectedFile = fileInput.files[0];
    fileDropText.textContent = selectedFile.name.toUpperCase();
    fileDrop.classList.add("has-file");
    analyzeBtn.disabled = false;

    // Play the actual uploaded video in the viewfinder panel.
    const url = URL.createObjectURL(selectedFile);
    const vfContent = document.getElementById("vfContent");
    vfContent.innerHTML = "";
    const videoEl = document.createElement("video");
    videoEl.id = "uploadVideo";
    videoEl.src = url;
    videoEl.controls = true;
    videoEl.autoplay = true;
    videoEl.loop = true;
    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.style.width = "100%";
    videoEl.style.height = "100%";
    videoEl.style.objectFit = "contain";
    vfContent.appendChild(videoEl);
    videoEl.play().catch(() => {
      // Autoplay was blocked by the browser -- controls are visible so the
      // user can just press play manually.
    });
    document.getElementById("sourceLabel").textContent = `FILE: ${selectedFile.name.toUpperCase()}`;
  }
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "ANALYZING...";
  setStatus("buffering");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderTimeline(data.timeline);

    const anyAnomaly = data.anomalies_detected > 0;
    const label = anyAnomaly
      ? `${data.anomalies_detected} OF ${data.clips_analyzed} CLIPS FLAGGED`
      : "NORMAL";
    setStatus(anyAnomaly ? "anomaly" : "normal", undefined, label);
    document.getElementById("sourceLabel").textContent = `FILE: ${data.filename.toUpperCase()}`;

    state.scores = data.timeline.map((r) => r.score).slice(-60);
    drawWaveform();
  } catch (e) {
    setStatus("standby");
    alert("Analysis failed: " + e.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "ANALYZE FOOTAGE";
  }
});

function renderTimeline(timeline) {
  const wrap = document.getElementById("timelineWrap");
  const el = document.getElementById("timeline");
  el.innerHTML = "";
  timeline.forEach((r) => {
    const div = document.createElement("div");
    div.className = "timeline-clip" + (r.is_anomaly ? " anomaly" : "");
    div.textContent = `${r.timestamp_sec}s`;
    div.title = `score: ${r.score.toFixed(5)}`;
    el.appendChild(div);
  });
  wrap.style.display = "block";
}

// ============================================================
// WEBCAM MODE
// ============================================================
const webcamVideo = document.getElementById("webcamVideo");
const webcamCanvas = document.getElementById("webcamCanvas");

document.getElementById("webcamStartBtn").addEventListener("click", startWebcam);
document.getElementById("webcamStopBtn").addEventListener("click", stopWebcam);

async function startWebcam() {
  try {
    state.webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (e) {
    alert("Could not access webcam: " + e.message);
    return;
  }
  webcamVideo.srcObject = state.webcamStream;
  webcamVideo.style.display = "block";
  document.getElementById("vfContent").innerHTML = "";
  document.getElementById("vfContent").appendChild(webcamVideo);
  document.getElementById("sourceLabel").textContent = "WEBCAM // LIVE";
  document.getElementById("webcamStartBtn").style.display = "none";
  document.getElementById("webcamStopBtn").style.display = "block";
  setStatus("buffering");

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  state.ws = new WebSocket(`${proto}//${location.host}/ws/webcam`);
  state.ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    if (data.status === "buffering") {
      setStatus("buffering");
      return;
    }
    pushScore(data.score);
    setStatus(data.is_anomaly ? "anomaly" : "normal", data.score);
  };

  state.webcamTimer = setInterval(() => {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    const ctx = webcamCanvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0, 128, 128);
    const dataUrl = webcamCanvas.toDataURL("image/jpeg", 0.7);
    state.ws.send(dataUrl);
  }, 200); // ~5 fps sampling into the sliding window
}

function stopWebcam() {
  if (state.webcamTimer) { clearInterval(state.webcamTimer); state.webcamTimer = null; }
  if (state.ws) { state.ws.close(); state.ws = null; }
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach((t) => t.stop());
    state.webcamStream = null;
  }
  document.getElementById("webcamStartBtn").style.display = "block";
  document.getElementById("webcamStopBtn").style.display = "none";
}

// ============================================================
// RTSP MODE
// ============================================================
document.getElementById("rtspStartBtn").addEventListener("click", startRtsp);
document.getElementById("rtspStopBtn").addEventListener("click", stopRtsp);

async function startRtsp() {
  const url = document.getElementById("rtspUrl").value.trim();
  if (!url) { alert("Enter an RTSP URL first."); return; }

  await fetch("/api/rtsp/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });

  document.getElementById("sourceLabel").textContent = "RTSP // CONNECTING";
  document.getElementById("rtspStartBtn").style.display = "none";
  document.getElementById("rtspStopBtn").style.display = "block";
  setStatus("buffering");

  state.rtspPolling = setInterval(pollRtsp, 500);
}

async function pollRtsp() {
  try {
    const res = await fetch("/api/rtsp/status");
    const data = await res.json();
    if (data.status === "ok") {
      document.getElementById("sourceLabel").textContent = "RTSP // LIVE";
      pushScore(data.score);
      setStatus(data.is_anomaly ? "anomaly" : "normal", data.score);
    } else if (data.status === "connected") {
      document.getElementById("sourceLabel").textContent = "RTSP // LIVE";
      setStatus("buffering");
    } else if (data.status === "error") {
      document.getElementById("sourceLabel").textContent = "RTSP // ERROR";
    }
  } catch (e) {
    // ignore transient poll errors
  }
}

async function stopRtsp() {
  await fetch("/api/rtsp/stop", { method: "POST" });
  stopRtspPolling();
  document.getElementById("sourceLabel").textContent = "RTSP // DISCONNECTED";
  document.getElementById("rtspStartBtn").style.display = "block";
  document.getElementById("rtspStopBtn").style.display = "none";
}

function stopRtspPolling() {
  if (state.rtspPolling) { clearInterval(state.rtspPolling); state.rtspPolling = null; }
}