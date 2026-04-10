// ---- DOM refs ----
const logBox = document.getElementById("logBox");
const statusDot = document.getElementById("statusDot");
const statusTxt = document.getElementById("statusText");
const riskValue = document.getElementById("riskValue");
const gaugeFill = document.getElementById("gaugeFill");
const gaugeLabel = document.querySelector(".gauge-label");

const CIRCUMFERENCE = 314.16; // 2 * π * 50

// --- Session & Graph Memory ---
let isSessionActive = false;
let sessionInterval;
let sessionStartTime = null;

const historyTime = [];
const historyScore = [];
const historyEvents = []; // store raw event codes for per-activity datasets

const stats = { reading: 0, talking: 0, looking_away: 0, tab_switch: 0, no_face: 0, crowd: 0 };

const COLOR_MAP = {
  normal: "#22c55e",
  decay: "#22c55e",
  reading: "#ef4444",
  talking: "#f97316",
  looking_away: "#a855f7",
  tab_switch: "#ffffff",
  no_face: "#64748b",
  crowd: "#4d44ef",
};

const LABEL_MAP = {
  reading: "Reading",
  talking: "Speaking",
  looking_away: "Looking Away",
  tab_switch: "Tab Switch",
  no_face: "No Face",
  crowd: "Multiple Faces",
};

// ---- Frame rate: 12 FPS (83ms interval) ----
const FRAME_INTERVAL = 125;

// ---- Helpers ----
function log(msg) {
  const t = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML = `<span class="time">${t}</span>${msg}`;
  logBox.appendChild(entry);
  logBox.scrollTop = logBox.scrollHeight;
}

function setRisk(score) {
  const clamped = Math.max(0, Math.min(100, score));
  riskValue.textContent = Math.round(clamped);
  const offset = CIRCUMFERENCE - (clamped / 100) * CIRCUMFERENCE;
  gaugeFill.style.strokeDashoffset = offset;

  if (clamped < 30) {
    gaugeFill.style.stroke = "#22c55e";
    gaugeLabel.textContent = "Low Risk";
  } else if (clamped < 70) {
    gaugeFill.style.stroke = "#eab308";
    gaugeLabel.textContent = "Medium Risk";
  } else {
    gaugeFill.style.stroke = "#ef4444";
    gaugeLabel.textContent = "High Risk";
  }
}

// ---- Webcam ----
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    document.getElementById("cam").srcObject = stream;
    log("Camera access granted");
    return true;
  } catch (err) {
    log("Camera error: " + err.message);
    return false;
  }
}

// ---- WebSocket (returns a Promise that resolves when connected) ----
let ws;
function connectWS() {
  return new Promise((resolve, reject) => {
    // Automatically uses wss:// (secure) if hosted, or ws:// if local
    const protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(protocol + window.location.host + "/ws");

    ws.addEventListener("open", () => {
      statusDot.classList.add("connected");
      statusTxt.textContent = "Connected";
      log("WebSocket connected to server");
      ws.send(JSON.stringify({ event: "connected" }));
      resolve();
    });

    ws.addEventListener("message", (event) => {
      try {
        const data = JSON.parse(event.data);
        setRisk(data.risk_score);

        if (data.message.includes("WARNING") || data.message.includes("System:")) {
          log("Server: " + data.message);
        }

        if (data.type === "vision_update" && data.vision_data) {
          drawVision(data.vision_data, data.vision_type);
        }

        // --- Track Graph History using real elapsed time ---
        if (isSessionActive && data.event_code) {
          const elapsed = ((Date.now() - sessionStartTime) / 1000).toFixed(1);
          historyTime.push(elapsed + "s");
          historyScore.push(data.risk_score);
          historyEvents.push(data.event_code);

          if (data.event_code === "reading") stats.reading++;
          if (data.event_code === "talking") stats.talking++;
          if (data.event_code === "looking_away") stats.looking_away++;
          if (data.event_code === "tab_switch") stats.tab_switch++;
          if (data.event_code === "no_face") stats.no_face++;
          if (data.event_code === "crowd") stats.crowd++;
        }
      } catch {
        log("Server: " + event.data);
      }
    });

    ws.addEventListener("close", () => {
      statusDot.classList.remove("connected");
      statusTxt.textContent = "Disconnected";
    });

    ws.addEventListener("error", () => {
      log("WebSocket connection failed.");
      reject(new Error("WebSocket failed"));
    });
  });
}

// ---- Browser Event Proctoring ----
document.addEventListener("visibilitychange", () => {
  if (ws && ws.readyState === WebSocket.OPEN && isSessionActive) {
    if (document.hidden) {
      ws.send(JSON.stringify({ event: "tab_switch" }));
    } else {
      ws.send(JSON.stringify({ event: "tab_focus" }));
    }
  }
});

window.addEventListener("blur", () => {
  if (ws && ws.readyState === WebSocket.OPEN && isSessionActive) {
    ws.send(JSON.stringify({ event: "window_blur" }));
  }
});

window.addEventListener("focus", () => {
  if (ws && ws.readyState === WebSocket.OPEN && isSessionActive) {
    ws.send(JSON.stringify({ event: "window_focus" }));
  }
});

// ---- Start / Stop Controls ----
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const videoElement = document.getElementById("cam");

startBtn.addEventListener("click", async () => {
  // Disable start button immediately
  startBtn.disabled = true;
  startBtn.style.background = "#3f3f46";
  startBtn.style.color = "#a1a1aa";
  startBtn.style.cursor = "not-allowed";

  log("Initializing camera...");

  // Step 1: Initialize camera, wait for it
  const cameraReady = await initCamera();
  if (!cameraReady) {
    startBtn.disabled = false;
    startBtn.style.background = "#22c55e";
    startBtn.style.color = "white";
    startBtn.style.cursor = "pointer";
    log("Session aborted — camera not available.");
    return;
  }

  // Step 2: Connect WebSocket, wait for open
  try {
    await connectWS();
  } catch {
    startBtn.disabled = false;
    startBtn.style.background = "#22c55e";
    startBtn.style.color = "white";
    startBtn.style.cursor = "pointer";
    log("Session aborted — server not reachable.");
    return;
  }

  // Step 3: Both ready — NOW start the session
  isSessionActive = true;
  sessionStartTime = Date.now();

  // Enable stop button
  stopBtn.disabled = false;
  stopBtn.style.background = "#ef4444";
  stopBtn.style.color = "white";
  stopBtn.style.cursor = "pointer";

  log("SESSION STARTED — Monitoring at 12 FPS");

  // Step 4: Start sending frames
  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d");

  sessionInterval = setInterval(() => {
    if (
      ws &&
      ws.readyState === WebSocket.OPEN &&
      videoElement.readyState === videoElement.HAVE_ENOUGH_DATA
    ) {
      captureCanvas.width = videoElement.videoWidth;
      captureCanvas.height = videoElement.videoHeight;
      captureCtx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
      const base64Image = captureCanvas.toDataURL("image/jpeg", 0.5);

      const isBackground = document.hidden || !document.hasFocus();

      ws.send(
        JSON.stringify({
          event: "frame",
          image: base64Image,
          frame_interval: FRAME_INTERVAL,
          is_background: isBackground,
        })
      );
    }
  }, FRAME_INTERVAL);
});

stopBtn.addEventListener("click", () => {
  isSessionActive = false;
  log("SESSION STOPPED. Generating Report...");
  clearInterval(sessionInterval);

  if (ws) ws.close();

  // Stop the camera
  const stream = videoElement.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    videoElement.srcObject = null;
  }

  // Populate report stats
  document.getElementById("statRead").innerText = stats.reading;
  document.getElementById("statSpeak").innerText = stats.talking;
  document.getElementById("statLook").innerText = stats.looking_away;
  document.getElementById("statTab").innerText = stats.tab_switch;

  // Show report modal
  document.getElementById("reportModal").style.display = "flex";

  // --- Build per-activity datasets for the audit chart ---
  // Each violation type gets its own line so colors are meaningful in the legend
  const violationTypes = ["reading", "talking", "looking_away", "tab_switch", "no_face", "crowd"];

  // Base risk score line (always visible, thin gray)
  const datasets = [
    {
      label: "Risk Score",
      data: [...historyScore],
      borderColor: "#4b5563",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      order: 1, // draw behind violation markers
    },
  ];

  // One scatter dataset per violation type — only plot points where that event occurred
  violationTypes.forEach((evtType) => {
    const points = [];
    let hasAny = false;
    for (let i = 0; i < historyEvents.length; i++) {
      if (historyEvents[i] === evtType) {
        points.push(historyScore[i]);
        hasAny = true;
      } else {
        points.push(null);
      }
    }
    if (hasAny) {
      datasets.push({
        label: LABEL_MAP[evtType] || evtType,
        data: points,
        borderColor: "transparent",
        borderWidth: 0,
        pointBackgroundColor: COLOR_MAP[evtType],
        pointBorderColor: COLOR_MAP[evtType],
        pointRadius: historyScore.length > 300 ? 2 : 4,
        spanGaps: false,
        showLine: false,
        order: 0, // draw on top
      });
    }
  });

  const ctx = document.getElementById("auditChart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: historyTime,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          grid: { color: "#27272a" },
          ticks: { color: "#a1a1aa" },
        },
        x: {
          grid: { display: false },
          ticks: {
            maxTicksLimit: 20,
            maxRotation: 0,
            color: "#a1a1aa",
          },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: {
            color: "#a1a1aa",
            usePointStyle: true,
            pointStyle: "circle",
            padding: 16,
            font: { size: 11 },
          },
        },
      },
    },
  });
});

// ---- AI Vision Overlay ----
const aiToggle = document.getElementById("aiToggle");
const overlayCanvas = document.getElementById("overlay");
const overlayCtx = overlayCanvas.getContext("2d");
let showVision = false;

aiToggle.addEventListener("change", (e) => {
  showVision = e.target.checked;
  if (!showVision) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }
});

function drawVision(visionData, visionType) {
  if (!showVision) return;

  overlayCanvas.width = videoElement.videoWidth;
  overlayCanvas.height = videoElement.videoHeight;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (visionType === "boxes") {
    overlayCtx.strokeStyle = "#ef4444";
    overlayCtx.lineWidth = 4;
    visionData.forEach((box) => {
      const x = box.xmin * overlayCanvas.width;
      const y = box.ymin * overlayCanvas.height;
      const w = box.width * overlayCanvas.width;
      const h = box.height * overlayCanvas.height;
      overlayCtx.strokeRect(x, y, w, h);
    });
  } else if (visionType === "mesh") {
    overlayCtx.fillStyle = "#22c55e";
    visionData.forEach((point) => {
      const x = point.x * overlayCanvas.width;
      const y = point.y * overlayCanvas.height;
      overlayCtx.beginPath();
      overlayCtx.arc(x, y, 1.5, 0, 2 * Math.PI);
      overlayCtx.fill();
    });
  }
}