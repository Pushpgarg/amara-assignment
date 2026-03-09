// ---- DOM refs ----
const logBox     = document.getElementById("logBox");
const statusDot  = document.getElementById("statusDot");
const statusTxt  = document.getElementById("statusText");
const riskValue  = document.getElementById("riskValue");
const gaugeFill  = document.getElementById("gaugeFill");
const gaugeLabel = document.querySelector(".gauge-label");

const CIRCUMFERENCE = 314.16; // 2 * π * 50

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
  riskValue.textContent = clamped;
  const offset = CIRCUMFERENCE - (clamped / 100) * CIRCUMFERENCE;
  gaugeFill.style.strokeDashoffset = offset;

  if (clamped < 30)       { gaugeFill.style.stroke = "#22c55e"; gaugeLabel.textContent = "Low Risk"; }
  else if (clamped < 70)  { gaugeFill.style.stroke = "#eab308"; gaugeLabel.textContent = "Medium Risk"; }
  else                    { gaugeFill.style.stroke = "#ef4444"; gaugeLabel.textContent = "High Risk"; }
}

// ---- Webcam ----
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    document.getElementById("cam").srcObject = stream;
    log("Camera access granted");
  } catch (err) {
    log("Camera error: " + err.message);
  }
}

// ---- WebSocket ----
let ws;
function connectWS() {
  ws = new WebSocket("ws://localhost:8000/ws");

  ws.addEventListener("open", () => {
    statusDot.classList.add("connected");
    statusTxt.textContent = "Connected";
    log("WebSocket connected to server");
    // UPDATED: Sending JSON instead of plain text
    ws.send(JSON.stringify({ event: "connected" }));
  });

  ws.addEventListener("message", (event) => {
    try {
      const data = JSON.parse(event.data);
      log("Server: " + data.message);
      setRisk(data.risk_score);
    } catch {
      log("Server: " + event.data);
    }
  });

  ws.addEventListener("close", () => {
    statusDot.classList.remove("connected");
    statusTxt.textContent = "Disconnected";
    log("WebSocket closed — retrying in 3s");
    setTimeout(connectWS, 3000);
  });

  ws.addEventListener("error", () => {
    log("WebSocket error");
  });
}

// ---- Browser Event Proctoring ----

// 1. Detect if the user switches tabs or minimizes the browser
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    // The user can't see the page anymore!
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ event: "tab_switch" }));
    }
  } else {
    log("System: User returned to the interview tab.");
  }
});

// 2. Detect if the user clicks outside the window (e.g., opening a notes app)
window.addEventListener("blur", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ event: "window_blur" }));
  }
});

// ---- Boot ----
initCamera();
connectWS();

// ---- Frame Extraction & Transmission ----

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");
const videoElement = document.getElementById("cam");

// Capture and send a frame every 1 second (1000 ms)
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN && videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
    
    // Match the canvas size to the video stream
    captureCanvas.width = videoElement.videoWidth;
    captureCanvas.height = videoElement.videoHeight;
    
    // Draw the current video frame onto the canvas
    captureCtx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
    
    // Convert the canvas to a lightweight JPEG Base64 string (quality: 0.5)
    const base64Image = captureCanvas.toDataURL("image/jpeg", 0.5);
    
    // Send it to the Python server
    ws.send(JSON.stringify({
      event: "frame",
      image: base64Image
    }));
  }
}, 1000);