from curses.ascii import TAB
from tkinter.tix import WINDOW

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import base64
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI(title="AI Proctor - Phase 0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Initialize MediaPipe Models ---
# 1. Bounding Box Detector (Fast, counts people)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 2. FaceMesh Detector (Heavy, extracts 468 points)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    risk_score = 0.0  
    is_in_background = False # Tracks if they are on another tab/app
    
    # ⭐️ TUNING VARIABLES (Points per second) ⭐️
    PENALTY_NO_FACE = 5.0      # Slower increase if face is missing
    PENALTY_CROWD = 10.0       # Moderate increase for multiple people
    DECAY_GOOD_BEHAVIOR = 0.5  # Very slow forgiveness rate
    TAB_SWITCH_PENALTY = 5    # Immediate penalty for tab switching
    WINDOW_BLUR_PENALTY = 5.0     # Penalty for losing window focus
    PENALTY_BACKGROUND = 10.0
    PENALTY_TALKING = 10.0      # Continuous penalty for speaking
    TALKING_THRESHOLD = 0.03    # Ratio of mouth opening to face height (0.03+ usually means talking)
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                event_type = payload.get("event")

                # --- Browser Event State Management ---
                if event_type == "tab_switch":
                    is_in_background = True
                    risk_score = min(100.0, risk_score + TAB_SWITCH_PENALTY)
                    msg = "WARNING: Tab switch detected!"
                elif event_type == "window_blur":
                    is_in_background = True
                    risk_score = min(100.0, risk_score + WINDOW_BLUR_PENALTY)
                    msg = "WARNING: Window lost focus!"
                elif event_type in ["tab_focus", "window_focus"]:
                    is_in_background = False
                    msg = "System: Candidate returned to interview."
                    # Send a quick update to log their return
                    response = {"status": "connected", "risk_score": risk_score, "message": msg}
                    await ws.send_text(json.dumps(response))
                    continue
                elif event_type == "connected":
                    msg = "Monitoring started."
                
                # --- Vision Pipeline ---
                elif event_type == "frame":
                    image_data = payload.get("image", "")
                    frame_interval = payload.get("frame_interval", 1000)
                    time_scale = frame_interval / 1000.0
                    
                    if "," in image_data:
                        base64_str = image_data.split(",")[1]
                        img_bytes = base64.b64decode(base64_str)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        results_detection = face_detection.process(img_rgb)
                        face_count = 0
                        bounding_boxes = []
                        
                        if results_detection.detections:
                            face_count = len(results_detection.detections)
                            for detection in results_detection.detections:
                                bbox = detection.location_data.relative_bounding_box
                                bounding_boxes.append({
                                    "xmin": bbox.xmin, "ymin": bbox.ymin,
                                    "width": bbox.width, "height": bbox.height
                                })
                        
                        msg = "Normal behavior."
                        vision_data = []
                        vision_type = "none"

                        if face_count == 0:
                            risk_score = min(100.0, risk_score + (PENALTY_NO_FACE * time_scale))
                            msg = "WARNING: Candidate not found!"
                            
                        elif face_count == 1:
                            mesh_results = face_mesh.process(img_rgb)
                            
                            if mesh_results.multi_face_landmarks:
                                landmarks = mesh_results.multi_face_landmarks[0].landmark
                                
                                # --- NEW: Lip Movement Math ---
                                # 13 is inner top lip, 14 is inner bottom lip
                                # 10 is top of head, 152 is bottom of chin
                                mouth_dist = landmarks[14].y - landmarks[13].y
                                face_height = landmarks[152].y - landmarks[10].y
                                
                                # Prevent division by zero just in case
                                mouth_ratio = mouth_dist / face_height if face_height > 0 else 0
                                is_talking = mouth_ratio > TALKING_THRESHOLD
                                
                                # --- State Hierarchy ---
                                if is_in_background:
                                    risk_score = min(100.0, risk_score + (PENALTY_BACKGROUND * time_scale))
                                    msg = "WARNING: Candidate is on another tab!"
                                elif is_talking:
                                    risk_score = min(100.0, risk_score + (PENALTY_TALKING * time_scale))
                                    msg = "WARNING: Speaking/Lip movement detected!"
                                elif risk_score > 0:
                                    # Only decay if they are alone, focused, AND quiet
                                    risk_score = max(0.0, risk_score - (DECAY_GOOD_BEHAVIOR * time_scale))
                                    
                                # Extract points to draw the green mesh on the frontend
                                for landmark in landmarks:
                                    vision_data.append({"x": landmark.x, "y": landmark.y})
                            
                            vision_type = "mesh"
                            
                        elif face_count > 1:
                            risk_score = min(100.0, risk_score + (PENALTY_CROWD * time_scale))
                            msg = f"WARNING: {face_count} faces detected!"
                            vision_data = bounding_boxes
                            vision_type = "boxes"

                        response = {
                            "status": "connected",
                            "risk_score": risk_score,
                            "message": msg,
                            "vision_data": vision_data,
                            "vision_type": vision_type,
                            "type": "vision_update"
                        }
                        await ws.send_text(json.dumps(response))
                    continue 
                else:
                    msg = "Unknown event logged."

                response = {"status": "connected", "risk_score": risk_score, "message": msg}
                await ws.send_text(json.dumps(response))
                
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)