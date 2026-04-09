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

# 2. FaceMesh Detector (Heavy, extracts 468 points + Iris)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5, 
    refine_landmarks=True  # Enables Iris Tracking
)

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    # --- State Memory Variables ---
    risk_score = 0.0  
    is_in_background = False 
    previous_mouth_ratio = 0.0 
    gaze_history = []  # Array to store the last 30 frames of eye positions
    
    # ⭐️ TUNING VARIABLES (Points per second) ⭐️
    PENALTY_NO_FACE = 5.0      
    PENALTY_CROWD = 10.0       
    DECAY_GOOD_BEHAVIOR = 0.5  
    TAB_SWITCH_PENALTY = 5    
    WINDOW_BLUR_PENALTY = 5.0     
    PENALTY_BACKGROUND = 10.0
    
    PENALTY_TALKING = 5.0    
    TALKING_VARIANCE_THRESHOLD = 0.015  
    YAWN_THRESHOLD = 0.15               
    
    PENALTY_LOOKING_AWAY = 5.0 
    HEAD_YAW_THRESHOLD = 4    
    
    # --- NEW: Reading Variables ---
    PENALTY_READING = 15.0
    READING_VARIANCE_THRESHOLD = 0.0015  # Variance needed to trigger reading penalty
    
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
                        
                        # 1. Detect number of faces
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
                        variance = 0.0

                        if face_count == 0:
                            risk_score = min(100.0, risk_score + (PENALTY_NO_FACE * time_scale))
                            msg = "WARNING: Candidate not found!"
                            
                        elif face_count == 1:
                            mesh_results = face_mesh.process(img_rgb)
                            
                            if mesh_results.multi_face_landmarks:
                                landmarks = mesh_results.multi_face_landmarks[0].landmark
                                
                                # --- Feature 1: Lip Movement Math ---
                                mouth_dist = landmarks[14].y - landmarks[13].y
                                face_height = landmarks[152].y - landmarks[10].y
                                current_mouth_ratio = mouth_dist / face_height if face_height > 0 else 0
                                mouth_movement_delta = abs(current_mouth_ratio - previous_mouth_ratio)
                                
                                is_currently_yawning = current_mouth_ratio > YAWN_THRESHOLD
                                was_previously_yawning = previous_mouth_ratio > YAWN_THRESHOLD
                                is_yawn_motion = is_currently_yawning or was_previously_yawning
                                is_talking = (mouth_movement_delta > TALKING_VARIANCE_THRESHOLD) and not is_yawn_motion
                                previous_mouth_ratio = current_mouth_ratio

                                # --- Feature 2: Head Pose Math (Yaw Estimation) ---
                                nose_x = landmarks[1].x
                                left_edge_x = landmarks[234].x
                                right_edge_x = landmarks[454].x
                                dist_left = abs(nose_x - left_edge_x)
                                dist_right = abs(right_edge_x - nose_x)
                                yaw_ratio = dist_left / max(dist_right, 0.001)
                                is_looking_away = yaw_ratio > HEAD_YAW_THRESHOLD or yaw_ratio < (1 / HEAD_YAW_THRESHOLD)
                                
                                # --- Feature 3: Iris Tracking (Reading Detection) ---
                                eye_openness = abs(landmarks[159].y - landmarks[145].y)
                                is_blinking = eye_openness < 0.015
                                is_reading = False
                                
                                if not is_blinking:
                                    eye_width = abs(landmarks[133].x - landmarks[33].x)
                                    pupil_pos = abs(landmarks[468].x - landmarks[33].x)
                                    gaze_ratio = pupil_pos / max(eye_width, 0.001)
                                    gaze_history.append(gaze_ratio)
                                    
                                    # Keep only the last 30 frames
                                    if len(gaze_history) > 30:
                                        gaze_history.pop(0)
                                        variance = np.var(gaze_history)
                                        if variance > READING_VARIANCE_THRESHOLD and not is_looking_away:
                                            is_reading = True
                                
                                # --- State Hierarchy (Apply Penalties) ---
                                if is_in_background:
                                    risk_score = min(100.0, risk_score + (PENALTY_BACKGROUND * time_scale))
                                    msg = "WARNING: Candidate is on another tab!"
                                elif is_looking_away:
                                    risk_score = min(100.0, risk_score + (PENALTY_LOOKING_AWAY * time_scale))
                                    msg = f"WARNING: Looking away! (Yaw Ratio: {yaw_ratio:.2f})"
                                elif is_reading:
                                    risk_score = min(100.0, risk_score + (PENALTY_READING * time_scale))
                                    msg = f"WARNING: Hidden screen reading detected! (Var: {variance:.5f})"
                                elif is_talking:
                                    risk_score = min(100.0, risk_score + (PENALTY_TALKING * time_scale))
                                    msg = f"WARNING: Speaking detected! (Movement: {mouth_movement_delta:.3f})"
                                elif risk_score > 0:
                                    risk_score = max(0.0, risk_score - (DECAY_GOOD_BEHAVIOR * time_scale))
                                    
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