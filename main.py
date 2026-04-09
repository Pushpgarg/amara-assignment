import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import base64
import cv2
import numpy as np

# Import our new module!
from proctor_engine import ProctorEngine

app = FastAPI(title="AI Proctor - Phase 0")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    # Create an independent engine instance for this specific candidate
    engine = ProctorEngine()
    is_in_background = False 
    
    TAB_SWITCH_PENALTY = 5    
    WINDOW_BLUR_PENALTY = 5.0     
    
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                event_type = payload.get("event")

                if event_type == "tab_switch":
                    is_in_background = True
                    # FIX: Updated to engine.risk.risk_score
                    engine.risk.risk_score = min(100.0, engine.risk.risk_score + TAB_SWITCH_PENALTY)
                    msg = "WARNING: Tab switch detected!"
                elif event_type == "window_blur":
                    is_in_background = True
                    # FIX: Updated to engine.risk.risk_score
                    engine.risk.risk_score = min(100.0, engine.risk.risk_score + WINDOW_BLUR_PENALTY)
                    msg = "WARNING: Window lost focus!"
                elif event_type in ["tab_focus", "window_focus"]:
                    is_in_background = False
                    msg = "System: Candidate returned to interview."
                    # FIX: Updated to engine.risk.risk_score
                    response = {"status": "connected", "risk_score": engine.risk.risk_score, "message": msg}
                    await ws.send_text(json.dumps(response))
                    continue
                elif event_type == "connected":
                    msg = "Monitoring started."
                
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
                        
                        # Pass the image to our engine and get the results back!
                        score, msg, vision_data, vision_type = engine.process_frame(img_rgb, time_scale, is_in_background)

                        response = {
                            "status": "connected",
                            "risk_score": score,
                            "message": msg,
                            "vision_data": vision_data,
                            "vision_type": vision_type,
                            "type": "vision_update"
                        }
                        await ws.send_text(json.dumps(response))
                    continue 
                else:
                    msg = "Unknown event logged."

                # FIX: Updated to engine.risk.risk_score
                response = {"status": "connected", "risk_score": engine.risk.risk_score, "message": msg}
                await ws.send_text(json.dumps(response))
                
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)