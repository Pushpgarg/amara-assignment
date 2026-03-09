import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import base64
import cv2
import numpy as np

app = FastAPI(title="AI Proctor - Phase 0")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    risk_score = 0  
    
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                event_type = payload.get("event")

                if event_type == "tab_switch":
                    risk_score = min(100, risk_score + 25)
                    msg = "WARNING: Tab switch detected!"
                elif event_type == "window_blur":
                    risk_score = min(100, risk_score + 10)
                    msg = "WARNING: Window lost focus!"
                elif event_type == "connected":
                    msg = "Monitoring started."
                elif event_type == "frame":
                    # --- NEW: Decode the incoming image ---
                    image_data = payload.get("image", "")
                    if "," in image_data:
                        # Remove the "data:image/jpeg;base64," prefix
                        base64_str = image_data.split(",")[1]
                        img_bytes = base64.b64decode(base64_str)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        # Print to your terminal to prove it works
                        print(f"Success! Received frame of shape: {img.shape}")
                        
                    msg = "Frame processed."
                    # We don't send a WebSocket message back for every single frame 
                    # to save bandwidth, so we just continue waiting for the next one.
                    continue 
                else:
                    msg = "Unknown event logged."

                response = {
                    "status": "connected",
                    "risk_score": risk_score,
                    "message": msg,
                }
                await ws.send_text(json.dumps(response))
                
            except json.JSONDecodeError:
                response = {
                    "status": "connected",
                    "risk_score": risk_score,
                    "message": f"Echo: {data}",
                }
                await ws.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)