import cv2
import numpy as np
import mediapipe as mp

class ProctorEngine:
    def __init__(self):
        # --- Initialize MediaPipe Models ---
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            min_detection_confidence=0.5, 
            refine_landmarks=True
        )
        
        # --- State Memory ---
        self.risk_score = 0.0
        self.previous_mouth_ratio = 0.0
        self.gaze_history = []
        self.previous_nose = None
        self.reading_violation_count = 0 # <--- NEW: Track total offenses
        
        # --- Tuning Variables ---
        self.PENALTY_NO_FACE = 5.0      
        self.PENALTY_CROWD = 10.0       
        self.DECAY_GOOD_BEHAVIOR = 0.5  
        self.PENALTY_BACKGROUND = 10.0
        
        self.PENALTY_TALKING = 5.0    
        self.TALKING_VARIANCE_THRESHOLD = 0.015  
        self.YAWN_THRESHOLD = 0.15               
        
        self.PENALTY_LOOKING_AWAY = 5.0 
        self.HEAD_YAW_THRESHOLD = 4    
        
        self.PENALTY_READING = 15.0
        self.READING_VARIANCE_THRESHOLD = 0.0015
        
        self.HEAD_MOTION_THRESHOLD = 0.04 # How fast the head can move before we pause detection

    def process_frame(self, img_rgb, time_scale, is_in_background):
        results_detection = self.face_detection.process(img_rgb)
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
            self.risk_score = min(100.0, self.risk_score + (self.PENALTY_NO_FACE * time_scale))
            msg = "WARNING: Candidate not found!"
            
        elif face_count == 1:
            mesh_results = self.face_mesh.process(img_rgb)
            
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                
                # --- NEW: Head Velocity Filter ---
                nose_x = landmarks[1].x
                nose_y = landmarks[1].y
                
                is_head_moving_fast = False
                if self.previous_nose:
                    # Calculate how far the nose moved since the last frame
                    nose_dist = np.sqrt((nose_x - self.previous_nose['x'])**2 + (nose_y - self.previous_nose['y'])**2)
                    if nose_dist > self.HEAD_MOTION_THRESHOLD:
                        is_head_moving_fast = True
                        
                # Save current nose position for the next frame
                self.previous_nose = {'x': nose_x, 'y': nose_y}

                # --- 1. Lip Movement ---
                mouth_dist = landmarks[14].y - landmarks[13].y
                face_height = landmarks[152].y - landmarks[10].y
                current_mouth_ratio = mouth_dist / face_height if face_height > 0 else 0
                mouth_movement_delta = abs(current_mouth_ratio - self.previous_mouth_ratio)
                
                is_currently_yawning = current_mouth_ratio > self.YAWN_THRESHOLD
                was_previously_yawning = self.previous_mouth_ratio > self.YAWN_THRESHOLD
                is_yawn_motion = is_currently_yawning or was_previously_yawning
                
                # Cannot be talking if the head is actively whipping around
                is_talking = (mouth_movement_delta > self.TALKING_VARIANCE_THRESHOLD) and not is_yawn_motion and not is_head_moving_fast
                self.previous_mouth_ratio = current_mouth_ratio

                # --- 2. Head Pose ---
                left_edge_x = landmarks[234].x
                right_edge_x = landmarks[454].x
                dist_left = abs(nose_x - left_edge_x)
                dist_right = abs(right_edge_x - nose_x)
                yaw_ratio = dist_left / max(dist_right, 0.001)
                is_looking_away = yaw_ratio > self.HEAD_YAW_THRESHOLD or yaw_ratio < (1 / self.HEAD_YAW_THRESHOLD)
                
                # --- 3. Iris Tracking ---
                # --- 3. Iris Tracking ---
                eye_openness = abs(landmarks[159].y - landmarks[145].y)
                is_blinking = eye_openness < 0.015
                is_reading = False
                
                # FIX: ALWAYS record the eye position if they are not blinking
                if not is_blinking:
                    eye_width = abs(landmarks[133].x - landmarks[33].x)
                    pupil_pos = abs(landmarks[468].x - landmarks[33].x)
                    gaze_ratio = pupil_pos / max(eye_width, 0.001)
                    self.gaze_history.append(gaze_ratio)
                    
                    if len(self.gaze_history) > 30:
                        self.gaze_history.pop(0)
                        variance = np.var(self.gaze_history)
                        
                        # Only trigger the reading penalty if the head is relatively still
                        if variance > self.READING_VARIANCE_THRESHOLD and not is_looking_away and not is_head_moving_fast:
                            is_reading = True
                            # Flush the buffer so the penalty doesn't skyrocket!
                            self.gaze_history.clear() 
                
                # --- 4. State Hierarchy ---
                if is_in_background:
                    self.risk_score = min(100.0, self.risk_score + (self.PENALTY_BACKGROUND * time_scale))
                    msg = "WARNING: Candidate is on another tab!"
                elif is_looking_away:
                    self.risk_score = min(100.0, self.risk_score + (self.PENALTY_LOOKING_AWAY * time_scale))
                    msg = f"WARNING: Looking away! (Yaw Ratio: {yaw_ratio:.2f})"
                elif is_reading:
                    # Increment the offense counter
                    self.reading_violation_count += 1
                    
                    # Scale the penalty based on how many times they've been caught
                    scaled_penalty = self.PENALTY_READING * self.reading_violation_count
                    
                    self.risk_score = min(100.0, self.risk_score + (scaled_penalty * time_scale))
                    msg = f"WARNING: Screen reading! (Offense #{self.reading_violation_count})"
                elif is_talking:
                    self.risk_score = min(100.0, self.risk_score + (self.PENALTY_TALKING * time_scale))
                    msg = f"WARNING: Speaking detected! (Movement: {mouth_movement_delta:.3f})"
                elif self.risk_score > 0:
                    self.risk_score = max(0.0, self.risk_score - (self.DECAY_GOOD_BEHAVIOR * time_scale))
    
                for landmark in landmarks:
                    vision_data.append({"x": landmark.x, "y": landmark.y})
            
            vision_type = "mesh"
            
        elif face_count > 1:
            self.risk_score = min(100.0, self.risk_score + (self.PENALTY_CROWD * time_scale))
            msg = f"WARNING: {face_count} faces detected!"
            vision_data = bounding_boxes
            vision_type = "boxes"

        return self.risk_score, msg, vision_data, vision_type