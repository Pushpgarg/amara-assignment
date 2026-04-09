import numpy as np

class VisionAnalyzer:
    def __init__(self):
        # Memory variables
        self.previous_mouth_ratio = 0.0
        self.gaze_history = []
        self.previous_nose = None
        
        # Math Thresholds
        self.TALKING_VARIANCE_THRESHOLD = 0.015  
        self.YAWN_THRESHOLD = 0.15               
        self.HEAD_YAW_THRESHOLD = 4    
        self.READING_VARIANCE_THRESHOLD = 0.0015
        self.HEAD_MOTION_THRESHOLD = 0.04 

    def analyze(self, landmarks):
        # 1. Head Velocity Filter
        nose_x, nose_y = landmarks[1].x, landmarks[1].y
        is_head_moving_fast = False
        
        if self.previous_nose:
            nose_dist = np.sqrt((nose_x - self.previous_nose['x'])**2 + (nose_y - self.previous_nose['y'])**2)
            if nose_dist > self.HEAD_MOTION_THRESHOLD:
                is_head_moving_fast = True
        self.previous_nose = {'x': nose_x, 'y': nose_y}

        # 2. Lip Movement
        mouth_dist = landmarks[14].y - landmarks[13].y
        face_height = landmarks[152].y - landmarks[10].y
        current_mouth_ratio = mouth_dist / face_height if face_height > 0 else 0
        mouth_movement_delta = abs(current_mouth_ratio - self.previous_mouth_ratio)
        
        is_yawn_motion = (current_mouth_ratio > self.YAWN_THRESHOLD) or (self.previous_mouth_ratio > self.YAWN_THRESHOLD)
        is_talking = (mouth_movement_delta > self.TALKING_VARIANCE_THRESHOLD) and not is_yawn_motion and not is_head_moving_fast
        self.previous_mouth_ratio = current_mouth_ratio

        # 3. Head Pose
        left_edge_x, right_edge_x = landmarks[234].x, landmarks[454].x
        dist_left, dist_right = abs(nose_x - left_edge_x), abs(right_edge_x - nose_x)
        yaw_ratio = dist_left / max(dist_right, 0.001)
        is_looking_away = yaw_ratio > self.HEAD_YAW_THRESHOLD or yaw_ratio < (1 / self.HEAD_YAW_THRESHOLD)

        # 4. Iris Tracking
        eye_openness = abs(landmarks[159].y - landmarks[145].y)
        is_blinking = eye_openness < 0.015
        is_reading = False
        variance = 0.0
        
        if not is_blinking:
            eye_width = abs(landmarks[133].x - landmarks[33].x)
            pupil_pos = abs(landmarks[468].x - landmarks[33].x)
            self.gaze_history.append(pupil_pos / max(eye_width, 0.001))
            
            if len(self.gaze_history) > 30:
                self.gaze_history.pop(0)
                variance = np.var(self.gaze_history)
                if variance > self.READING_VARIANCE_THRESHOLD and not is_looking_away and not is_head_moving_fast:
                    is_reading = True
                    self.gaze_history.clear() 

        return {
            "is_talking": is_talking, "is_looking_away": is_looking_away, 
            "is_reading": is_reading, "mouth_delta": mouth_movement_delta, 
            "yaw_ratio": yaw_ratio, "variance": variance
        }