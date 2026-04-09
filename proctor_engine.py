import mediapipe as mp
from vision_analyzer import VisionAnalyzer
from risk_assessor import RiskAssessor

class ProctorEngine:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)
        
        # Initialize our two sub-modules
        self.vision = VisionAnalyzer()
        self.risk = RiskAssessor()

    def process_frame(self, img_rgb, time_scale, is_in_background):
        results_detection = self.face_detection.process(img_rgb)
        face_count = len(results_detection.detections) if results_detection.detections else 0
        
        vision_data = []
        vision_type = "none"

        if face_count == 0:
            self.risk.risk_score = min(100.0, self.risk.risk_score + (self.risk.PENALTY_NO_FACE * time_scale))
            msg = "WARNING: Candidate not found!"
            
        elif face_count == 1:
            mesh_results = self.face_mesh.process(img_rgb)
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                
                # 1. Analyze the raw math
                states = self.vision.analyze(landmarks)
                
                # 2. Calculate the score based on the math
                score, msg = self.risk.calculate(states, is_in_background, time_scale)
                
                for landmark in landmarks:
                    vision_data.append({"x": landmark.x, "y": landmark.y})
            vision_type = "mesh"
            
        elif face_count > 1:
            self.risk.risk_score = min(100.0, self.risk.risk_score + (self.risk.PENALTY_CROWD * time_scale))
            msg = f"WARNING: {face_count} faces detected!"
            
            for detection in results_detection.detections:
                bbox = detection.location_data.relative_bounding_box
                vision_data.append({"xmin": bbox.xmin, "ymin": bbox.ymin, "width": bbox.width, "height": bbox.height})
            vision_type = "boxes"

        return self.risk.risk_score, msg, vision_data, vision_type