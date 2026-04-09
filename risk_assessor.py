class RiskAssessor:
    def __init__(self):
        self.risk_score = 0.0
        self.reading_violation_count = 0
        
        # Penalty Tuning
        self.PENALTY_NO_FACE = 5.0      
        self.PENALTY_CROWD = 10.0       
        self.DECAY_GOOD_BEHAVIOR = 0.5  
        self.PENALTY_BACKGROUND = 10.0
        self.PENALTY_TALKING = 5.0    
        self.PENALTY_LOOKING_AWAY = 5.0 
        self.PENALTY_READING = 15.0

    def calculate(self, states, is_in_background, time_scale):
        msg = "Normal behavior."

        # State Hierarchy
        if is_in_background:
            self.risk_score = min(100.0, self.risk_score + (self.PENALTY_BACKGROUND * time_scale))
            msg = "WARNING: Candidate is on another tab!"
        elif states.get("is_looking_away"):
            self.risk_score = min(100.0, self.risk_score + (self.PENALTY_LOOKING_AWAY * time_scale))
            msg = f"WARNING: Looking away! (Yaw Ratio: {states['yaw_ratio']:.2f})"
        elif states.get("is_reading"):
            self.reading_violation_count += 1
            scaled_penalty = self.PENALTY_READING * self.reading_violation_count
            self.risk_score = min(100.0, self.risk_score + (scaled_penalty * time_scale))
            msg = f"WARNING: Screen reading! (Offense #{self.reading_violation_count})"
        elif states.get("is_talking"):
            self.risk_score = min(100.0, self.risk_score + (self.PENALTY_TALKING * time_scale))
            msg = f"WARNING: Speaking detected! (Movement: {states['mouth_delta']:.3f})"
        elif self.risk_score > 0:
            self.risk_score = max(0.0, self.risk_score - (self.DECAY_GOOD_BEHAVIOR * time_scale))

        return self.risk_score, msg