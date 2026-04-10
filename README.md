## 🧠 Dynamic Risk Scoring Engine (State Machine)

Unlike legacy proctoring tools that use binary "cheating/not cheating" flags, proposed solution utilizes a **Continuous Risk Assessor**. The score floats between `0.0` and `100.0`. 

To ensure fairness, the engine applies a **Good Behavior Decay**. If no violations are detected in the current frame, the system dynamically reduces the candidate's risk score by `-0.5 points per second`.

When violations occur, penalties are injected using a strict **State Hierarchy** to prevent double-penalizing the candidate for overlapping behaviors. The hierarchy evaluates in the following order: `Background Tab` > `Looking Away` > `Reading` > `Talking` > `Decay`.

### 1. OS & Browser Focus (Tab Switching)
* **The Logic:** The client-side DOM continuously polls `document.hidden` and `!document.hasFocus()` every 83ms, sending the "Absolute Truth" of the candidate's OS focus to Python, bypassing spoofable `blur` events.
* **The Penalty:** * **Instant Strike:** +5.0 points immediately upon losing window focus.
  * **Continuous Bleed:** +10.0 points per second for every second the candidate remains on another tab.

### 2. External Focus (Looking Away)
* **The Math:** Calculates 3D head yaw by measuring the Euclidean distance from the tip of the nose (Landmark 1) to the extreme left and right edges of the face silhouette (Landmarks 234 & 454). If the ratio is severely skewed (`> 4.0` or `< 0.25`), the candidate is flagged.
* **The Penalty:** Continuous bleed of **+5.0 points per second**.

### 3. Invisible AI Screen Assistance (Reading Detection)
* **The Math:** Tracks the exact horizontal placement of the pupil relative to the eye corners. This ratio is logged into a rolling 30-frame array (representing exactly 2.5 seconds of history at 12 FPS). If the statistical variance `np.var(gaze_history)` exceeds `0.0015`, it perfectly maps to the rhythmic, left-to-right **Saccadic Eye Movements** of reading a hidden screen.
* **Smart Filters:** The buffer ignores blinks (`eye_openness < 0.015`) and pauses tracking if the Head Velocity filter detects rapid head shaking.
* **The Penalty (Violation Multiplier):** Because reading is highly indicative of ChatGPT/AI usage, this triggers an escalating penalty. 
  * **Offense 1:** +15 points
  * **Offense 2:** +30 points
  * **Offense 3:** +45 points (Guarantees Risk Score maxes out at 100).

### 4. Speech & Whisper Detection (Visual VAD)
* **The Math:** Measures the vertical distance between the upper and lower lip mapped against overall face height. If the delta between consecutive frames exceeds `0.015`, active speech is flagged.
* **Smart Filters:** Includes a **Yawn Filter** (`ratio > 0.15`) and a **Head Velocity Filter** (`nose_dist > 0.04`) to prevent false positives when a candidate stretches or adjusts their posture.
* **The Penalty:** Continuous bleed of **+5.0 points per second**.

### 5. Crowd & Missing Face Detection
* **The Math:** Utilizes a lightweight Bounding Box model triggered exactly once per second (`frame_counter % 12 == 0`) to conserve server compute.
* **The Penalty:** * Missing Face: **+5.0 points per second**. 
  * Multiple Faces: **+10.0 points per second**.