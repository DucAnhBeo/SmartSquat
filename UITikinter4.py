import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model # type: ignore
import mediapipe as mp

# Load model & label
model = load_model("Demo code project\squat_pose_model_4Labels.h5")
label_map = ['Excessive Lean', 'Rounded Back', 'Not Deep Enough', 'perfect']

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Tính góc giữa 3 điểm
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Trích đặc trưng 3 góc: gối, thân, lưng
def extract_features(results):
    lm = results.pose_landmarks.landmark
    origin_x = lm[mp_pose.PoseLandmark.RIGHT_HIP].x
    origin_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y
    
    def rel(p):
        return [lm[p].x - origin_x, lm[p].y - origin_y]

    shoulder = rel(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    hip = rel(mp_pose.PoseLandmark.RIGHT_HIP)
    knee = rel(mp_pose.PoseLandmark.RIGHT_KNEE)
    ankle = rel(mp_pose.PoseLandmark.RIGHT_ANKLE)
    nose = rel(mp_pose.PoseLandmark.NOSE)

    knee_angle = calculate_angle(hip, knee, ankle) / 180
    torso_angle = calculate_angle(shoulder, hip, knee) / 180
    back_angle = calculate_angle(hip, shoulder, nose) / 180

    return np.array([knee_angle, torso_angle, back_angle])

# Chuẩn hoá sequence về 100 frame
def normalize_sequence(seq, target_length=100):
    seq = np.array(seq)
    n = len(seq)

    if n == 0:
        return np.zeros((target_length, 3))
    if n < target_length:
        pad = np.zeros((target_length - n, 3))
        return np.concatenate([seq, pad], axis=0)
    if n > target_length:
        indices = np.linspace(0, n - 1, num=target_length).astype(int)
        return seq[indices]
    return seq

# Xử lý video realtime: đếm rep + dự đoán từng rep
def process_video_realtime(video_path):
    cap = cv2.VideoCapture(video_path)
    rep_count = 0
    in_rep = False
    last_label = "N/A"
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
            knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            knee_angle = calculate_angle(hip, knee, ankle)

            features = extract_features(results)

            if knee_angle < 150:
                in_rep = True
                sequence.append(features)
            elif knee_angle > 160 and in_rep:
                in_rep = False
                rep_count += 1

                input_seq = normalize_sequence(sequence, 100)
                input_seq = np.expand_dims(input_seq, axis=0)
                pred = model.predict(input_seq, verbose=0)
                last_label = label_map[np.argmax(pred)]
                sequence = []  # reset cho rep mới

        # Hiển thị thông tin
        cv2.putText(frame, f"Reps: {rep_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Label: {last_label}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Squat Rep Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Giao diện chọn video
def choose_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if video_path:
        process_video_realtime(video_path)

# Giao diện chính
root = tk.Tk()
root.title("Realtime Squat Rep Analyzer")
root.geometry("320x200")

tk.Label(root, text="Phân tích tư thế Squat từng rep", font=("Arial", 14)).pack(pady=15)
tk.Button(root, text="Chọn video", command=choose_video, font=("Arial", 12)).pack(pady=10)

root.mainloop()
