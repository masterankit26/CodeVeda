import cv2
import os
from ultralytics import YOLO
from config.zones import RESTRICTED_ZONES, CAMERA_ID
from utils.logger import save_alert
from utils.helpers import draw_zones, inside_zone

VIDEO_PATH = "assets/campus_footage.mp4"
MODEL_PATH = "yolov8n.pt"

def run_detection(video_path):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    try:
        print("[INFO] Loading YOLOv8 model...")
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video stream")
        return

    print("[INFO] Starting surveillance... Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        draw_zones(frame, RESTRICTED_ZONES)
        results = model(frame)[0]

        person_detected = []
        helmet_detected = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            label = model.names[int(class_id)]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if label == "person":
                person_detected.append((cx, cy))
            elif label == "helmet":
                helmet_detected.append((cx, cy))

        if person_detected and not helmet_detected:
            save_alert(frame, "No Helmet", CAMERA_ID)

        for (px, py) in person_detected:
            for zone_name, zone_coords in RESTRICTED_ZONES.items():
                if inside_zone(px, py, zone_coords):
                    save_alert(frame, f"Entered {zone_name}", CAMERA_ID)

        cv2.imshow("Smart Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Surveillance ended. All alerts saved.")

if __name__ == "__main__":
    run_detection(VIDEO_PATH)