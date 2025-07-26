import cv2, os, json
from datetime import datetime

def save_alert(frame, event_type, camera_id="CAM_01"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{camera_id}_{timestamp}_{event_type}.jpg"
    path = os.path.join("alerts", filename)
    os.makedirs("alerts", exist_ok=True)
    cv2.imwrite(path, frame)

    meta = {
        "timestamp": timestamp,
        "camera_id": camera_id,
        "event_type": event_type,
        "frame_path": path
    }

    with open("alerts/logs.json", "a") as f:
        f.write(json.dumps(meta) + "\n")
    print(f"[ALERT] {event_type} at {timestamp}")