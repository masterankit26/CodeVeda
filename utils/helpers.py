import cv2
import numpy as np

def draw_zones(frame, zones):
    for name, coords in zones.items():
        pts = np.array(coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 0, 255), thickness=2)
        cv2.putText(frame, name, coords[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def inside_zone(x, y, zone_coords):
    return cv2.pointPolygonTest(np.array(zone_coords, np.int32), (int(x), int(y)), False) >= 0