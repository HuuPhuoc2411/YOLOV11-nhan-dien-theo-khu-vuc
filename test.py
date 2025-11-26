import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import json
import time
import torch

# sử dụng chuột trái để chấm điểm
# vẽ xong một khu vực thì bấm chuột phải để hoàn thành
# dùng phím enter để lưu lại tọa độ và kết thúc vẽ khu vực
# vẽ sai cần vẽ lại thì bấm phím C để clear
# code sẽ tự chạy
# ==================== CONFIGURATION ====================
OUTPUT_DIR = "output_interactive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- CHỈNH 2 DÒNG NÀY THEO MÁY BẠN ----
VIDEO_INPUT = r"D:\yolov11\ee.mp4"
YOLO_MODEL  = r"D:\yolov11\yolov10n.pt"
# --------------------------------------

CONFIDENCE = 0.3

# Tăng tốc: bỏ qua bớt frame (1 = không bỏ, 2 = nhanh hơn ~2x,...)
FRAME_SKIP = 2

# Khoảng cách tối đa (pixel) để coi 2 detection là cùng 1 object giữa 2 frame
MAX_TRACK_DIST = 60
# Số frame tối đa không thấy thì xóa track
MAX_TRACK_AGE = 20

# Object colors (BGR format)
COLORS = {
    'car': (0, 255, 255),        # Cyan
    'truck': (255, 0, 255),      # Magenta
    'motorcycle': (0, 255, 0),   # Green
    'person': (0, 165, 255),     # Orange
    'bus': (255, 200, 0),        # Sky Blue
    'bicycle': (0, 255, 255)     # Yellow
}

# Zone colors for drawing
ZONE_COLORS = [
    (0, 100, 255),   # Orange-Red
    (255, 0, 200),   # Magenta
    (0, 255, 0),     # Green
    (255, 200, 0),   # Cyan
    (255, 0, 100)    # Purple
]

# ==================== INTERACTIVE ZONE DRAWER ====================
class ZoneDrawer:
    def __init__(self, frame, video_path):
        self.original = frame.copy()
        self.zones = []
        self.current_zone = []
        self.video_path = video_path

        # Fit display to screen (max 1600x900)
        self.orig_height, self.orig_width = frame.shape[:2]
        max_width, max_height = 1600, 900

        scale_w = max_width / self.orig_width
        scale_h = max_height / self.orig_height
        self.scale = min(scale_w, scale_h, 1.0)

        self.display_width = int(self.orig_width * self.scale)
        self.display_height = int(self.orig_height * self.scale)

        self.frame = cv2.resize(self.original, (self.display_width, self.display_height))

        print(f"\nOriginal video: {self.orig_width}x{self.orig_height}")
        print(f"Display size: {self.display_width}x{self.display_height}")
        print(f"Scale factor: {self.scale:.2f}")

        self.window_name = "ZONE DRAWER - Draw zones and press ENTER"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coord -> original coord
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.current_zone.append((orig_x, orig_y))
            self.redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_zone) >= 3:
                zone_num = len(self.zones) + 1
                zone_name = f"ZONE {zone_num}"
                self.zones.append({
                    'name': zone_name,
                    'points': self.current_zone.copy(),
                    'color': ZONE_COLORS[len(self.zones) % len(ZONE_COLORS)]
                })
                print(f"  ✓ Zone {zone_num} created with {len(self.current_zone)} points")
                self.current_zone = []
                self.redraw()

    def redraw(self):
        self.frame = cv2.resize(self.original, (self.display_width, self.display_height))

        # Draw grid
        grid_color = (40, 40, 40)
        grid_spacing = 50
        for x in range(0, self.display_width, grid_spacing):
            cv2.line(self.frame, (x, 0), (x, self.display_height), grid_color, 1)
        for y in range(0, self.display_height, grid_spacing):
            cv2.line(self.frame, (0, y), (self.display_width, y), grid_color, 1)

        # Draw finished zones
        for zone in self.zones:
            disp_pts = np.array([(int(p[0] * self.scale), int(p[1] * self.scale))
                                 for p in zone['points']])
            color = zone['color']

            overlay = self.frame.copy()
            cv2.fillPoly(overlay, [disp_pts], color)
            cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)

            cv2.polylines(self.frame, [disp_pts], True, (255, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.polylines(self.frame, [disp_pts], True, color, 2, lineType=cv2.LINE_AA)

            for pt in disp_pts:
                cv2.circle(self.frame, tuple(pt), 4, (255, 255, 255), -1)
                cv2.circle(self.frame, tuple(pt), 3, color, -1)

            cx = int(np.mean([p[0] for p in disp_pts]))
            cy = int(np.mean([p[1] for p in disp_pts]))
            text = zone['name']
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            cv2.rectangle(self.frame,
                          (cx - text_size[0] // 2 - 8, cy - text_size[1] // 2 - 8),
                          (cx + text_size[0] // 2 + 8, cy + text_size[1] // 2 + 8),
                          (0, 0, 0), -1)
            cv2.rectangle(self.frame,
                          (cx - text_size[0] // 2 - 8, cy - text_size[1] // 2 - 8),
                          (cx + text_size[0] // 2 + 8, cy + text_size[1] // 2 + 8),
                          color, 2)
            cv2.putText(self.frame, text,
                        (cx - text_size[0] // 2, cy + text_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        # Draw current zone being created
        if len(self.current_zone) > 0:
            for i, pt in enumerate(self.current_zone):
                disp_pt = (int(pt[0] * self.scale), int(pt[1] * self.scale))
                cv2.circle(self.frame, disp_pt, 6, (0, 255, 255), -1)
                cv2.putText(self.frame, str(i + 1), (disp_pt[0] + 8, disp_pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)

                if i > 0:
                    prev_pt = self.current_zone[i - 1]
                    prev_disp = (int(prev_pt[0] * self.scale), int(prev_pt[1] * self.scale))
                    cv2.line(self.frame, prev_disp, disp_pt, (0, 255, 255), 2)

        # Status text
        status = f"Zones: {len(self.zones)} | Points current: {len(self.current_zone)}"
        cv2.putText(self.frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        cv2.imshow(self.window_name, self.frame)

    def run(self):
        print("\n" + "=" * 70)
        print("INTERACTIVE ZONE DRAWING")
        print("=" * 70)
        print("LEFT CLICK  : add point")
        print("RIGHT CLICK : finish current zone (>=3 points)")
        print("ENTER       : start processing")
        print("C           : clear all zones")
        print("ESC         : exit")
        print("=" * 70)

        self.redraw()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                if len(self.zones) > 0:
                    cv2.destroyWindow(self.window_name)
                    return self.zones
                else:
                    print("Please create at least one zone.")
            elif key == ord('c'):
                self.zones = []
                self.current_zone = []
                print("All zones cleared.")
                self.redraw()
            elif key == 27:  # ESC
                cv2.destroyWindow(self.window_name)
                return None

# ==================== HELPER FUNCTIONS ====================
def point_in_polygon(point, polygon):
    """Ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def draw_gradient_line(img, pt1, pt2, color1, color2, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    distance = math.hypot(x2 - x1, y2 - y1)
    steps = int(distance)
    if steps <= 0:
        return
    for i in range(steps):
        a = i / steps
        x = int(x1 + (x2 - x1) * a)
        y = int(y1 + (y2 - y1) * a)
        b = int(color1[0] + (color2[0] - color1[0]) * a)
        g = int(color1[1] + (color2[1] - color1[1]) * a)
        r = int(color1[2] + (color2[2] - color1[2]) * a)
        cv2.circle(img, (x, y), thickness, (b, g, r), -1)

def draw_zone_overlay(img, zone, total_count=None):
    """Vẽ zone + label (có thể kèm tổng số object đã đếm)."""
    points = np.array(zone['points'])
    color = zone['color']

    overlay = img.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    cv2.polylines(img, [points], True, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.polylines(img, [points], True, color, 2, lineType=cv2.LINE_AA)

    cx = int(np.mean([p[0] for p in points]))
    cy = int(np.mean([p[1] for p in points]))

    if total_count is None:
        text = zone['name']
    else:
        text = f"{zone['name']} ({total_count})"

    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

    cv2.rectangle(img,
                  (cx - ts[0] // 2 - 8, cy - ts[1] // 2 - 8),
                  (cx + ts[0] // 2 + 8, cy + ts[1] // 2 + 8),
                  (0, 0, 0), -1)
    cv2.rectangle(img,
                  (cx - ts[0] // 2 - 8, cy - ts[1] // 2 - 8),
                  (cx + ts[0] // 2 + 8, cy + ts[1] // 2 + 8),
                  color, 2)
    cv2.putText(img, text,
                (cx - ts[0] // 2, cy + ts[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)

def box_polygon_overlap_ratio(box, polygon, samples=7):
    """
    Ước lượng phần trăm diện tích bbox nằm trong polygon
    bằng cách lấy mẫu lưới điểm.
    """
    x1, y1, x2, y2 = map(int, box)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w == 0 or h == 0:
        return 0.0

    step_x = max(1, w // samples)
    step_y = max(1, h // samples)

    total = 0
    inside = 0

    for xx in range(x1, x2 + 1, step_x):
        for yy in range(y1, y2 + 1, step_y):
            total += 1
            if point_in_polygon((xx, yy), polygon):
                inside += 1

    if total == 0:
        return 0.0
    return inside / total

def best_zone_for_box(box, zones):
    """
    Tìm zone có tỉ lệ overlap lớn nhất với bbox.
    Trả về (zone, ratio).
    """
    best_zone = None
    best_ratio = 0.0
    for zone in zones:
        ratio = box_polygon_overlap_ratio(box, zone['points'])
        if ratio > best_ratio:
            best_ratio = ratio
            best_zone = zone
    return best_zone, best_ratio

def draw_zone_stats_panel(frame, zone_total_counts, start_x=10, start_y=50):
    """
    Vẽ bảng thống kê cộng dồn từng zone:
    ZONE 1: car=20, motorcycle=10, ...
    """
    lines = []
    for zname, cls_dict in zone_total_counts.items():
        parts = []
        for cls_name, cnt in cls_dict.items():
            if cnt > 0:
                parts.append(f"{cls_name}={cnt}")
        if parts:
            line = f"{zname}: " + ", ".join(parts)
        else:
            line = f"{zname}: 0"
        lines.append(line)

    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    width = 0
    line_height = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        width = max(width, w)
        line_height = max(line_height, h)

    pad = 8
    panel_width = width + 2 * pad
    panel_height = line_height * len(lines) + 2 * pad + (len(lines) - 1) * 4

    x1, y1 = start_x, start_y
    x2, y2 = x1 + panel_width, y1 + panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    y_text = y1 + pad + line_height
    for line in lines:
        cv2.putText(frame, line, (x1 + pad, y_text),
                    font, font_scale, (0, 255, 255), thickness + 1, lineType=cv2.LINE_AA)
        y_text += line_height + 4

# ==================== MAIN PROCESSING ====================
def process_video(zones, video_path):
    print("\n" + "=" * 70)
    print("STARTING VIDEO PROCESSING")
    print("=" * 70)

    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL)

    # Dùng GPU nếu có
    if torch.cuda.is_available():
        model.to("cuda")
        print("Model device: CUDA")
    else:
        print("Model device: CPU")

    names = model.model.names

    print(f"Opening video: {video_path}")
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps == 0 or fps is None:
        fps = 25

    output_path = os.path.join(OUTPUT_DIR, "output_with_zones_fast_count_total.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Zones: {len(zones)}")
    print("=" * 70)

    frame_count = 0
    current_fps = 0
    fps_counter = 0
    start_time = time.time()
    last_fps_time = start_time

    center_point = (width // 2, height // 2)

    # lưu detection của frame trước để tracking
    tracks = {}          # id -> {'center': (x,y), 'last_seen': frame}
    next_track_id = 1

    # CỘNG DỒN: zone -> class -> count
    zone_total_counts = {zone['name']: defaultdict(int) for zone in zones}
    # Mỗi zone lưu các ID đã được đếm (để không đếm lại)
    zone_counted_ids = {zone['name']: set() for zone in zones}

    # Cache results nếu FRAME_SKIP > 1
    last_boxes = None
    last_classes = None

    while True:
        success, frame = cam.read()
        if not success:
            break

        frame_count += 1
        fps_counter += 1

        now = time.time()
        if now - last_fps_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            last_fps_time = now

        # RUN YOLO (theo FRAME_SKIP)
        if (frame_count % FRAME_SKIP == 0) or (last_boxes is None):
            results = model.predict(frame, conf=CONFIDENCE, imgsz=640, verbose=False)
            last_boxes = results[0].boxes.xyxy.cpu().numpy()
            last_classes = results[0].boxes.cls.cpu().numpy()

        boxes = last_boxes
        classes = last_classes

        detections = []  # mỗi phần tử: {'box', 'cls', 'center', 'id'}

        # Lấy ra các detection (tọa độ + class)
        for box, cls in zip(boxes, classes):
            obj_class = names[int(cls)]
            if obj_class not in COLORS:
                continue
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                'box': (x1, y1, x2, y2),
                'cls': obj_class,
                'center': (cx, cy)
            })

        # --------- SIMPLE TRACKER: gán ID cho từng detection ----------
        used_track_ids = set()

        for det in detections:
            cx, cy = det['center']
            best_id = None
            best_dist = 1e9

            for tid, tinfo in tracks.items():
                if tid in used_track_ids:
                    continue
                px, py = tinfo['center']
                dist = math.hypot(cx - px, cy - py)
                if dist < MAX_TRACK_DIST and dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = next_track_id
                next_track_id += 1

            used_track_ids.add(best_id)
            tracks[best_id] = {'center': (cx, cy), 'last_seen': frame_count}
            det['id'] = best_id

        # Xóa các track đã quá lâu không thấy
        to_delete = []
        for tid, tinfo in tracks.items():
            if frame_count - tinfo['last_seen'] > MAX_TRACK_AGE:
                to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]
        # --------------------------------------------------------------

        frame_totals = defaultdict(int)

        # Xử lý từng detection (đã có ID)
        for det in detections:
            x1, y1, x2, y2 = det['box']
            obj_class = det['cls']
            obj_id = det['id']
            center = det['center']

            # Tìm zone có overlap lớn nhất
            best_zone, ratio = best_zone_for_box((x1, y1, x2, y2), zones)

            # CHỈ ĐẾM NẾU DIỆN TÍCH TRONG ZONE >= 50%
            if best_zone is None or ratio < 0.5:
                continue

            zone_name = best_zone['name']

            # Nếu ID này CHƯA được đếm trong zone này => cộng dồn
            if obj_id not in zone_counted_ids[zone_name]:
                zone_total_counts[zone_name][obj_class] += 1
                zone_counted_ids[zone_name].add(obj_id)

            # Dùng để hiển thị tổng object phát hiện trong frame (không quan trọng lắm)
            frame_totals[obj_class] += 1

            color = COLORS[obj_class]

            # Vẽ bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Label: CLASS#ID
            label = f"{obj_class.upper()}#{obj_id}"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - ts[1] - 6),
                          (x1 + ts[0] + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Vision line
            draw_gradient_line(frame, center_point, center, (255, 255, 0), color, 2)
            cv2.circle(frame, center, 4, color, -1, lineType=cv2.LINE_AA)

        # Vẽ zone + tổng đã đếm cho zone đó
        for zone in zones:
            zname = zone['name']
            total_for_zone = sum(zone_total_counts[zname].values())
            draw_zone_overlay(frame, zone, total_count=total_for_zone)

        # Bảng thống kê chi tiết từng zone
        draw_zone_stats_panel(frame, zone_total_counts, start_x=10, start_y=50)

        # Vẽ FPS + tổng
        cv2.putText(
            frame,
            f"Frame: {frame_count:05d} | FPS: {current_fps:02d} | Total(frame): {sum(frame_totals.values()):04d}",
            (15, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA
        )

        writer.write(frame)
        cv2.imshow("Processing Video - Press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

        if frame_count % 100 == 0:
            print(f"Frame {frame_count:05d} | FPS: {current_fps:02d}")

    cam.release()
    writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Time: {total_time:.2f}s")
    print(f"Avg FPS (processing): {frame_count / total_time:.2f}")
    print("\n=== FINAL ZONE COUNTS ===")
    for zname, cls_dict in zone_total_counts.items():
        print(f"{zname}:")
        for cls_name, cnt in cls_dict.items():
            print(f"  {cls_name}: {cnt}")
    print("=" * 70)

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INTERACTIVE TRAFFIC MONITORING SYSTEM (FAST + UNIQUE ZONE COUNTS)")
    print("=" * 70)

    print("CHECKING FILES...")
    print("MODEL EXISTS? :", os.path.exists(YOLO_MODEL))
    print("VIDEO EXISTS? :", os.path.exists(VIDEO_INPUT))

    if not os.path.exists(YOLO_MODEL):
        print("❌ ERROR: YOLO MODEL NOT FOUND")
        exit(1)
    if not os.path.exists(VIDEO_INPUT):
        print("❌ ERROR: VIDEO FILE NOT FOUND")
        exit(1)

    cam = cv2.VideoCapture(VIDEO_INPUT)
    ok, first_frame = cam.read()
    cam.release()
    if not ok:
        print("❌ ERROR: Cannot read first frame of the video")
        exit(1)

    drawer = ZoneDrawer(first_frame, VIDEO_INPUT)
    zones = drawer.run()
    cv2.destroyAllWindows()
    time.sleep(0.2)

    if not zones:
        print("No zones created. Exiting...")
        exit(0)

    zones_file = os.path.join(OUTPUT_DIR, "zones_config_fast_count_total.json")
    with open(zones_file, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"Zones saved to: {zones_file}")

    process_video(zones, VIDEO_INPUT)

