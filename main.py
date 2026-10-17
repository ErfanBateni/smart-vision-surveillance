import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Smart Vision Surveillance & Geofencing System"
  )
  parser.add_argument(
      "--source",
      type=str,
      default="sample_video.mp4",
      help="Path to input video or webcam index (e.g. 0)",
  )
  parser.add_argument(
      "--output",
      type=str,
      default="output.mp4",
      help="Path to export output video",
  )
  parser.add_argument(
      "--conf",
      type=float,
      default=0.4,
      help="Confidence threshold for YOLO model",
  )
  return parser.parse_args()


def run_surveillance(source_path, output_path, conf_thresh):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = YOLO("yolov8n.pt").to(device)

  cap = cv2.VideoCapture(int(source_path) if source_path.isdigit() else source_path)
  if not cap.isOpened():
    raise FileNotFoundError(f"Source video not found: {source_path}")

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)

  out = cv2.VideoWriter(
      output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
  )

  zone_pts = np.array(
      [
          [int(width * 0.55), int(height * 0.40)],
          [int(width * 0.95), int(height * 0.40)],
          [int(width * 0.95), int(height * 0.90)],
          [int(width * 0.45), int(height * 0.90)],
      ],
      np.int32,
  ).reshape((-1, 1, 2))

  line_in_y = int(height * 0.55)
  line_out_y = int(height * 0.65)

  track_history = defaultdict(lambda: deque(maxlen=30))
  passed_first_line = {}
  counted_ids = set()
  breached_ids = set()
  in_count = 0
  out_count = 0

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    results = model.track(
        frame,
        classes=[0],
        conf=conf_thresh,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        device=device,
    )

    is_breached = False
    current_intruders = []

    if results[0].boxes.id is not None:
      boxes = results[0].boxes.xyxy.cpu().numpy()
      track_ids = results[0].boxes.id.int().cpu().tolist()

      for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        cx = int((x1 + x2) / 2)
        cy = int(y2)

        track_history[track_id].append((cx, cy))
        inside_zone = cv2.pointPolygonTest(zone_pts, (cx, cy), False) >= 0

        box_color = (0, 0, 255) if inside_zone else (70, 220, 70)
        status_text = (
            f"BREACH! ID:{track_id}" if inside_zone else f"ID:{track_id}"
        )

        if inside_zone:
          is_breached = True
          breached_ids.add(track_id)
          current_intruders.append(track_id)

        pts = np.array(track_history[track_id], dtype=np.int32).reshape(
            (-1, 1, 2)
        )
        cv2.polylines(
            frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(
            frame,
            (cx, cy),
            4,
            (0, 0, 255) if inside_zone else (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame,
            status_text,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
        )

        if track_id not in counted_ids:
          if cy > line_in_y and track_id not in passed_first_line:
            passed_first_line[track_id] = "crossed_A"
          if (
              passed_first_line.get(track_id) == "crossed_A"
              and cy > line_out_y
          ):
            in_count += 1
            counted_ids.add(track_id)
            passed_first_line.pop(track_id, None)

          elif cy < line_out_y and track_id not in passed_first_line:
            passed_first_line[track_id] = "crossed_B"
          if (
              passed_first_line.get(track_id) == "crossed_B"
              and cy < line_in_y
          ):
            out_count += 1
            counted_ids.add(track_id)
            passed_first_line.pop(track_id, None)

    zone_overlay = frame.copy()
    cv2.fillPoly(
        zone_overlay,
        [zone_pts],
        (0, 0, 200) if is_breached else (0, 165, 255),
    )
    cv2.polylines(
        frame,
        [zone_pts],
        isClosed=True,
        color=(0, 0, 255) if is_breached else (0, 200, 255),
        thickness=2,
    )
    cv2.addWeighted(zone_overlay, 0.35, frame, 0.65, 0, frame)

    cv2.line(
        frame, (0, line_in_y), (int(width * 0.45), line_in_y), (0, 200, 255), 2
    )
    cv2.line(
        frame,
        (0, line_out_y),
        (int(width * 0.45), line_out_y),
        (255, 120, 0),
        2,
    )

    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (10, 15), (280, 155), (20, 20, 20), -1)
    cv2.addWeighted(hud_overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(
        frame,
        f"INFLOW    : {in_count}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"OUTFLOW   : {out_count}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 160, 255),
        2,
    )
    cv2.putText(
        frame,
        f"ZONE VIOLATIONS: {len(breached_ids)}",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        f"ACTIVE INTRUDERS: {len(current_intruders)}",
        (25, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (50, 50, 255) if is_breached else (200, 200, 200),
        2,
    )

    out.write(frame)

  cap.release()
  out.release()
  print(f"Execution complete. Output exported to {output_path}")


if __name__ == "__main__":
  args = parse_arguments()
  run_surveillance(args.source, args.output, args.conf)