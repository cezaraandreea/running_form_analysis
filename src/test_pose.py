import cv2
from pathlib import Path
import argparse

try:
    from src.pose_estimation import PoseEstimator
except ModuleNotFoundError:
    from pose_estimation import PoseEstimator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "run_test2.mp4"

parser = argparse.ArgumentParser(description="Test pentru Pose Estimation cu MediaPipe.")
parser.add_argument(
    "--video",
    type=str,
    default=str(DEFAULT_VIDEO_PATH),
    help="Calea către fișierul video de test (mp4/avi/etc).",
)
args = parser.parse_args()
VIDEO_PATH = Path(args.video).expanduser().resolve()

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Fișierul video nu există: {VIDEO_PATH}\n"
        "Rulează cu: python src/test_pose.py --video \"D:/cale/catre/video.mp4\""
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise ValueError(f"Nu pot deschide video: {VIDEO_PATH}")

with PoseEstimator() as pe:
    frame_idx = 0
    detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        pose_frame = pe.process_frame(frame, frame_idx, timestamp_ms)

        if pose_frame is not None:
            detected += 1
            frame = pe.draw_skeleton(frame, pose_frame)

        cv2.imshow("Pose Estimation Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print(f"Frame-uri procesate: {frame_idx}")
print(f"Frame-uri cu detectie: {detected}")
print(f"Detection rate: {100*detected/max(frame_idx,1):.2f}%")