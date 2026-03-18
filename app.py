import json
import os
import signal
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BoKing", page_icon="🥊", layout="wide")


def request_app_shutdown() -> None:
    """
    Gracefully stop the local Streamlit process when the user clicks Close.
    """
    pid = os.getpid()

    def _shutdown() -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            os._exit(0)

    # Small delay allows the UI message to render before termination.
    threading.Timer(0.5, _shutdown).start()


@dataclass
class FighterState:
    name: str
    color: Tuple[int, int, int]
    center_history: List[Tuple[float, float]]
    punches_landed: int = 0
    punch_attempts: int = 0
    last_punch_frame: int = -9999


def create_tracker() -> cv2.Tracker:
    """Create the best available OpenCV tracker across different builds."""
    constructors = [
        "TrackerCSRT_create",
        "TrackerKCF_create",
        "TrackerMIL_create",
        "legacy_TrackerCSRT_create",
        "legacy_TrackerKCF_create",
        "legacy_TrackerMIL_create",
    ]

    for constructor in constructors:
        if constructor.startswith("legacy_"):
            legacy_name = constructor.replace("legacy_", "")
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, legacy_name):
                return getattr(cv2.legacy, legacy_name)()
        elif hasattr(cv2, constructor):
            return getattr(cv2, constructor)()

    raise RuntimeError(
        "No supported OpenCV tracker found. Install opencv-contrib-python for best results."
    )


def detect_fighters_hog(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect likely people and return two best bounding boxes."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    detections = []
    for (x, y, w, h), score in zip(boxes, weights):
        detections.append((x, y, w, h, float(score)))

    detections.sort(key=lambda item: item[4], reverse=True)
    best_boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h, _ in detections[:2]]

    if len(best_boxes) < 2:
        h, w = frame.shape[:2]
        # Fallback: assume two fighters start from opposite sides.
        best_boxes = [
            (int(0.05 * w), int(0.2 * h), int(0.35 * w), int(0.7 * h)),
            (int(0.60 * w), int(0.2 * h), int(0.35 * w), int(0.7 * h)),
        ]

    best_boxes = sorted(best_boxes, key=lambda b: b[0])
    return best_boxes[:2]


def sanitize_bbox(
    box: Tuple[float, float, float, float], frame_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Convert any bbox to a plain Python int tuple and clamp it to frame bounds.
    This prevents OpenCV tracker init/update argument-type errors on some builds.
    """
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = [int(round(float(v))) for v in box]

    x = max(0, min(x, frame_w - 2))
    y = max(0, min(y, frame_h - 2))
    w = max(2, min(w, frame_w - x))
    h = max(2, min(h, frame_h - y))
    return (x, y, w, h)


def center_of(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def maybe_count_punch(
    attacker: FighterState,
    defender: FighterState,
    frame_id: int,
    fps: float,
    min_speed_px_per_sec: float,
    hit_distance_px: float,
    min_frames_between_punches: int,
) -> None:
    if len(attacker.center_history) < 3 or len(defender.center_history) < 2:
        return

    prev = attacker.center_history[-2]
    cur = attacker.center_history[-1]
    d_prev = distance(prev, defender.center_history[-2])
    d_cur = distance(cur, defender.center_history[-1])
    displacement = distance(prev, cur)
    speed = displacement * fps

    if speed < min_speed_px_per_sec:
        return

    attacker.punch_attempts += 1

    if (
        d_cur < hit_distance_px
        and d_cur < d_prev
        and frame_id - attacker.last_punch_frame >= min_frames_between_punches
    ):
        attacker.punches_landed += 1
        attacker.last_punch_frame = frame_id


def process_video(video_path: Path) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Uploaded video appears empty.")

    boxes = detect_fighters_hog(first_frame)
    tracker_a = create_tracker()
    tracker_b = create_tracker()

    init_box_a = sanitize_bbox(boxes[0], first_frame.shape)
    init_box_b = sanitize_bbox(boxes[1], first_frame.shape)

    tracker_a.init(first_frame, init_box_a)
    tracker_b.init(first_frame, init_box_b)

    fighter_a = FighterState("Fighter A", (0, 255, 0), [center_of(tuple(map(float, init_box_a)))])
    fighter_b = FighterState("Fighter B", (0, 128, 255), [center_of(tuple(map(float, init_box_b)))])

    stats_rows = []
    progress = st.progress(0.0)
    frame_slot = st.empty()

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ok_a, box_a = tracker_a.update(frame)
        ok_b, box_b = tracker_b.update(frame)

        if not (ok_a and ok_b):
            frame_id += 1
            continue

        box_a = tuple(map(float, box_a))
        box_b = tuple(map(float, box_b))

        ca = center_of(box_a)
        cb = center_of(box_b)
        fighter_a.center_history.append(ca)
        fighter_b.center_history.append(cb)

        frame_h, frame_w = frame.shape[:2]
        hit_distance_px = max(frame_w, frame_h) * 0.15

        maybe_count_punch(
            fighter_a,
            fighter_b,
            frame_id,
            fps,
            min_speed_px_per_sec=220.0,
            hit_distance_px=hit_distance_px,
            min_frames_between_punches=max(1, int(fps * 0.2)),
        )
        maybe_count_punch(
            fighter_b,
            fighter_a,
            frame_id,
            fps,
            min_speed_px_per_sec=220.0,
            hit_distance_px=hit_distance_px,
            min_frames_between_punches=max(1, int(fps * 0.2)),
        )

        for fighter, box in [(fighter_a, box_a), (fighter_b, box_b)]:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), fighter.color, 2)
            cv2.putText(
                frame,
                f"{fighter.name} | Landed: {fighter.punches_landed}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                fighter.color,
                2,
                cv2.LINE_AA,
            )

        if frame_id % int(max(1, fps // 2)) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb, caption=f"Processing frame {frame_id}", use_container_width=True)

        stats_rows.append(
            {
                "time_sec": frame_id / fps,
                "fighter_a_landed": fighter_a.punches_landed,
                "fighter_b_landed": fighter_b.punches_landed,
                "fighters_distance_px": distance(ca, cb),
            }
        )

        frame_id += 1
        if total_frames > 0:
            progress.progress(min(frame_id / total_frames, 1.0))

    cap.release()
    progress.progress(1.0)

    stats_df = pd.DataFrame(stats_rows)
    summary = {
        "fighter_a": {
            "punches_landed": fighter_a.punches_landed,
            "punch_attempts_detected": fighter_a.punch_attempts,
        },
        "fighter_b": {
            "punches_landed": fighter_b.punches_landed,
            "punch_attempts_detected": fighter_b.punch_attempts,
        },
        "processed_frames": frame_id,
        "fps": fps,
        "note": "Punch detection is heuristic (motion-based) and should be reviewed for official use.",
    }

    return {"summary": summary, "timeline": stats_df}


st.title("🥊 BoKing")
st.caption(
    "Upload boxing sparring footage, track both fighters, and get punch-landed stats back."
)

if st.sidebar.button("Close BoKing App"):
    st.sidebar.success("Closing BoKing...")
    request_app_shutdown()

with st.expander("How it works", expanded=True):
    st.markdown(
        """
        1. Upload your sparring footage.
        2. BoKing detects two fighters and tracks them frame-by-frame.
        3. It estimates punch attempts/landed punches from high-speed movement toward the opponent.
        4. You get a summary table + timeline data to download.

        > This is an MVP analytics model and not a certified judging system.
        """
    )

uploaded = st.file_uploader(
    "Upload sparring video",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=False,
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    st.video(str(tmp_path))

    if st.button("Analyze with BoKing", type="primary"):
        try:
            results = process_video(tmp_path)
            summary = results["summary"]
            timeline: pd.DataFrame = results["timeline"]

            st.subheader("Fight Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fighter A - Landed", summary["fighter_a"]["punches_landed"])
                st.metric("Fighter A - Attempts", summary["fighter_a"]["punch_attempts_detected"])
            with col2:
                st.metric("Fighter B - Landed", summary["fighter_b"]["punches_landed"])
                st.metric("Fighter B - Attempts", summary["fighter_b"]["punch_attempts_detected"])

            st.subheader("Timeline")
            if not timeline.empty:
                chart_df = timeline[["time_sec", "fighter_a_landed", "fighter_b_landed"]].set_index("time_sec")
                st.line_chart(chart_df)
                st.dataframe(timeline, use_container_width=True)
            else:
                st.warning("No trackable frames found. Try a higher quality video with both fighters visible.")

            st.download_button(
                "Download JSON Summary",
                data=json.dumps(summary, indent=2),
                file_name="boking_summary.json",
                mime="application/json",
            )
            st.download_button(
                "Download Timeline CSV",
                data=timeline.to_csv(index=False),
                file_name="boking_timeline.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
