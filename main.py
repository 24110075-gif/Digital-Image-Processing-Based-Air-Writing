# main.py
import cv2
import math
import time
import config
from processing.pipeline_processor import render_trajectory_to_64x64
from tracking.webcam import Webcam
from tracking.hand_tracker import HandTracker
from trajectory.trajectory_manager import TrajectoryManager
from reconstruction.handwriting import HandwritingCanvas
from reconstruction.ui_dashboard import DashboardUI
from recognition.classifier import CharacterClassifier
import numpy as np


# -----------------------------------------------------------------
#  Writing state machine:
#
#   IDLE --> WRITING --> PENDING --> RECOGNIZING --> IDLE
#                ^           | (resumed writing)
#                +-----------+
#
#  - IDLE       : Khong viet, cho gesture WRITING.
#  - WRITING    : Dang thu trajectory tu ngon tro.
#  - PENDING    : Ngung WRITING, cho PENDING_DURATION giay (timestamp).
#                 Neu nguoi dung bat dau viet lai -> quay ve WRITING.
#  - RECOGNIZING: Da cho du, goi CNN roi ve IDLE.
# -----------------------------------------------------------------
STATE_IDLE        = "IDLE"
STATE_WRITING     = "WRITING"
STATE_PENDING     = "PENDING"
STATE_RECOGNIZING = "RECOGNIZING"


def print_controls():
    """In huong dan thao tac gesture khi khoi dong chuong trinh."""
    print("=" * 52)
    print("  AIR-WRITING CONTROLS")
    print("=" * 52)
    print()
    print("  WRITING  : Index UP, other fingers CLOSED")
    print("             -> Write with index fingertip")
    print()
    print("  SPACE    : Index + Pinky UP, Middle + Ring CLOSED")
    print("             -> Hold ~0.5s to add SPACE")
    print()
    print("  DELETE   : All fingers CLOSED (fist)")
    print("             -> Delete last character")
    print()
    print("  MODE     : Index + Middle UP (V-sign)")
    print("             -> Switch LETTER / NUMBER mode")
    print()
    print("  IMPORTANT:")
    print("  * PINCH is NO LONGER used for writing.")
    print("  * Keep the index finger FREE while writing.")
    print("  * Gestures require several stable frames.")
    print("  * SPACE requires holding ~0.5s to avoid accidents.")
    print("  * CNN fires ~{:.1f}s after you stop writing.".format(config.PENDING_DURATION))
    print()
    print("  KEYBOARD SHORTCUTS:")
    print("  [d] Toggle CNN 64x64 debug window")
    print("  [c] Clear canvas")
    print("  [s] Save canvas to file")
    print("  [q] Quit")
    print("=" * 52)
    print()


def _count_trajectory_points(trajectory_mgr) -> int:
    """Count total trajectory points across all strokes (including current)."""
    total = sum(len(s) for s in trajectory_mgr.raw_strokes)
    total += len(trajectory_mgr.current_stroke)
    return total


def main():
    with Webcam(camera_index=0, width=config.CANVAS_WIDTH, height=config.CANVAS_HEIGHT) as webcam:
        tracker = HandTracker(
            min_detection_con=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_con=config.MIN_TRACKING_CONFIDENCE,
            pinch_threshold=config.PINCH_THRESHOLD,
            stabilization_frames=config.GESTURE_STABILIZATION_FRAMES,
        )

        trajectory_mgr = TrajectoryManager(
            smoothing_window=config.SMOOTHING_WINDOW_SIZE,
            max_jump_distance=config.MAX_JUMP_DISTANCE,
            min_stroke_points=config.MIN_STROKE_POINTS,
        )

        canvas_mgr = HandwritingCanvas(
            width=config.CANVAS_WIDTH,
            height=config.CANVAS_HEIGHT,
            bg_color=config.CANVAS_BG_COLOR,
            draw_color=config.DRAWING_COLOR,
            thickness=config.DRAWING_THICKNESS,
        )

        dashboard_ui = DashboardUI(width=1280, height=720)

        # CNN Classifier (load once)
        classifier = None
        try:
            classifier = CharacterClassifier(config.MODEL_PATH, class_labels=config.CLASS_LABELS)
        except FileNotFoundError as e:
            print(f"[Warning] {e}")
            print("[Notice] System running in tracking-only mode. "
                  "Load model to 'models/airwriting_cnn.keras' to enable CNN.")
        except Exception as e:
            print(f"[Error] Cannot init CNN Classifier: {e}")

        print_controls()
        print("[System] Air-Writing System Initialized!")
        print()

        # Runtime state
        writing_state  = STATE_IDLE     # IDLE | WRITING | PENDING | RECOGNIZING
        pending_start  = None           # time.time() when PENDING begins
        prev_fingertip = None           # fingertip position from previous frame

        prev_gesture   = None           # last printed gesture (spam prevention)
        prev_log_state = None           # last printed writing state
        current_mode   = "LETTER"       # LETTER | NUMBER

        # Per-gesture cooldowns (frame counters)
        mode_cooldown   = 0
        delete_cooldown = 0
        space_cooldown  = 0
        # SPACE hold counter: must hold SPACE gesture for SPACE_HOLD_FRAMES before triggering
        space_hold_ctr  = 0
        # DELETE hold counter: must hold DELETE gesture (fist) for DELETE_HOLD_FRAMES before triggering
        delete_hold_ctr = 0
        # PENDING resume counter: consecutive WRITING frames needed to resume from PENDING
        pending_resume_ctr = 0

        show_cnn_debug = config.SHOW_RECOGNITION_DEBUG

        last_recognized_char = None
        last_confidence = 0.0
        recognized_text = ""

        # Main loop
        while True:
            success, raw_frame = webcam.read_frame()
            if not success:
                print("Cannot read frame from camera!")
                break

            # 1. Flip frame for mirror view
            frame = cv2.flip(raw_frame, 1)

            # 2. MediaPipe tracking + gesture recognition
            frame, stable_gesture, mapped_fingertip = tracker.process_frame(frame)

            # Log gesture changes (no spam)
            if stable_gesture != prev_gesture:
                if stable_gesture not in ("IDLE", "NO_HAND"):
                    print(f"[GESTURE] {stable_gesture}")
                prev_gesture = stable_gesture

            # Decrement cooldowns every frame
            if mode_cooldown   > 0: mode_cooldown   -= 1
            if delete_cooldown > 0: delete_cooldown -= 1
            if space_cooldown  > 0: space_cooldown  -= 1
            # Reset SPACE hold counter if gesture is not SPACE this frame
            if stable_gesture != "SPACE":
                space_hold_ctr = 0
            # Reset DELETE hold counter if gesture is not DELETE this frame
            if stable_gesture != "DELETE":
                delete_hold_ctr = 0
            # Reset PENDING resume counter if gesture is not WRITING
            if stable_gesture != "WRITING":
                pending_resume_ctr = 0

            # ============================================================
            #  GESTURE CONTROL LOGIC
            # ============================================================

            # MODE: toggle LETTER / NUMBER
            if stable_gesture == "MODE" and mode_cooldown == 0:
                current_mode = "NUMBER" if current_mode == "LETTER" else "LETTER"
                mode_cooldown = 30          # ~1s at 30fps
                print(f"[MODE] {current_mode}")

            # SPACE: must hold gesture for SPACE_HOLD_FRAMES consecutive frames (~0.5s)
            elif stable_gesture == "SPACE" and space_cooldown == 0:
                space_hold_ctr += 1
                if space_hold_ctr >= config.SPACE_HOLD_FRAMES:
                    if writing_state == STATE_WRITING:
                        trajectory_mgr.finish_stroke()
                    writing_state = STATE_IDLE
                    pending_start = None
                    recognized_text += " "
                    space_cooldown = 30
                    space_hold_ctr = 0
                    print("[GESTURE] SPACE")

            # DELETE: must hold fist gesture for DELETE_HOLD_FRAMES consecutive frames (~0.5s)
            elif stable_gesture == "DELETE" and delete_cooldown == 0:
                delete_hold_ctr += 1
                if delete_hold_ctr >= config.DELETE_HOLD_FRAMES:
                    if writing_state == STATE_WRITING:
                        trajectory_mgr.finish_stroke()
                    writing_state = STATE_IDLE
                    pending_start = None
                    if recognized_text:
                        recognized_text = recognized_text[:-1]
                    trajectory_mgr.clear()
                    last_recognized_char = None
                    last_confidence = 0.0
                    delete_cooldown = 20
                    delete_hold_ctr = 0
                    print("[GESTURE] DELETE")

            # WRITING: collect index fingertip trajectory
            elif stable_gesture == "WRITING":

                if writing_state == STATE_IDLE:
                    # First frame of WRITING gesture
                    writing_state = STATE_WRITING
                    pending_start = None

                elif writing_state == STATE_PENDING:
                    # Require several consecutive WRITING frames to resume from PENDING
                    pending_resume_ctr += 1
                    if pending_resume_ctr >= config.GESTURE_STABILIZATION_FRAMES:
                        writing_state = STATE_WRITING
                        pending_start = None
                        pending_resume_ctr = 0

                # Always collect trajectory while in WRITING state
                if writing_state == STATE_WRITING:
                    if mapped_fingertip:
                        pt_i = (int(round(mapped_fingertip[0])), int(round(mapped_fingertip[1])))
                        cv2.circle(frame, pt_i, 6, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
                    trajectory_mgr.add_point(mapped_fingertip)

            # Other gestures (IDLE, NO_HAND): start PENDING if we just left WRITING
            else:
                if writing_state == STATE_WRITING:
                    # Just left WRITING -> finish stroke, start PENDING timer
                    trajectory_mgr.finish_stroke()
                    writing_state = STATE_PENDING
                    pending_start = time.time()
                    print("[WRITING] Stopped -> PENDING ({:.1f}s)".format(config.PENDING_DURATION))

                if mapped_fingertip:
                    pt_i = (int(round(mapped_fingertip[0])), int(round(mapped_fingertip[1])))
                    cv2.circle(frame, pt_i, 6, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)


            # Check if PENDING timer has expired (non-blocking, checked every frame)
            if writing_state == STATE_PENDING and pending_start is not None:
                elapsed = time.time() - pending_start
                if elapsed >= config.PENDING_DURATION:
                    writing_state = STATE_RECOGNIZING

            # RECOGNIZING: call CNN once, then reset to IDLE
            if writing_state == STATE_RECOGNIZING:
                total_pts = _count_trajectory_points(trajectory_mgr)
                print(f"[DEBUG] RECOGNIZING: total_pts={total_pts}, classifier={'OK' if classifier else 'None'}")

                if classifier is not None and total_pts >= config.MIN_TRAJECTORY_POINTS:
                    smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
                    canvas_mgr.update_canvas(smoothed_strokes)

                    # Render trajectory sang anh 64x64 bang render_trajectory_to_64x64 (DIP Pipeline)
                    img_64 = render_trajectory_to_64x64(
                        smoothed_strokes,
                        target_size=config.CNN_INPUT_SIZE,
                        padding=6,
                        line_thickness=3
                    )
                    non_zero = cv2.countNonZero(img_64) if img_64 is not None else 0

                    char, conf = classifier.predict(
                        img_64,
                        mode=current_mode,
                        confidence_threshold=config.CONFIDENCE_THRESHOLD
                    )
                    if char is not None:
                        last_recognized_char = char
                        last_confidence = conf
                        recognized_text += char
                        print(f"[WRITING] END -> CNN RECOGNITION: '{char}' ({conf:.2f}%)")
                    else:
                        print(f"[WRITING] END -> CNN: unrecognizable or confidence below {config.CONFIDENCE_THRESHOLD}%")
                elif classifier is None:
                    print(f"[WRITING] END -> CNN Classifier not loaded! Cannot recognize.")
                elif total_pts < config.MIN_TRAJECTORY_POINTS:
                    print(f"[WRITING] END -> Skipped CNN ({total_pts} pts < min {config.MIN_TRAJECTORY_POINTS})")

                # Clear trajectory, back to IDLE
                trajectory_mgr.clear()
                writing_state = STATE_IDLE
                pending_start = None

            prev_fingertip = mapped_fingertip

            # Log writing-state changes
            if writing_state != prev_log_state:
                if writing_state == STATE_WRITING:
                    print("[STATE] -> WRITING")
                prev_log_state = writing_state

            # ============================================================
            #  TRAJECTORY -> CANVAS & FRAME RENDER
            # ============================================================
            smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
            canvas_mgr.update_canvas(smoothed_strokes)
            current_canvas = canvas_mgr.get_canvas()

            # Preview real-time trajectory on camera frame (smooth anti-aliased neon cyan)
            for stroke in smoothed_strokes:
                for i in range(1, len(stroke)):
                    pt1 = (int(round(float(stroke[i - 1][0]))), int(round(float(stroke[i - 1][1]))))
                    pt2 = (int(round(float(stroke[i][0]))), int(round(float(stroke[i][1]))))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 3, cv2.LINE_AA)


            # ============================================================
            #  DUAL-PANEL DASHBOARD UI RENDERING
            # ============================================================
            master_dashboard = dashboard_ui.create_dashboard(
                camera_frame=frame,
                canvas_img=current_canvas,
                stable_gesture=stable_gesture,
                writing_state=writing_state,
                current_mode=current_mode,
                recognized_text=recognized_text,
                last_recognized_char=last_recognized_char,
                last_confidence=last_confidence,
                pending_start=pending_start,
                pending_duration=config.PENDING_DURATION,
                show_cnn_debug=show_cnn_debug
            )

            # Display Master Dual-Panel Dashboard Window
            cv2.imshow("Air-Writing System (Dual-Panel Dashboard)", master_dashboard)

            if show_cnn_debug and classifier is not None:
                smoothed_strokes = trajectory_mgr.get_smoothed_strokes()
                debug_64x64 = render_trajectory_to_64x64(
                    smoothed_strokes,
                    target_size=config.CNN_INPUT_SIZE,
                    padding=6,
                    line_thickness=3
                )
                cv2.imshow("CNN Input Debug (64x64)",
                           cv2.resize(debug_64x64, (200, 200), interpolation=cv2.INTER_NEAREST))

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_cnn_debug = not show_cnn_debug
                if not show_cnn_debug:
                    cv2.destroyWindow("CNN Input Debug (64x64)")
                print(f"[Debug] CNN debug window: {'ON' if show_cnn_debug else 'OFF'}")
            elif key == ord('c'):
                trajectory_mgr.clear()
                last_recognized_char = None
                last_confidence = 0.0
                recognized_text = ""
                writing_state = STATE_IDLE
                pending_start = None
                space_hold_ctr = 0
                print("[System] Canvas cleared.")
            elif key == ord('s'):
                canvas_mgr.save_image()


if __name__ == "__main__":
    main()
