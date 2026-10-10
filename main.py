import os, json, threading, queue, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import pyttsx3

APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="COD AI Analyst", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#080a0f 0%,#151015 65%,#250b10 100%); color:#f3f3f3}
h1,h2,h3 {color:#ff394d !important}
[data-testid="stMetric"] {background:#17191f;border:1px solid #54212a;padding:14px;border-radius:10px}
.stButton>button {background:#b7192b;color:white;border:0;border-radius:8px}
</style>
""", unsafe_allow_html=True)

st.title("🎯 COD AI ANALYST")
st.caption("Gameplay review • visible-target detection • movement tracking • performance coaching")
st.warning("Analysis only: process recorded gameplay or a permitted capture. This tool does not read game memory, reveal enemies through walls, inject into the game, or bypass anti-cheat.")

with st.sidebar:
    st.header("⚙️ Analysis settings")
    model_name = st.selectbox("Detection model", ["yolov8n.pt", "yolov8s.pt"], index=0,
                              help="COCO models detect general people, not COD-specific enemy identities.")
    conf = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    sample_every = st.slider("Process every Nth frame", 1, 8, 2)
    max_width = st.selectbox("Processing width", [640, 960, 1280], index=1)
    st.caption("Best results: upload a clear gameplay recording. Detection may confuse teammates, enemies, and HUD elements.")

tab_analyze, tab_stats, tab_aim, tab_voice = st.tabs(["🎥 Video analysis", "📊 Match report", "🎯 Aim coach", "🎙️ Voice assistant"])

if "report" not in st.session_state:
    st.session_state.report = None
if "annotated_video" not in st.session_state:
    st.session_state.annotated_video = None

with tab_analyze:
    uploaded = st.file_uploader("Upload a gameplay video", type=["mp4", "mov", "mkv", "avi"])
    if uploaded:
        st.video(uploaded)
        if st.button("Run AI analysis", type="primary"):
            input_path = OUTPUT_DIR / "input_video.mp4"
            input_path.write_bytes(uploaded.getvalue())
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                st.error("Could not open this video. Try MP4/H.264.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                scale = min(1.0, max_width / max(1, width))
                out_w, out_h = int(width * scale), int(height * scale)
                output_path = OUTPUT_DIR / "annotated_video.mp4"
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps/sample_every), (out_w, out_h))
                progress = st.progress(0)
                status = st.empty()
                frame_idx = 0
                processed = 0
                rows = []
                track_positions = {}
                heat = np.zeros((out_h, out_w), dtype=np.float32)
                try:
                    model = YOLO(model_name)
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if frame_idx % sample_every != 0:
                            frame_idx += 1
                            continue
                        frame = cv2.resize(frame, (out_w, out_h))
                        result = model.track(frame, persist=True, conf=conf, classes=[0], verbose=False)[0]
                        count = 0
                        if result.boxes is not None:
                            boxes = result.boxes
                            xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
                            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                            scores = boxes.conf.cpu().numpy() if boxes.conf is not None else [0] * len(xyxy)
                            for box, tid, score in zip(xyxy, ids, scores):
                                x1,y1,x2,y2 = map(int, box)
                                cx, cy = (x1+x2)//2, (y1+y2)//2
                                count += 1
                                if 0 <= cy < out_h and 0 <= cx < out_w:
                                    heat[cy, cx] += 1
                                label = f"Visible person {tid if tid is not None else ''} {float(score):.2f}"
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (40,55,255), 2)
                                cv2.putText(frame, label, (x1, max(18,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .48, (40,55,255), 2)
                                if tid is not None:
                                    track_positions.setdefault(int(tid), []).append((frame_idx/fps, cx, cy))
                                    pts = track_positions[int(tid)][-18:]
                                    for a,b in zip(pts, pts[1:]):
                                        cv2.line(frame, (a[1],a[2]), (b[1],b[2]), (0,220,255), 2)
                                rows.append({"time_seconds": round(frame_idx/fps,2), "track_id": tid, "confidence": round(float(score),3),
                                             "center_x":cx, "center_y":cy})
                        cv2.putText(frame, "COD AI ANALYST | RECORDED FOOTAGE", (14,28), cv2.FONT_HERSHEY_SIMPLEX, .65, (40,55,255), 2)
                        writer.write(frame)
                        processed += 1
                        frame_idx += 1
                        if total:
                            progress.progress(min(1.0, frame_idx/total))
                        status.text(f"Processed {frame_idx:,} / {total:,} frames • visible-person detections this frame: {count}")
                    cap.release()
                    writer.release()
                    csv_path = OUTPUT_DIR / "detections.csv"
                    df = pd.DataFrame(rows)
                    df.to_csv(csv_path, index=False)
                    heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
                    heat_path = OUTPUT_DIR / "tactical_heatmap.png"
                    cv2.imwrite(str(heat_path), heat_color)
                    duration = total/fps if fps else 0
                    report = {
                        "video_name": uploaded.name, "duration_seconds": round(duration,2),
                        "frames_processed": processed, "detections": len(df),
                        "unique_tracks": int(df["track_id"].dropna().nunique()) if not df.empty else 0,
                        "average_confidence": round(float(df["confidence"].mean()),3) if not df.empty else 0,
                        "sample_every": sample_every, "model": model_name
                    }
                    (OUTPUT_DIR/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    st.session_state.report = report
                    st.session_state.annotated_video = str(output_path)
                    st.success("Analysis complete.")
                    st.rerun()
                except Exception as e:
                    cap.release()
                    writer.release()
                    st.error(f"Analysis failed: {e}")
                    st.info("If this is the first run, keep the PC online so YOLO can download its model weights.")

    if st.session_state.annotated_video and Path(st.session_state.annotated_video).exists():
        st.subheader("Annotated gameplay")
        st.video(st.session_state.annotated_video)
        with open(st.session_state.annotated_video, "rb") as f:
            st.download_button("Download annotated video", f, file_name="cod_annotated.mp4", mime="video/mp4")
        heat_path = OUTPUT_DIR/"tactical_heatmap.png"
        if heat_path.exists():
            st.subheader("Detection heatmap")
            st.image(str(heat_path), caption="Hotter areas indicate more detected visible-person centers.")
            st.download_button("Download heatmap", heat_path.read_bytes(), "tactical_heatmap.png", "image/png")
        csv_path = OUTPUT_DIR/"detections.csv"
        if csv_path.exists():
            st.download_button("Download detections CSV", csv_path.read_bytes(), "detections.csv", "text/csv")

with tab_stats:
    st.subheader("Match statistics")
    report = st.session_state.report
    if report:
        cols = st.columns(4)
        cols[0].metric("Video length", f'{report["duration_seconds"]:.1f}s')
        cols[1].metric("Frames analyzed", f'{report["frames_processed"]:,}')
        cols[2].metric("Person detections", f'{report["detections"]:,}')
        cols[3].metric("Tracks observed", f'{report["unique_tracks"]:,}')
        st.write(f'**Model:** {report["model"]} · **Confidence average:** {report["average_confidence"]} · **Sampling:** every {report["sample_every"]} frame(s)')
        st.info("These are computer-vision counts, not official COD K/D, kills, or enemy counts. The general model cannot reliably distinguish enemies from teammates.")
        if (OUTPUT_DIR/"detections.csv").exists():
            df = pd.read_csv(OUTPUT_DIR/"detections.csv")
            if not df.empty:
                st.line_chart(df.groupby("time_seconds").size().rename("detections per sampled frame"))
                st.dataframe(df.head(300), use_container_width=True)
        st.download_button("Download JSON report", json.dumps(report, indent=2), "match_report.json", "application/json")
    else:
        st.info("Analyze a video first to create a report.")


with tab_aim:
    st.subheader("Aim settings & recorded-clip coach")
    st.caption("Recommendations and analysis only. This tab does not control the game or automate aiming.")
    control = st.radio("Choose your current input", ["PS5 controller", "Mouse & keyboard"], horizontal=True)
    if control == "PS5 controller":
        st.markdown("### Controller — starting point")
        st.write("- Aim Assist: **On** (in MWIII controller settings)")
        st.write("- Aim Response Curve: start with **Dynamic**; compare with Standard if it feels too twitchy")
        st.write("- Target Aim Assist: test the available in-game setting in a private match")
        st.write("- Sensitivity: begin around **5–7 horizontal / vertical**, then adjust one step at a time")
        st.write("- Deadzone: use the lowest value that avoids stick drift")
        st.info("These are starting points, not universal best settings. Menu names and options can vary by game update.")
    else:
        st.markdown("### Mouse & keyboard — starting point")
        st.write("- Disable mouse acceleration in Windows and keep pointer precision settings consistent")
        st.write("- Choose a sensitivity that lets you turn comfortably without losing fine control")
        st.write("- Keep ADS multipliers consistent while testing")
        st.write("- Change one setting at a time and compare the same training drill")
        st.info("Controller Aim Assist generally applies to controller input; do not expect it to behave the same while using a mouse.")

    st.markdown("### Estimate camera motion from a clip")
    st.write("Upload a short gameplay clip to estimate frame-to-frame camera movement. This is only a motion proxy: it cannot reliably isolate weapon recoil from turning, explosions, hit reactions, or camera shake.")
    aim_video = st.file_uploader("Upload a short clip for motion analysis", type=["mp4", "mov", "mkv", "avi"], key="aim_video")
    if aim_video and st.button("Analyze camera motion"):
        tmp = OUTPUT_DIR / "aim_motion_input.mp4"
        tmp.write_bytes(aim_video.getvalue())
        cap2 = cv2.VideoCapture(str(tmp))
        prev_gray = None
        motion_rows = []
        frame_no = 0
        fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30
        total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        bar = st.progress(0)
        status2 = st.empty()
        while True:
            ok2, frame2 = cap2.read()
            if not ok2:
                break
            if frame_no % max(1, sample_every) == 0:
                frame2 = cv2.resize(frame2, (min(640, frame2.shape[1]), int(frame2.shape[0] * min(1.0, 640/frame2.shape[1]))))
                gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_rows.append({
                        "time_seconds": round(frame_no/fps2, 2),
                        "mean_motion_px": float(np.mean(mag)),
                        "horizontal_motion_px": float(np.mean(np.abs(flow[..., 0]))),
                        "vertical_motion_px": float(np.mean(np.abs(flow[..., 1])))
                    })
                prev_gray = gray
            frame_no += 1
            if total2:
                bar.progress(min(1.0, frame_no/total2))
            status2.text(f"Analyzing frame {frame_no:,} / {total2:,}")
        cap2.release()
        if motion_rows:
            motion_df = pd.DataFrame(motion_rows)
            st.metric("Average frame motion", f'{motion_df["mean_motion_px"].mean():.2f} px')
            st.line_chart(motion_df.set_index("time_seconds")[["horizontal_motion_px", "vertical_motion_px"]])
            st.dataframe(motion_df.head(300), use_container_width=True)
            st.download_button("Download motion analysis CSV", motion_df.to_csv(index=False), "aim_motion_analysis.csv", "text/csv")
            st.caption("Higher values mean more visual change between sampled frames; they are not a direct measurement of recoil or aim accuracy.")
        else:
            st.warning("Not enough frames to calculate motion. Try a longer clip.")

with tab_voice:
    st.subheader("Voice assistant")
    st.write("Ask for a quick explanation or coaching tip. Speech playback works locally; microphone speech-to-text is optional and not enabled in this starter.")
    question = st.text_input("Ask the coach", placeholder="How can I improve my aim?")
    if st.button("Get coaching answer"):
        q = question.lower()
        if any(k in q for k in ["aim", "accuracy", "shoot"]):
            answer = "Review missed shots in short clips. Keep your crosshair near expected enemy height, use controlled bursts at range, and compare sensitivity changes one at a time."
        elif any(k in q for k in ["movement", "move", "position", "route"]):
            answer = "Review your movement path and avoid crossing open lanes without cover. Compare routes where you survived with routes where you were eliminated."
        elif any(k in q for k in ["heat", "map", "area"]):
            answer = "Use the heatmap as a visual summary of where visible detections occurred. It is not an enemy radar and does not show people behind walls."
        elif any(k in q for k in ["stat", "report", "number", "track"]):
            r = st.session_state.report
            answer = f'The latest analysis recorded {r["detections"]} person detections and {r["unique_tracks"]} track IDs.' if r else "Analyze a gameplay video first, then I can summarize its computer-vision statistics."
        else:
            answer = "Start by reviewing one short clip at a time. Note crosshair placement, cover usage, route choice, and whether you re-peek the same angle."
        st.success(answer)
        try:
            engine = pyttsx3.init()
            engine.say(answer)
            engine.runAndWait()
        except Exception:
            st.caption("Text-to-speech is unavailable on this Windows setup; the written answer is still available.")

st.divider()
st.caption("Starter build v0.1 • Runs locally on Windows • Video analysis only • Not affiliated with Activision.")
