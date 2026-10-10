# COD AI Analyst (Windows)

A local starter project for reviewing **recorded Call of Duty: Modern Warfare III gameplay**.

## Features
- Detects visible people in recorded footage using a general-purpose YOLO model.
- Tracks detections across sampled frames and draws short movement trails.
- Creates a detection heatmap.
- Exports a CSV and JSON report.
- Provides basic coaching prompts and local text-to-speech.
- Tactical red/black Streamlit interface.

## Important limitations
- The included model is trained for general `person` detection, not Call of Duty enemy classification.
- It cannot reliably tell enemy players from teammates, mannequins, or on-screen artwork.
- Counts are computer-vision detections, **not** official kills, deaths, K/D, or match statistics.
- It only analyzes a video file you provide. It does not read game memory, access hidden game state, reveal enemies through walls, inject into the game, or bypass anti-cheat.
- First launch requires internet access to install dependencies and download model weights.

## Install / launch
1. Install **Python 3.11 (64-bit)** from https://www.python.org/downloads/windows/ and enable the Python launcher.
2. Extract this ZIP to a folder such as `C:\COD-AI-Analyst`.
3. Double-click `Launch-COD-AI.bat`.
4. Wait for dependencies to install. The first run can take several minutes.
5. Your browser should open the local dashboard. If it does not, open the address printed in the terminal, usually `http://localhost:8501`.
6. In **Video analysis**, upload an MP4/MOV/MKV/AVI recording and click **Run AI analysis**.

## Troubleshooting
- If `py -3.11` is not found, install Python 3.11 x64 and reopen the terminal.
- If video writing fails, try an MP4 encoded with H.264.
- If processing is slow, choose `yolov8n.pt`, reduce processing width to 640, and analyze every 3–5 frames.
- If Windows blocks the batch file, right-click it and choose **Run**; only run files you trust.
- To stop the app, close the terminal window.

## Project structure
- `main.py` — dashboard and analysis pipeline
- `requirements.txt` — Python dependencies
- `Launch-COD-AI.bat` — Windows launcher
- `outputs/` — generated video, heatmap, CSV, and JSON report

## Next improvements
A later version can add manual match event tagging, round-by-round notes, game-specific HUD OCR for user-visible stats, and a proper offline speech-to-text engine.


## Version 0.2 additions
- Input-style selector for PS5 controller or mouse & keyboard.
- In-app starting-point guidance for official in-game controller settings and mouse consistency.
- Experimental recorded-clip optical-flow analysis to estimate camera motion over time.
- Motion values are a visual proxy, not a direct recoil measurement; turning and other effects influence them.


## Build a Windows EXE
A precompiled Windows binary is not included because this project must be packaged and tested on Windows with its dependencies. To build it on your Windows PC:
1. Install Python 3.11 x64.
2. Double-click `Build-EXE.bat` and wait for dependencies and PyInstaller to install.
3. The output will be in `dist\\COD_AI_ANALYST\\`.
4. Run `COD_AI_ANALYST.exe` inside that folder. Keep the full folder together; the build is `onedir`, not a single-file EXE.
5. If the build fails, use `Launch-COD-AI.bat` as the reliable fallback and share the final error message for troubleshooting.

The first analysis may still need internet access to download YOLO model weights. Windows security software may scan newly built executables.
