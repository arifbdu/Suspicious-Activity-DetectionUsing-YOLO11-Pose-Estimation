@echo off
setlocal
cd /d "%~dp0"
set "FRAME_STRIDE=3"
set "YOLO_IMGSZ=480"
set "MAX_VIDEO_WIDTH=960"
where py >nul 2>&1
if not errorlevel 1 (set "PYTHON=py -3") else (set "PYTHON=python")
%PYTHON% --version >nul 2>&1
if errorlevel 1 (echo Python 3 not found.& pause& exit /b 1)
%PYTHON% check_setup.py
if errorlevel 1 (
  %PYTHON% -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (pause& exit /b 1)
)
echo FAST MODE: 1 of 3 frames analyzed. Not equivalent to paper evaluation.
echo Open http://127.0.0.1:5000
%PYTHON% app.py
if errorlevel 1 pause
