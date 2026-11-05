@echo off
setlocal
cd /d "%~dp0"
set "FRAME_STRIDE=1"
set "YOLO_IMGSZ=640"
set "MAX_VIDEO_WIDTH=1280"
where py >nul 2>&1
if not errorlevel 1 (
  set "PYTHON=py -3"
) else (
  set "PYTHON=python"
)
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
  echo Python 3 not found. Install Python 3.10 or 3.11 from python.org.
  pause
  exit /b 1
)
%PYTHON% check_setup.py
if errorlevel 1 (
  echo Installing missing packages into the existing Python installation...
  %PYTHON% -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo Package installation failed. Check the error above.
    pause
    exit /b 1
  )
)
echo.
echo Open http://127.0.0.1:5000 in your browser.
%PYTHON% app.py
if errorlevel 1 pause
