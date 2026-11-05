"""Check environment without installing or changing packages."""
import importlib.util
import sys
MODULES = ["flask", "ultralytics", "torch", "cv2", "joblib", "numpy", "pandas", "sklearn", "imageio_ffmpeg"]
missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
if missing:
    print("Missing modules:", ", ".join(missing))
    sys.exit(1)
print("All required Python modules are already installed. Skipping installation.")
