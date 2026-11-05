"""Optimized single-camera pose-based frame classification.
Frame skipping is approximate: see README for recall/accuracy implications.
"""
from functools import lru_cache
from pathlib import Path
import os
import subprocess
import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
FEATURES = [f'{axis}{i}' for i in range(17) for axis in ('x','y')]
SKELETON = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def env_int(name, default, low, high):
    try: return max(low, min(high, int(os.getenv(name,str(default)))))
    except (ValueError,TypeError): return default

@lru_cache(maxsize=1)
def get_models():
    # CUDA requires a CUDA-enabled PyTorch build; pip's CPU wheels cannot use the GPU.
    import torch
    choice=os.getenv('DEVICE','auto').strip().lower()
    if choice=='auto': device='0' if torch.cuda.is_available() else 'cpu'
    elif choice in ('cpu','mps') or choice.isdigit():device=choice
    else:device='cpu'
    classifier=joblib.load(BASE/'models'/'best_model.pkl')
    pose=YOLO(str(BASE/'models'/'yolo11n-pose.pt'))
    print(f'[Proctor-AI] YOLO device: {device}; stride: {env_int("FRAME_STRIDE",1,1,60)}; imgsz: {env_int("YOLO_IMGSZ",416,256,1280)}',flush=True)
    return pose, classifier, device

def infer(frame, threshold=0.55):
    pose, classifier, device=get_models()
    size=env_int('YOLO_IMGSZ',640,256,1280)
    result=pose.predict(source=frame,device=device,imgsz=size,conf=threshold,verbose=False)[0]
    if result.boxes is None or result.keypoints is None:
        return []
    boxes=result.boxes.xyxy.cpu().numpy()
    points=result.keypoints.xyn.cpu().numpy()
    n=min(len(boxes),len(points))
    if not n:return []
    raw=np.full((n,17,2),np.nan,dtype=np.float64)
    for i in range(n):
        count=min(17,len(points[i]));raw[i,:count]=points[i][:count,:2]
    # A single pandas frame + one classifier.predict for every person in frame.
    features=pd.DataFrame(raw.reshape(n,-1),columns=FEATURES)
    predictions=classifier.predict(features)
    return [(boxes[i],raw[i],int(predictions[i])) for i in range(n)]

def draw(frame, detections):
    h,w=frame.shape[:2]
    counts={'normal':0,'suspicious':0}
    for box,pts,pred in detections:
        suspicious=pred==1
        counts['suspicious' if suspicious else 'normal']+=1
        label='Suspicious' if suspicious else 'Normal'
        color=(0,0,255) if suspicious else (0,220,0)
        x1,y1,x2,y2=(int(v) for v in box)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
        cv2.rectangle(frame,(x1,max(0,y1-th-11)),(x1+tw+10,y1),color,-1)
        cv2.putText(frame,label,(x1+5,max(13,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        for a,b in SKELETON:
            if np.all(np.isfinite(pts[[a,b]])) and np.all(pts[[a,b]]>0):
                pa=(int(pts[a,0]*w),int(pts[a,1]*h))
                pb=(int(pts[b,0]*w),int(pts[b,1]*h))
                cv2.line(frame,pa,pb,(255,125,36),2)
        for point in pts:
            if np.all(np.isfinite(point)) and np.all(point>0):
                cv2.circle(frame,(int(point[0]*w),int(point[1]*h)),3,(0,255,0),-1)
    return frame,counts

def process_video(source,destination,preview,progress=None,frame_callback=None):
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():raise ValueError('Cannot read video: use MP4 (H.264), MOV or AVI.')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps<1 or fps>120:fps=25.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width<1 or height<1:cap.release();raise ValueError('Invalid video dimensions')
    width_limit=env_int('MAX_VIDEO_WIDTH',1280,320,1920)
    resize=min(1.,width_limit/max(width,height))
    w=max(2,int(width*resize)//2*2);h=max(2,int(height*resize)//2*2)
    # OpenCV's mp4v output often cannot play inline in Chrome/Edge/Firefox.
    # Write an intermediate file, then encode browser-compatible H.264 via FFmpeg.
    intermediate=destination.with_name(destination.stem+'-intermediate.mp4')
    writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():cap.release();raise RuntimeError('Could not initialize video writer')
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    stride=env_int('FRAME_STRIDE',1,1,60)
    # The last detection is redrawn on skipped frames: labels can lag motion by
    # up to stride-1 frames. This is NOT tracking or per-frame inference.
    last=[];frames=0;analyzed=0;flagged=0;max_people=0
    if progress:progress(0,total)
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frame=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
            if frames%stride==0:
                last=infer(frame); analyzed+=1
                if any(pred==1 for _,_,pred in last):flagged+=1
                max_people=max(max_people,len(last))
            frame,_=draw(frame,last)
            if frames==0:cv2.imwrite(str(preview),frame)
            writer.write(frame)
            if frame_callback is not None: frame_callback(frame)
            frames+=1
            if progress and frames%20==0:progress(frames,total)
    finally:cap.release();writer.release()
    if not frames:raise ValueError('No readable frames found')
    if progress:progress(frames,total)
    # imageio-ffmpeg ships an FFmpeg executable via pip on common platforms,
    # requiring no manual installation and no hard-coded executable paths.
    try:
        import imageio_ffmpeg
        ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
        command=[ffmpeg_executable,'-hide_banner','-loglevel','error','-y',
                 '-i',str(intermediate),'-an','-c:v','libx264','-preset',
                 os.getenv('H264_PRESET','veryfast'),'-crf',os.getenv('H264_CRF','25'),
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
        # Some alternate FFmpeg distributions may lack libx264; fail explicitly.
        proc=subprocess.run(command,capture_output=True,text=True)
        if proc.returncode!=0 or not destination.exists() or destination.stat().st_size==0:
            raise RuntimeError('FFmpeg H.264 encoding failed: '+(proc.stderr[-1800:] if proc.stderr else 'No playable video produced'))
    finally:
        intermediate.unlink(missing_ok=True)
    return {'frames':frames,'analyzed_frames':analyzed,'flagged_frames':flagged,
            'max_people':max_people,'seconds':round(frames/fps,1),
            'frame_stride':stride,'device':get_models()[2]}
