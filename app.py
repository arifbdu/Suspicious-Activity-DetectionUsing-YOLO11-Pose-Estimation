"""Run from any directory: python path/to/app.py."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os, uuid, threading, time, logging
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from detector import process_video

BASE=Path(__file__).resolve().parent
UPLOADS=BASE/'uploads'; OUTPUTS=BASE/'outputs'
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=min(int(os.getenv('MAX_UPLOAD_MB','250')),1024)*1024*1024
ALLOWED={'.mp4','.avi','.mov','.mkv','.webm','.m4v'}
executor=ThreadPoolExecutor(max_workers=1)
jobs={}; guard=threading.Lock()
live={}; live_guard=threading.Lock()

class LiveFeed:
    def __init__(self):
        self.condition=threading.Condition()
        self.seq=0
        self.jpeg=None
        self.finished=False
    def publish(self, frame):
        # Limit network encoding to ~12fps. This does not affect saved MP4 frames.
        now=time.monotonic()
        if now-getattr(self,"last_publish",0)<0.08:return
        self.last_publish=now
        ok,enc=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,75])
        if ok:
            with self.condition:
                self.jpeg=enc.tobytes()
                self.seq+=1
                self.condition.notify_all()
    def finish(self):
        with self.condition:
            self.finished=True
            self.condition.notify_all()

def update_job(key, **fields):
    with guard: jobs[key].update(fields)

def work(key, src, dst, preview):
    with live_guard: feed=live[key]
    update_job(key,status='processing',message='Loading pose and KNN models…')
    try:
        def progress(done,total):
            percentage=round(100*done/total) if total else None
            update_job(key,frames_done=done,frames_total=total,percentage=percentage,message=f'Analyzing video: {done} / {total or "?"} frames')
        stats=process_video(src,dst,preview,progress,frame_callback=feed.publish)
        update_job(key,status='done',percentage=100,message='Completed: browser-ready H.264 video',result=stats)
    except Exception as exc:
        logging.exception('Detection job %s failed',key)
        dst.unlink(missing_ok=True);preview.unlink(missing_ok=True)
        update_job(key,status='error',message=f'{type(exc).__name__}: {exc}')
    finally:
        feed.finish()
        src.unlink(missing_ok=True)

@app.get('/')
def index():return render_template('index.html')

@app.post('/detect')
def detect():
    upload=request.files.get('video')
    if not upload or not upload.filename:return jsonify(error='Select a video.'),400
    extension=Path(secure_filename(upload.filename)).suffix.lower()
    if extension not in ALLOWED:return jsonify(error='Unsupported video format.'),400
    key=uuid.uuid4().hex
    src=UPLOADS/f'{key}{extension}';dst=OUTPUTS/f'{key}.mp4';preview=OUTPUTS/f'{key}.jpg'
    upload.save(src)
    with live_guard:live[key]=LiveFeed()
    with guard:jobs[key]={'status':'queued','percentage':0,'message':'Waiting for processing worker…','frames_done':0,'frames_total':None}
    executor.submit(work,key,src,dst,preview)
    return jsonify(job_id=key,status_url=url_for('status',key=key),stream_url=url_for('stream',key=key)),202


@app.get('/stream/<key>')
def stream(key):
    with live_guard: feed=live.get(key)
    if feed is None:return jsonify(error='Unknown stream ID'),404
    def frames():
        last_seen=0
        while True:
            with feed.condition:
                feed.condition.wait_for(lambda: feed.seq>last_seen or feed.finished, timeout=15)
                if feed.seq>last_seen:
                    last_seen=feed.seq
                    jpg=feed.jpeg
                else:
                    jpg=None
                ended=feed.finished
            if jpg is not None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpg)).encode()+b'\r\n\r\n'+jpg+b'\r\n'
            if ended:break
    response=Response(stream_with_context(frames()),mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control']='no-store, no-cache, must-revalidate'
    response.headers['X-Accel-Buffering']='no'
    return response

@app.get('/status/<key>')
def status(key):
    with guard: record=jobs.get(key)
    if not record:return jsonify(error='Unknown job ID (server may have restarted).'),404
    response=dict(record)
    if response.get('status')=='done':
        # Worker threads lack a request context, so URLs are created here.
        v=f'{key}.mp4';img=f'{key}.jpg'
        response['result']['video']=url_for('output',filename=v)
        response['result']['preview']=url_for('output',filename=img)
        response['result']['download']=url_for('download',filename=v)
    return jsonify(response)

@app.get('/outputs/<path:filename>')
def output(filename):
    if not filename.endswith(('.mp4','.jpg')):return '',404
    return send_from_directory(OUTPUTS,filename,mimetype='video/mp4' if filename.endswith('.mp4') else 'image/jpeg',conditional=True)

@app.get('/download/<path:filename>')
def download(filename):
    if not filename.endswith('.mp4'):return '',404
    return send_from_directory(OUTPUTS,filename,as_attachment=True,download_name='proctor_ai_annotated.mp4')

@app.errorhandler(413)
def too_large(_):return jsonify(error='File too large. Try a shorter video or increase MAX_UPLOAD_MB.'),413

if __name__=='__main__':
    app.run(host=os.getenv('HOST','127.0.0.1'),port=int(os.getenv('PORT','5000')),debug=False,threaded=True)
