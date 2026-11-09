import sys
import numpy
sys.path.append(r'E:\btl-httm\.venv\Lib\site-packages')
from pathlib import Path
import cv2,json,yaml
from ultralytics import YOLO
from src.tracking.track_manager import TrackManager
from src.behaviors.fall_run import MotionAndFallDetector
out=Path('data/fall_evaluation');out.mkdir(exist_ok=True)
cfg=yaml.safe_load(Path('configs/config.yaml').read_text(encoding='utf-8'))
report=[]
for name in ['fall_sample.mp4','01_video.mp4','02_video.mp4','people_walking.mp4']:
 cap=cv2.VideoCapture('data/test_videos/'+name); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); model=YOLO('weights/yolov8n-pose.pt'); tm=TrackManager(); det=MotionAndFallDetector({'enabled':False},cfg['behaviors']['falling']); events=[]; seen=set(); frames=0; thumbs=[]
 while True:
  ok,frame=cap.read()
  if not ok:break
  t=frames/fps
  r=model.track(frame,persist=True,verbose=False,conf=0.4,classes=[0],device='cpu')[0]
  if r.boxes.id is not None:
   for box,tid,k in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.id.int().tolist(),r.keypoints.data.cpu().numpy()):
    tr=tm.update_track(tid,tuple(box),(box[0],box[3]),k,t);det.process(tr,t,frame,[])
    if tr.fall_phase=='FALL_LATCHED' and tid not in seen:
     seen.add(tid);events.append({'track':tid,'seconds':round(t,3)});cv2.imwrite(str(out/(name+f'-alert-{tid}.jpg')),frame)
  if frames in [int(n*f) for f in (.1,.3,.5,.7,.9)]:
   thumb=cv2.resize(frame,(320,180));cv2.putText(thumb,f'{t:.1f}s',(5,20),0,.6,(0,0,255),2);thumbs.append(thumb)
  frames+=1
 cap.release()
 if thumbs:cv2.imwrite(str(out/(name+'-contact.jpg')),numpy.concatenate(thumbs,axis=1))
 row={'video':name,'fps':fps,'frames':frames,'seconds':round(frames/fps,2),'alerts':events};report.append(row);print(row,flush=True)
 (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
