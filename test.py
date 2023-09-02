import re
import streamlit as sl
import tempfile
import cv2
import os
from pathlib import Path
from boxmot import DeepOCSORT
import detect
import numpy as np
import pandas as pd
import torch
import sys
import time
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
def click():
    sl.session_state.clicked=True
# @sl.cache_data
def head_detect(_tracker,f_num,df,img0,_model,_device, 
    imgsz,conf_thres=0.25,
    iou_thres=0.45,
    max_det=3,
    ):
    img = letterbox(img0, imgsz, stride=_model.stride, auto=_model.pt and not _model.jit)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    shape_mat=np.array([img0.shape[1],img0.shape[0],img0.shape[1],img0.shape[0]])
    img = torch.from_numpy(img).to(_device)
    img = img.float()  # uint8 to fp16/32
    img /= 255 
    if len(img.shape) == 3:
        img = img[None]
    pred = _model(img, augment=False, visualize=False)
    # sl.write(pred.cpu().numpy())
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
    for j, det in enumerate(pred): 
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        cpu=det.cpu().numpy()
        tracks=_tracker.update(cpu,img0)
    xyxys = tracks[:, 0:4].astype('int') # float64 to int
    normals=xyxys / shape_mat
    # print(normal)
    ids = tracks[:, 4].astype('int') # float64 to int
    confs = tracks[:, 5]
    clss = tracks[:, 6].astype('int') # float64 to int
    # inds = tracks[:, 7].astype('int')  float64 to int
    # print(df.shape)
    for normal, id, conf, cls in zip(normals, ids, confs, clss):
        new=[f_num,normal[0],normal[1],normal[2],normal[3],id]
        df.loc[df.shape[0]]=new
if 'clicked' not in sl.session_state:
    sl.session_state.clicked=False
sl.sidebar.header("Gaze Follow")
sl.header("Gaze Follow")
file=sl.file_uploader("Video",type=['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv'])
if file==None:
    sl.write("Please upload a video.")
else:
    tfile=tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    sl.video(file, format='video/mp4', start_time=0)

    vid = cv2.VideoCapture(tfile.name)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=int(vid.get(cv2.CAP_PROP_FOURCC))
    frame_num=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frame_num)
    frame_bar=sl.progress(0,text="Progressing frame 0 out of %d"%frame_num)
    cur_frame=0
    frame_list = []
    if (vid.isOpened()==False):
        sl.write("Something Wrong occured while reading the video.")
    while (vid.isOpened()):
        success, frame = vid.read()
        if success:
            frame_list.append(frame)
            cur_frame+=1
            frame_bar.progress(cur_frame/frame_num,text="Progressing frame %d out of %d"%(cur_frame,frame_num))  
        else:
            break
    vid.release()
    sl.write('Frame List Created.')
    # sl.write(frame_list[0])
    sl.divider()
    tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=True,
)   
    head_model = 'head_detector_best.pt'
    model, device, imgsz = detect.load_model(head_model)
    t3 = time_sync()

    print_every=500 # print info
    # placeholder=sl.empty()
    df=pd.DataFrame(data=None,columns=['frame','xmin','ymin','xmax','ymax','id'])
    d_prog=sl.progress(0)
    for i in range(frame_num):
        img0=frame_list[i]
        new=head_detect(_tracker=tracker,df=df,f_num=i+1,img0=img0,_model=model,
                                                    _device=device,imgsz=imgsz)
        d_prog.progress((i+1)/frame_num,text="Detecting heads from frame %d"%(i+1))
        # placeholder.image(img0,"Frame %d/%d"%(i+1,frame_num),channels="BGR")
        # if (i+1)%print_every==0:
        #     placeholder.write(f'Finished Processing {(i+1):d}/{frame_num:d} ({time_sync()-t3:.3f}s)')
    sl.write("Head detection done.")
    sl.divider()

    skip=False
    df['id']=df['id'].map(int)
    label=df
    recorded=['kid','teacher','neither']
    for ind in range(df.shape[0]):
        if skip:
            continue
        if label.iloc[ind]['id'] not in recorded:
            recorded.append(label.iloc[ind]['id'])
            frame_ind=int(label.iloc[ind]['frame'])
            img=frame_list[frame_ind-1].copy()
            w,h=img.shape[1],img.shape[0]
            data=df[df['frame']==frame_ind]
            if data.shape[0]!=0:
                l_data=label[df['frame']==frame_ind] 
                if data.shape[0]>2 and 'kid' in l_data['id'].unique() and 'teacher' in l_data['id'].unique():
                    label.loc[label['id']==label.iloc[ind]['id'],'id']='neither'
                    continue
            for num in data.index:
                [xm,ym,xM,yM]=[int(data['xmin'][num]*w),int(data['ymin'][num]*h),
                               int(data['xmax'][num]*w),int(data['ymax'][num]*h)]
                cv2.rectangle(img,
                        (xm,ym),
                        (xM,yM),
                        (0,255,0),2)
                if num==ind:
                    cv2.putText(img,
                    'id:%d'%data['id'][num],
                    (xm,yM+50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,(0,255,0),3)
            cv2.putText(img,
                    'frame:%d'%frame_ind,
                    (0,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,(0,0,255),3)
            
            cv2.putText(img,
                            "Is the headbox with id:%d the kid,the teacher or neither"%label.iloc[ind]['id'],
                            (0,img.shape[0]-80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(255,255,255),2)
            cv2.putText(img,
                            "Type k if kid,t if teacher,n if neither,q to quit,s to skip",
                            (0,img.shape[0]-30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(255,255,255),2)
            cv2.namedWindow('img',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img', 800, 600)
            time.sleep(0.01)
            #print("Is the headbox with id:%d the kid,the teacher or neither?\nType k if kid,t if teacher,n if neither,q to quit: "%label.iloc[num]['id'])
            cv2.imshow('img',img)
            order=-1
            while order not in [ord('q'),ord('k'),ord('t'),ord('n'),ord('s')]:
                order=cv2.waitKey(0) & 0xFF
            if order==ord('q'):
                sys.exit(0)
            elif order==ord('k'):
                if data.shape[0]>1:
                    re_ids=data[data['id']!=label.iloc[ind]['id']]
                    for id_ind in re_ids.index:
                        matrix= (df['id']==re_ids.loc[id_ind]['id']) & (df['frame']>=re_ids.loc[id_ind]['frame']) & (label['id']=='kid') 
                        if 'teacher' in data['id'].map(str).unique():
                            label.loc[matrix,'id']='neither'
                        else:
                            label.loc[matrix,'id']='teacher'
                label.loc[df['id']==label.iloc[ind]['id'],'id']='kid'
                #print(label[:50])
            elif order==ord('t'):
                if data.shape[0]>1:
                    re_ids=data[data['id']!=label.iloc[ind]['id']]
                    for id_ind in re_ids.index:
                        matrix= (df['id']==re_ids.loc[id_ind]['id']) & (df['frame']>=re_ids.loc[id_ind]['frame']) & (label['id']=='teacher')
                        if 'kid' in data['id'].map(str).unique():
                            label.loc[matrix,'id']='neither'
                        else:
                            label.loc[matrix,'id']='kid'
                label.loc[df['id']==label.iloc[ind]['id'],'id']='teacher'
            elif order==ord('n'):
                label.loc[df['id']==label.iloc[ind]['id'],'id']='neither'
            elif order==ord('s'):
                skip=True
                continue
            cv2.destroyAllWindows()
    sl.write("Labelling done.")
    sl.divider()
    # sl.write(label)
    # sl.write(fourcc,fps,height,width)
    out=cv2.VideoWriter('out.mp4',fourcc,fps,(width,height))
    for i in range(frame_num):
        img=frame_list[i]
        frame_ind=i+1
        frame_anno=label[label['frame']==frame_ind]
        for num in frame_anno.index:
            [xm,ym,xM,yM]=[int(frame_anno['xmin'][num]*w),int(frame_anno['ymin'][num]*h),
                           int(frame_anno['xmax'][num]*w),int(frame_anno['ymax'][num]*h)]
            cv2.rectangle(img,
                    (xm,ym),
                    (xM,yM),
                    (0,255,0),2)
            cv2.putText(img,
                '%s'%frame_anno['id'][num],
                (xm,yM+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)
        out.write(img)
    out.release()
    sl.video('out.mp4', format='video/mp4', start_time=0)