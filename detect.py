import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import torch
import cv2
import numpy as np
import pandas as pd
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from boxmot import DeepOCSORT
# Adopted from YOLOv3 🚀 by Ultralytics, GPL-3.0 license
# Modified for head detection usage in gaze following
# mlist=glob.glob()
import argparse
#cv2.namedWindow()
def get_args_parser():
    parser = argparse.ArgumentParser('General Setup', add_help=False)
    parser.add_argument('--model_weights', type=str, default='/home/changfei/Tool_head_detector/head_detector_best.pt', help='path to load model weights')
    parser.add_argument('--input_img_folder', type=str, help='path to image folder')
    parser.add_argument('--txt_file', type=str, help='name of the txt file')
    return parser


def load_model(model_weights: str,
               imgsz=(640,640),  # inference size (pixels)
               device='cpu',
               ):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(model_weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # warmup
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  
    # print(model)
    return model, device, imgsz

def run_single_folder(tracker,
        model, 
        device, 
        dataset,
        csv_name,  # save results to an csv file in the output folder
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=3,  # maximum detections per image
        print_every=500 # print info
        ):
    
    t3 = time_sync()
    counter = 0
    total = len(dataset)
    df=pd.DataFrame(data=None,columns=['frame','xmin','ymin','xmax','ymax','id'])
    for path, im, im0s, vid_cap, _ in dataset:
        shape_mat=np.array([im0s.shape[1],im0s.shape[0],im0s.shape[1],im0s.shape[0]])
        # im0s=cv2.resize(im0s,(im.shape[2],im.shape[1]))
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        f_num=Path(path)
        # print(f_num.stem)
        # Inference
        pred = model(im, augment=False, visualize=False)
        print(pred)
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        
        for i, det in enumerate(pred): 
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
            cpu=det.cpu().numpy()
            # org_im=im.cpu().numpy()
            # print(org_im)
            # print(cpu)
            tracks=tracker.update(cpu,im0s)
        # print(tracks)
        
        xyxys = tracks[:, 0:4].astype('int') # float64 to int
        normals=xyxys / shape_mat
        # print(normal)
        ids = tracks[:, 4].astype('int') # float64 to int
        confs = tracks[:, 5]
        clss = tracks[:, 6].astype('int') # float64 to int
        inds = tracks[:, 7].astype('int') # float64 to int
        # print(df.shape)
        for normal, id, conf, cls in zip(normals, ids, confs, clss):
            new=[f_num.stem,normal[0],normal[1],normal[2],normal[3],id]
            df.loc[df.shape[0]]=new
        # in case you have segmentations or poses alongside with your detections you can use
        # the ind variable in order to identify which track is associated to each seg or pose by:
        # segs = segs[inds]
        # poses = poses[inds]
        # you can then zip them together: zip(tracks, poses)

        # print bboxes with their associated id, cls and conf
        if tracks.shape[0] != 0:
            for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                im0s = cv2.rectangle(
                    im0s,
                    (xyxy[0], xyxy[1]),
                    (xyxy[2], xyxy[3]),
                    (0,255,0)
                )
                cv2.putText(
                    im0s,
                    f'id: {id}, conf: {round(conf,2)}',
                    (xyxy[0], xyxy[3]+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0)
                )
        cv2.imshow('frame', im0s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter+=1
        if counter%print_every==0:
            LOGGER.info(f'Finished Processing {counter:d}/{total:d} ({time_sync()-t3:.3f}s)')
    cv2.destroyAllWindows()
    df.to_csv(csv_name)


        # Process predictions
        # for i, det in enumerate(pred):  # per image
        #     print(det)
        #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        #     p = Path(p)  # to Path
        #     txt_path = str(txt_name) # im.txt
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        #         # Print results
        #         for c in det[:, -1].unique():
        #             n = (det[:, -1] == c).sum()  # detections per class
                
        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             xy_normalized = tuple((torch.tensor(xyxy).view(1, 4) / gn).view(-1))  # normalized xywh
        #             with open(txt_path, 'a') as f:
        #                 f.write('%s'%p.stem) # number of frame
        #                 line = (cls, *xy_normalized)
        #                 f.write((', %g' * len(line))% line + '\n')
        # counter+=1
        # if counter%print_every==0:
        #     LOGGER.info(f'Finished Processing {counter:d}/{total:d} ({time_sync()-t3:.3f}s)')



if __name__ == "__main__":

    # parser = argparse.ArgumentParser('Run head detection', parents=[get_args_parser()])
    # args = parser.parse_args()

    # model, device, imgsz = load_model(args.model_weights)
    # dataset = LoadImages(args.input_img_folder, img_size=imgsz, stride=model.stride, auto=model.pt and not model.jit)
    # run_single_folder(model, device, dataset, args.txt_file)
    tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=True,
)   
    # tracker.update()
    # Load model
    model_weights = 'head_detector_best.pt'
    model, device, imgsz = load_model(model_weights)
    
    ################################################################################
    # Modify code here for different folder stucture
    # ################################################################################
    input_img_folder = r'F:\video_frames\Got05'
    csv_file = r'F:\video_frames\Got05.csv'

    dataset = LoadImages(input_img_folder, img_size=imgsz, stride=model.stride, auto=model.pt and not model.jit)

    run_single_folder(tracker,model, device, dataset, csv_file)
    # LOGGER.info('Done: %s'%input_img_folder)
