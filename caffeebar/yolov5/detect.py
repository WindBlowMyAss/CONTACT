# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import base64
import json
import math
import os
import sys
import tempfile
import time
import tkinter
import traceback
from pathlib import Path
from tkinter import CENTER, messagebox
from turtle import numinput
from typing import *

import cv2
from cv2 import sqrt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import argmin, imag, isin

from models.common import DetectMultiBackend
from utils.AsynDataloader import AsyncLoader
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.SocketComm import AsynchronousSender
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


CENTER_POINTS = [       # 桌子的坐标(归一化后的)
    [0.1890625, 0.3990740740740741],
    [0.36041666666666666, 0.7592592592592593],
    [0.4348958333333333, 0.32314814814814813],
    [0.625, 0.5092592592592593],
    [0.625, 0.26851851851851855],
    [0.7421875, 0.3574074074074074]
]

CENTER_POINTS_222 = [
    []
]
# KEPT_CLASS_LABELS = ('cup', 'person', 'chair')   # 保留可视化的类
KEPT_CLASS_LABELS = ("chair", "toilet")
# KEPT_CLASS_LABELS = ()
TABLE_PERSON_THREDHOLD = [
    # 3, 3, 4, 4, 3, 3
    1, 1, 1, 1, 1, 1
]
TABLE_CROWD_FRAME_NUMBER_COUNT = 20    # 拥挤超过 200 帧报警
# 当桌子人数超过阈值时, 给count + 1, 当count超过 TABLE_CROWD_FRAME_NUMBER_COUNT时会报警,
# 提示 x桌子是拥挤的, 当桌子不再拥挤是, count - 10, 当 count < TABLE_CROWD_FRAME_NUMBER_COUNT时会报警不再报警
# 被报警的桌子上会画一个圈
COMIC_DIST = 350
# MASK 区域的淹没，是二值化图像
MASK = cv2.imread("mask222.bmp", cv2.IMREAD_GRAYSCALE)
_, MASK = cv2.threshold(MASK, 127, 1, cv2.THRESH_BINARY_INV)
MH, MW = MASK.shape
DENSITY_THRES = 2
VISUALIZE_DENSITY = False

MASK_CHAIR_ID = [(mask_id, cv2.threshold(cv2.imread(f"222_{mask_id}.bmp", cv2.IMREAD_GRAYSCALE), 127, 1, cv2.THRESH_BINARY_INV)[1]) for mask_id in range(1,5)]


# 初始化Socket连接参数和目标文件夹
HOST = "127.0.0.1"
PORT = 55555


def table_id_map(id: int) -> Tuple[int]:
    """我们的桌子ID与其他课题组ID的映射

    Args:
        id (int): _description_

    Returns:
        int: _description_
    """
    assert 0 <= id <= 5
    return {
        0: [2],
        1: [1],
        2: [5, 6],
        3: [3, 4],
        4: [9],
        5: [7, 8]
    }[id]


class DetectionAnalyzerBase:
    def update(self, img: np.ndarray, dets: np.ndarray, *args, **kwargs) -> None:
        pass

    def result(self) -> Dict[str, Any]:
        pass


class BondingBoxAnalyzer(DetectionAnalyzerBase):
    def update(self, img: np.ndarray, dets: np.ndarray, *args, **kwargs) -> None:
        return super().update(img, dets, *args, **kwargs)

    def result(self) -> Dict[str, Any]:
        return super().result()

# 计算目标区域内目标person的两两间距离
def distance_person(x, y):
    h1 = x[1]
    h2 = y[1]
    w1 = x[0]
    w2 = y[0]
    dist = ((h1 - h2) ** 2 + (w1 - w2) ** 2) ** 0.5
    return dist
    
@torch.no_grad()
def run(weights=ROOT / 'yolov5l.pt',  # model.pt path(s)
        source=ROOT / 'data/coco/images/train2021/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    # 非阻塞进程通信
    sender = AsynchronousSender(HOST, PORT)
    
    # 用于检测是否拥挤
    countCrownd = dict()
    crownedWarn = set()
    countDanger = dict()
    dangerWarn = set()

    # 用于暂时保存结果
    tOld = time.time()
    bigi = 0
    avg = 0

    #####################################################################################
    #                        yolo v5 代码区 （不建议修改
    #####################################################################################
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        dataset = AsyncLoader(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    for path, im, im0s, vid_cap, s in dataset:
        print("INFO_HERE_"*10)
        print(path, im.shape)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det) == 0:
                continue
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            cuplabel = 0
            cupnumb = 0
            guestnumb = 0
            desknumb = 0
            chairnumb = 0
           
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if names[int(c)] == 'cup':
                    cuplabel = 1
                    cupnumb = n
                if names[int(c)] == 'person':
                    guestnumb = n
                if names[int(c)] == 'dining table':
                    desknumb = n
                if names[int(c)] == 'chair':
                    chairnumb = n

            # Write results
            vh, vw, _ = annotator.im.shape
            numOfPerson = dict()    # 用于记录某个桌子的人数
            # 用于检测人群密度
            densityDetList = []
            distance_personList = []

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                # if cls == 0:    # person的cls id是0
                if names[int(cls)] in KEPT_CLASS_LABELS:
                    # 检测到person, 分配给最近的桌子

                    center = xywh[:2]  # 归一化(0~1)的中心点

                    def disto(p):
                        return pow(center[0] - p[0], 2) + pow(center[1] - p[1], 2)
                    dis2 = [disto(P) for P in CENTER_POINTS]
                    minidx = argmin(dis2)

                    numOfPerson[minidx] = numOfPerson.get(minidx, 0) + 1    # 计数
                    table_pos = CENTER_POINTS[minidx]
                    # annotator.im = cv2.arrowedLine(annotator.im,
                    #     pt1=[int(v) for v in [center[0]*vw, center[1]*vh]],
                    #     pt2=[int(v) for v in [table_pos[0]*vw, table_pos[1]*vh]],
                    #     color=(0,0,255),
                    #     thickness=3)

                    # 检测mask内的人数
                    cy = int(MH * center[1])
                    cx = int(MW * center[0])
                    if MASK[cy, cx] == 1:
                        densityDetList.append((cy, cx))
                    
                    # chair_id = -1
                    for mask_chair in MASK_CHAIR_ID:
                        if mask_chair[1][cy, cx] == 1:
                            chair_id = mask_chair[0]
                            cv2.putText(annotator.im, str(chair_id), (cx, cy), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=1.2, color=(0, 255, 255), thickness=3)

                if save_txt:  # Write to file
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    if names[c] in KEPT_CLASS_LABELS or True:
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        label = None if hide_labels else f"chair {conf:.2f}"
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator.box_label(xyxy, label, color=(255, 0, 0))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            if len(numOfPerson) > 0:
                mostPersonId = max(numOfPerson.keys(), key=lambda x: numOfPerson[x])
                mostCenter = CENTER_POINTS[mostPersonId]
                mostCenter = [int(mostCenter[0] * vw), int(mostCenter[1] * vh)]

                for k, v in numOfPerson.items():
                    countCrownd[k] = max(
                        0,
                        countCrownd.get(k, 0) + (1 if v > TABLE_PERSON_THREDHOLD[k] else -10)
                    )
                    if countCrownd[k] > TABLE_CROWD_FRAME_NUMBER_COUNT:
                        crownedWarn.add(k)
                    else:
                        if k in crownedWarn:
                            crownedWarn.remove(k)

                # 计算两两距离
                for i in range(len(densityDetList)):
                    for j in range(i+1, len(densityDetList)):
                        distance_personList.append(distance_person(densityDetList[i], densityDetList[j]))
                print(distance_personList)
                
                for i in range(len(distance_personList)):                  
                    countDanger[i] = max(
                        0,
                        countDanger.get(k, 0) + (1 if distance_personList[i] > COMIC_DIST else -10)
                    )
                    if countDanger[i] >TABLE_CROWD_FRAME_NUMBER_COUNT:
                        dangerWarn.add(i)
                    else:
                        if i in dangerWarn:
                            dangerWarn.remove(i)
                    
                # cv2.circle(annotator.im, mostCenter, radius=40, color=(0,0,255), thickness=3)
                print(f"{mostPersonId}号人最多, 有{numOfPerson[mostPersonId]}人")

                print("{}是拥挤的".format("和".join([f"{x}号桌" for x in crownedWarn])) if len(crownedWarn) else "没有桌子拥挤")

            # 显示密度
            if len(densityDetList) and VISUALIZE_DENSITY:
                # viewMask = np.zeros((MH, MW, 3), dtype=np.uint8)
                # viewMask[:,:,2] = MASK*255
                annotator.im[:, :, 2] += MASK * 40
                
            # for box in densityDetList:
            #     print(f"box:{box}")
            #     print(box[0], box[1])
            #     cv2.circle(annotator.im, (box[1], box[0]), radius=40, color=(0, 255, 0), thickness=-1)
            
            print(densityDetList)
            # 打印区域内人数
            print(len(densityDetList))
            # 计算两两距离
            for i in range(len(densityDetList)):
                for j in range(i+1, len(densityDetList)):
                    distance_personList.append(distance_person(densityDetList[i], densityDetList[j]))
            print(distance_personList)
                

            # sender.send(densityDetList)
            print(f"区域内有{len(densityDetList)}人！")

            # for table_id in crownedWarn:
            #     mostCenter = CENTER_POINTS[table_id]
            #     mostCenter = [int(mostCenter[0]*vw), int(mostCenter[1]*vh)]
            #     cv2.circle(annotator.im, mostCenter, radius=40, color=(0,0,255), thickness=3)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # 12.16新增

            # Stream results
            # im0 = annotator.result()
            # if chairnumb > 0:
            #     density = guestnumb / chairnumb
            #     if (cupnumb > 2) and (density > 1):
            #         cv2.putText(im0, "Uncomfortable Environment!", (1400, 50),
            #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0, 255, 255), thickness=3)
            #         cv2.putText(im0, "(Messy Desktop)", (1400, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                     fontScale=1.2, color=(0, 255, 255), thickness=3)
            #         cv2.putText(im0, "(Crowded Space)", (1400, 140), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                     fontScale=1.2, color=(0, 255, 255), thickness=3)
            #     elif (cupnumb > 2) and (density <= 1):
            #         cv2.putText(im0, "Uncomfortable Environment!", (1400, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                     fontScale=1.2, color=(0, 255, 255), thickness=3)
            #         cv2.putText(im0, "(Messy Desktop)", (1400, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                     fontScale=1.2, color=(0, 255, 255), thickness=3)
            #     elif (cupnumb < 2) and (density > 1):
            #         cv2.putText(im0, "Uncomfortable Environment!", (1400, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                     fontScale=1.2, color=(0, 255, 255), thickness=3)
            #         cv2.putText(im0, "(Crowded Space)", (1400, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                     fontScale=1.2, color=(0, 255, 255), thickness=3)
            #     else:
            #         cv2.putText(im0, "Comfortable Environment!", (1400, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                     fontScale=1.2, color=(0, 255, 255), thickness=3)
            # cv2.putText(annotator.im, "Noisy Table: 3", (800, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                     fontScale=1.2, color=(0, 255, 255), thickness=3)

            # cv2.putText(im0,str(cupnumb),(1400,150),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.2,color=(0,255,255),thickness=3)

            if view_img:
                im0 = cv2.resize(im0, (1080, 540), interpolation=cv2.INTER_CUBIC)  # 1080，540
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                
        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[0] != save_path:  # new video
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv2.VideoWriter):
                        vid_writer[0].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0)
            
        # 保存结果
        pred_dict = []
        for det in pred:
            for *xyxy, conf, cls in reversed(det):
                # tensor 转为 int
                cls = int(cls.item())
                if names[cls] not in KEPT_CLASS_LABELS:
                    continue
                xyxy = [int(t.item()) for t in xyxy]
                conf = float(conf.item())

                pred_dict.append({
                    "xyxy": xyxy,
                    "conf": conf,
                    "cls": cls,
                    "name": names[cls]
                })

        tables_status = []
        for tid in range(6):
            for rid in table_id_map(tid):
                # tid 我们定义的桌子id
                # rid 是真正的id
                tables_status.append({
                    "id": rid,
                    "num": numOfPerson.get(tid, 0),
                    "cr_status": "1" if tid in crownedWarn else "0",
                })

        now_time = int(time.time() * 1000)
        saved_result = {
            "timestamp": now_time,
            # "image": img_orig,  # np.ndarray
            "bbs": pred_dict,
            "tables": tables_status,
            "density": [{"id": x["id"], "de_status": x["cr_status"]} for x in tables_status],
        }
        sender.send(saved_result)
        # saved_result = {
        #     "time": now_time,
        #     "filename": f"{now_time}.jpg",
        #     # "ori_img": numpy_to_base64(img_orig),
        #     "pred": pred_dict,
        #     "message": "OK"
        # }
        # resultKept.put(saved_result)
        # imageKept.put(img_orig)

        # with open(f"saved_data/{now_time}.json", "w", encoding="utf8") as fp:
        #     json.dump(saved_result, fp, ensure_ascii=False, indent=4)
        # assert False
        tNew = time.time()
        tLast = tNew - tOld
        tOld = tNew
        avg = avg * (bigi / (bigi + 1)) + tLast / (bigi + 1)
        bigi += 1
        print(f"time:{tLast:.3f}s\tavg:{avg:.3f}")
            
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5m.pt', help='model path(s)')
    parser.add_argument(
        '--source',
        type=str,
        default=r"rtsp://admin:HK88888888@192.168.1.214:554",
        help='file/dir/URL/glob, 0 for webcam'
    )
    # parser.add_argument(
    #     '--source', type=str,
    #     default=r"/home/dbcloud/videos/position1-2.mp4",
    #     help='file/dir/URL/glob, 0 for webcam'
    # )
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # 额外参数

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
