import argparse
import os
import sys
from pathlib import Path
import traceback
from typing import *
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import dbm
import pickle
import redis

from models.common import DetectMultiBackend
from utils.AsynDataloader import AsyncLoader
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync
# from utils.SocketComm import AsynchronousSender
# from utils.plugin import PluginBase

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
DB = redis.Redis("127.0.0.1", port=6379)

def init_plugins(pluginRoot:str="plugins", white_list:List[str]=[], black_list:List[str]=[]) -> List[Dict]:
    """初始化插件

    Args:
        pluginRoot (str, optional): 插件位置. Defaults to "plugins".

    Returns:
        List[Dict]
    """
    import importlib
    plugins = []
    for pluginName in os.listdir(pluginRoot):
        if (len(white_list) > 0 and pluginName in white_list) or (len(white_list) == 0 and pluginName not in black_list):
            pluginPath = os.path.join(pluginRoot, pluginName)
            if os.path.isfile(pluginPath) and pluginName.endswith(".py"):
                module = f"{pluginRoot}.{pluginName[:-3]}"
                try:
                    plugin = importlib.import_module(module)
                    if not plugin.ENABLED:
                        continue
                except ImportError:
                    LOGGER.warning(f"{pluginName}加载失败\n{traceback.format_exc()}")
                    continue
                else:
                    plugins.append(plugin.Plugin())
    
    sorted(plugins, key=lambda plugin: plugin.sequence)
    LOGGER.info(f"成功加载{len(plugins)}个插件："+"\t".join([f"{plug.name}({plug.enabled})" for plug in plugins]))
    return plugins


@torch.no_grad()
def run(opt,
        plugins=[]
        ):
    # sender = AsynchronousSender("127.0.0.1", 55555)
    winname = str(time.time())
    # 仅考虑2种输入：视频文件、直播流
    source = str(opt.source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size

    # Half
    half = opt.half and pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    if not is_file and is_url:
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = AsyncLoader(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

    vid_path, vid_writer = "", None

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0


    for path, im, im0s, vid_cap, _ in dataset:
        # 路径，预处理后的图像([batch_size,] channel, h, w)，原始图像，videoCapture，调试信息
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        det = pred[0]
        # if len(det) == 0:
        #     # continue
        #     LOGGER.info("未检测到任何目标")

        seen += 1
        p = Path(path)  # to Path
        im0 = im0s.copy()
        save_path = str(save_dir / p.name)  # im.jpg
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn = np.array(im0.shape)[[1, 0, 1, 0]]
        annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        
        bboxes = [(
            tuple(int(x) for x in xyxy),
            tuple(int(x) for x in xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()),
            float(conf),
            int(cls),
            names[int(cls)]
            ) for *xyxy, conf, cls in reversed(det)]    # xyxy的元素和conf是tensor类型
        
        ##########################################################
        ts = int(time.time() * 1000)
        baseLayer = np.zeros(im0.shape, dtype=im0.dtype)
        sharedData = {}
        for plugin in plugins:
            try:
                printMsg, annoLayer, *savedResult= plugin.run(im0, bboxes, **sharedData, db=DB, opt=opt)
            except Exception as e:
                if plugin.fatal:
                    raise e
                else:
                    LOGGER.warning(f"插件{plugin}出错")
                    LOGGER.warning(traceback.format_exc())
                    continue
            else:
                LOGGER.info(f"[{plugin}]\t{printMsg}")
            
            if isinstance(annoLayer, np.ndarray):
                if annoLayer.shape[:2] == im0.shape[:2]:
                    baseLayer[annoLayer>0] = annoLayer[annoLayer>0]
            elif annoLayer is None:
                pass
            else:
                LOGGER.warning(f"插件{plugin}返回了不正确的图层，期望图层type:nd.array, shape:{im0.shape}")
                
            if len(savedResult):
                savedResult = savedResult[0]
                if isinstance(savedResult, dict):
                    sharedData.update(savedResult)
                else:
                    LOGGER.warning(f"插件{plugin}返回了不正确的共享数据，期望type:Dict[str, Any]")
                
        im0[baseLayer>0] = baseLayer[baseLayer>0]

        LOGGER.debug(f"{sharedData=}")
        # sender.send({
        #     "timestamp": ts,
        #     **sharedData
        # })
        # 5秒过期
        DB.set("ts", ts)
        DB.set("data", pickle.dumps(sharedData))
        DB.expire("ts", 5)
        DB.expire("data", 5)
        
        # try:
        #     with dbm.open("/home/dbcloud/caffeebar/tmp.db", "c") as db:
        #         for k, v in sharedData.items():
        #             db[str(k)] = pickle.dumps((ts, v))
        # except FileNotFoundError as e:
        #     pass
        
        ###########################################################
        if opt.view_img:
            im0 = cv2.resize(im0, (1080, 540), interpolation=cv2.INTER_CUBIC)  # 1080，540
            cv2.imshow(winname, im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        # print("save video", opt.save_video)
        if opt.save_video:
        # if True:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                try:
                    vid_writer.write(im0)
                except KeyboardInterrupt:
                    vid_writer.release()
                    break

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if opt.update:
        strip_optimizer(opt.weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5l.pt', help='model path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5l.pt', help='model path(s)')
    parser.add_argument(
        '--source',
        type=str,
        default=r"rtsp://admin:HK88888888@192.168.1.222:554",
        help='file/dir/URL/glob, 0 for webcam'
    )
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
    parser.add_argument("--save-video", action='store_true')
    # 额外参数
    parser.add_argument("--white_list", nargs='+', type=str, default=[], help="plugin white list")
    parser.add_argument("--black_list", nargs='+', type=str, default=[], help="plugin black list")
    
    # chair.py
    parser.add_argument("--tableId", type=int, default=0)
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    
    plugins = init_plugins(
        white_list=opt.white_list,
        black_list=opt.black_list,
        # view_img=opt.view_img,
    )
    run(opt, plugins)
