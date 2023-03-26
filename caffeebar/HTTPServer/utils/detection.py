from typing import *
from utils import multicomm
from PIL import Image
import cv2
import numpy as np


def get_detections(id:int, imageType:str="b64") -> Tuple[bool, str, Dict[str, Any], int]:
    """获取目标检测结果

    Args:
        id (int): 摄像机id
        imageType (str, optional): 返回图片的类型. Defaults to "b64".

    Returns:
        Tuple[bool, str, AttrData, int]: status, img, data, timestamp
    """
    detResult = multicomm.get("det").get()
    if not isinstance(detResult, dict):
        return False, None, None, 0
    else:
        if imageType != 'none':
            # TODO
            img = "尚未实现"
        else:
            img = None
        data = {
            "bbs": detResult.get("bbs", []),
            "tables": detResult.get("tables", []),
            "density": detResult.get("density", [])
        }
        timestamp = detResult["timestamp"]
        return True, img, data, timestamp


# YUN_DET = cv2.FaceDetectorYN.create(
#     model="/home/dbcloud/caffeebar/HTTPServer/utils/face_detection_yunet_2021dec.onnx",
#     config='',
#     input_size=(320, 320),
#     score_threshold=0.8,
#     nms_threshold=0.3,
#     top_k=5000,
#     backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
#     target_id=cv2.dnn.DNN_TARGET_CPU
# )

# def face_detection(imgbuf:np.ndarray) -> List[Dict[str, Any]]:
#     """使用yuNet进行人脸检测

#     Args:
#         imgbuf (np.ndarray): 

#     Returns:
#         List[Dict[str, Any]]: 
#             [
#                 {
#                     "bb": [top, right, bottom, left],
#                     "face": Img
#                 },
#                 ...
#             ]
#     """
#     imgbuf = cv2.imdecode(imgbuf, cv2.IMREAD_COLOR)
#     YUN_DET.setInputSize((imgbuf.shape[1], imgbuf.shape[0]))
#     _, faces = YUN_DET.detect(imgbuf) # faces: None, or nx15 np.array
#     locations = []
#     if faces is not None:
#         for face in faces:
#             coords = face[:-1].astype(np.int32)
#             coords = [coords[1], coords[0]+coords[2], coords[1]+coords[3], coords[0]]
#             locations.append({
#                 "bb": coords,
#                 "face": imgbuf[coords[0]:coords[2], coords[3]:coords[1], :]
#             })
#     return locations