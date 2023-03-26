from typing import *

import cv2
import numpy as np

from utils.plugin import PluginBase
from utils.plots import Annotator, colors
"""
密度检测
"""
ENABLED = True     # 是否启用该插件

# CONFIG
DISTANCE_THRESHOLD = 40000
CONF_THRESHOLD = 0.5

# CONST

# 是否启用Mask
ENABLE_MASK = False

class Plugin(PluginBase):
    def __init__(self):
        super().__init__(
            enabled=ENABLED,
            sequence=0,      # 插件运行顺序，越小越靠前
            name="人群密度检测"
        )

        # 插件初始化（定义配置等）
        if ENABLE_MASK:
            mask = cv2.imread("plugins/densityMask/mask222.bmp", cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY_INV)
            self.mask = mask - cv2.erode(mask.copy(), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    def run(self, img: np.ndarray, bboxes: List[Tuple[List, List, float, int, str]], **kwargs) -> Tuple[str, np.ndarray]:
        # 每次帧执行一次
        annotator = Annotator(np.zeros(img.shape, dtype=img.dtype))

        densityDetList = []
        for xyxy, xywh, conf, clsID, clsName in bboxes:
            if clsName == "person" and conf > CONF_THRESHOLD:
                if not ENABLE_MASK or self.mask[xywh[1], xywh[0]] > 0:
                    densityDetList.append(xywh)
                    label = f'{clsName} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(clsID, bgr=True))
                    
        # layer = np.zeros(shape=img.shape, dtype=img.dtype)
        
        mindis = 1e10
        if len(densityDetList) > 0:
            for i in range(len(densityDetList)):
                for j in range(i + 1, len(densityDetList)):
                    dis = pow(densityDetList[i][0] - densityDetList[j][0], 2) + \
                        pow(densityDetList[i][1] - densityDetList[j][1], 2)

                    
                    if dis < mindis:
                        mindis = dis

                    if dis < DISTANCE_THRESHOLD:
                        color = (0,0,255)
                    else:
                        color = (0, 255, 0)
                    # cv2.line(layer, densityDetList[i][:2], densityDetList[j][:2], color=color, thickness=2, lineType=4)
                    # annotator.im
                    cv2.line(annotator.im, densityDetList[i][:2], densityDetList[j][:2], color=color, thickness=5, lineType=4)
                    

        # if ENABLE_MASK:
        #     if mindis < DISTANCE_THRESHOLD:
        #         layer[:, :, 2] += self.mask * 120
        #     elif len(densityDetList):
        #         layer[:, :, 1] += self.mask * 120

        return f"区域内有{len(densityDetList)}人，最近距离{mindis}", annotator.result(), {
            "dist_warn": mindis < DISTANCE_THRESHOLD
        }
