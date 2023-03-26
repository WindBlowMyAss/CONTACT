import time
from math import floor
from typing import *

import cv2
import numpy as np

from utils.plugin import PluginBase

"""
空桌检测
"""
ENABLED = True

# CONFIG
# VISUALIZATION = True
TABLE_PERSON_THREDHOLD = [
    # 3, 3, 4, 4, 3, 3
    1, 1, 1, 1, 1, 1
]
TABLE_CROWD_TIME_THRESHOLD = 5      # 拥挤超过 n 秒报警

# CONSTANT
CENTER_POINTS = [       # 桌子的坐标(归一化后的)
    [0.1890625, 0.3990740740740741],
    [0.36041666666666666, 0.7592592592592593],
    [0.4348958333333333, 0.32314814814814813],
    [0.625, 0.5092592592592593],
    [0.625, 0.26851851851851855],
    [0.7421875, 0.3574074074074074]
]
TID2MID_MAP = {     # 我们对桌子的编号与课题一对桌子的标号不同
    1: 1,
    2: 0,
    3: 3,
    4: 3,
    5: 2,
    6: 2,
    7: 5,
    8: 5,
    9: 4
}


class Plugin(PluginBase):
    def __init__(self):
        super().__init__(
            enbaled=ENABLED,
            sequence=0,      # 插件运行顺序，越小越靠前
            name="空桌检测",
            fatal=True
        )

        # 插件初始化（定义配置等）
        self.densityDetList = []
        self.countCrownded = [0] * len(CENTER_POINTS)
        self.cronwdedWarning = [False] * len(CENTER_POINTS)

    def run(self, img: np.ndarray, bboxes: List[Tuple[List, List, float, int, str]], **kwargs) -> Tuple[str, np.ndarray, Optional[Dict[str, Any]]]:
        layer = np.zeros(shape=img.shape, dtype=img.dtype)
        # 每次帧执行一次
        tablePersonCount = [0] * len(CENTER_POINTS)

        # 将person分配给最近的桌子并统计每个桌子的人数
        for xyxy, xywh, _, cls, label in bboxes:
            if label == "person":
                personPos = xywh[:2]
                
                # 将每个person分配到最近的桌子
                dis2table = [
                    pow(personPos[0] - x * img.shape[1], 2) + pow(personPos[1] - y * img.shape[0], 2)
                    for x, y in CENTER_POINTS]
                minidx = 0
                minval = dis2table[0]
                for i in range(1, len(dis2table)):
                    if dis2table[i] < minval:
                        minidx = i
                        minval = dis2table[i]
                tablePersonCount[minidx] += 1

                tablePos = CENTER_POINTS[minidx]    
                
                # 可视化分配规则
                layer = cv2.arrowedLine(layer,
                                        pt1=personPos,
                                        pt2=(round(tablePos[0] * img.shape[1]), round(tablePos[1] * img.shape[0])),
                                        color=(0, 0, 255),
                                        thickness=3)

        for k, v in enumerate(tablePersonCount):
            # 桌子附近的人超过阈值则累计时间，累计一定时间后报警
            if v > TABLE_PERSON_THREDHOLD[k]:
                if self.countCrownded[k] == 0:
                    self.countCrownded[k] = time.time()
                else:
                    if time.time() - self.countCrownded[k] > TABLE_CROWD_TIME_THRESHOLD:
                        self.cronwdedWarning[k] = True
            else:
                self.countCrownded[k] = 0
                self.cronwdedWarning[k] = False

        for i, v in enumerate(self.cronwdedWarning):
            tablePos = CENTER_POINTS[i]
            if v:
                cv2.circle(layer, (round(tablePos[0] * img.shape[1]), round(tablePos[1] * img.shape[0])),
                           radius=40, color=(0, 0, 255), thickness=-1)

        return '\t'.join([f"{i}号桌有{v}人" + "(拥挤)" if self.cronwdedWarning[i] else "(不拥挤)" for i, v in enumerate(tablePersonCount)]), layer, {
            "tables": [
                {
                    "id": tid,
                    "num": tablePersonCount[mid],
                    "cr_status": self.cronwdedWarning[mid]
                } for tid, mid in TID2MID_MAP.items()
            ]
        }
