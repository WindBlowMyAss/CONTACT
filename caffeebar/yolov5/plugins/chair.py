from typing import *

import cv2
import numpy as np
import redis
import time

from utils.plots import Annotator
from utils.plugin import PluginBase

"""
空椅子检测
"""
ENABLED = True

VIEW_PERSON_LABEL = True        # 是否在画面中显示person的bbox
VIEW_CHAIR_LABEL = True         # 是否在画面中显示chair的bbox
VIEW_STATUS = True              # 是否在画面中显示椅子的状态

STATUS_QUEUE_LENGTH = 13

class Plugin(PluginBase):
    def __init__(self):
        super().__init__(
            enabled=ENABLED,    # 是否启用该插件
            sequence=0,      # 插件运行顺序，越小越靠前
            name="空椅子检测",
            fatal=True
        )

        # 插件初始化（定义配置等）
        self.chairMaskMap = {}
        self.chairMaskInfo = {}     # id:(left, top, width, height)
        self.chairStatusQueue = {}
        
        for id in range(1, 5):
            # 读取mask
            mask = cv2.imread(f"plugins/chairMask_V2/222_{id}.bmp", cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY_INV)
            self.chairMaskMap[id] = mask
            
            # 计算mask大小和位置
            rowMax = np.argmax(mask, axis=0)
            colMax = np.argmax(mask, axis=1)
            left = np.argmax(rowMax)
            top = np.argmax(colMax)
            maskWidth, maskHeight = 0, 0
            for i in range(left, mask.shape[1]):
                if rowMax[i] == 0:
                    maskWidth = i - left
                    break
            for j in range(top, mask.shape[0]):
                if colMax[j] == 0:
                    maskHeight = j - top
                    break
            self.chairMaskInfo[id] = (left, top, maskWidth, maskHeight)

            # 初始化状态队列
            self.chairStatusQueue[id] = []
        
    def run(self, img: np.ndarray, bboxes: List[Tuple[List, List, float, int, str]], db:redis.Redis=None, **kwargs) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        anno = Annotator(np.zeros(shape=img.shape, dtype=img.dtype))
        opt = kwargs.get("opt", None)
        if opt is not None:
            tableId = opt.tableId
        else:
            tableId = 0
            
        # 3种可能的状态：miss(椅子丢失)，available(椅子空闲)，busy(已被占用)
        chairStatus = {
            id: [False, False] for id in self.chairMaskMap.keys()   # [person, chair]
        }
        
        for xyxy, xywh, conf, clsID, clsName in bboxes:
            for id, mask in self.chairMaskMap.items():
                if mask[xywh[1], xywh[0]] == 1:
                    if clsName == "person":
                        chairStatus[id][0] = True
                        if VIEW_PERSON_LABEL:
                            anno.box_label(xyxy, "person")
                    elif clsName in {"chair", "toilet"}:
                        chairStatus[id][1] = True
                        if VIEW_CHAIR_LABEL:
                            anno.box_label(xyxy, f"chair {id}")
                         
        chairStatusStr = {}   
        for id, status in chairStatus.items():
            curStatus = "available" if  not status[0] and status[1] else "busy"
            self.chairStatusQueue[id].insert(0, curStatus)
            while len(self.chairStatusQueue[id]) > STATUS_QUEUE_LENGTH:
                self.chairStatusQueue[id].pop()
            
            if len(self.chairStatusQueue[id]) == STATUS_QUEUE_LENGTH:
                chairStatusStr[id] = max(("available", "busy"), key=lambda s: self.chairStatusQueue[id].count(s))
            else:
                chairStatusStr[id] = "pending"
        # chairStatusStr = {
        #     # id: "busy" if status[0] else ("available" if status[1] else "miss") for id, status in chairStatus.items()
        #     id: "busy" if status[0] else ("available" if status[1] else "busy") for id, status in chairStatus.items()
        # }
        
        if VIEW_STATUS:
            for id, mask in self.chairMaskMap.items():
                # anno.text(center, chairStatusStr[id])   #PIL only
                cv2.putText(anno.im, chairStatusStr[id], 
                            (self.chairMaskInfo[id][0], self.chairMaskInfo[id][1] + self.chairMaskInfo[id][3] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX , 2, color=(255,255,255), thickness=3)
        
        _db_key = f"table:{tableId}"
        if db is not None:
            for id in chairStatus.keys():
                db.hset(_db_key, mapping={
                    f"status:{id}": chairStatusStr[id],
                })
            db.hset(_db_key, "num", sum(1 for x in chairStatus.values() if x[0]))
            db.hset(_db_key, "ts", time.time())
            # 过期
            db.expire(_db_key, 3)
            
        return "椅子状态：{}".format(" ".join([f"{id}:{status}" for id, status in chairStatusStr.items()])), anno.result(), {
            f"table_{tableId}": {
                id: {
                    "status": chairStatusStr[id],
                } for id in chairStatus.keys()
            }
        }
