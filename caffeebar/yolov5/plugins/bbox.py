from typing import *

import numpy as np

from utils.plots import Annotator, colors
from utils.plugin import PluginBase

"""
显示检测结果的boudingbox
"""
ENABLED = True


KEPT_CLASS_LABELS = ()  # 需要保留的类（标签），留空表示全部保留
KEPT_CLASS_ID = ()      # 需要保留的类（ID）


class Plugin(PluginBase):
    def __init__(self):
        super().__init__(
            enabled=ENABLED,
            sequence=0,
            name="bbox可视化"
        )

    def run(self, img: np.ndarray, bboxes: List[Tuple[List, List, float, int, str]], **kwargs) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        annotator = Annotator(np.zeros(img.shape, dtype=img.dtype))
        count = {}
        for xyxy, xywh, conf, clsID, clsName in bboxes:
            if len(KEPT_CLASS_LABELS) == 0 or clsName in KEPT_CLASS_LABELS or clsID in KEPT_CLASS_ID:
                label = f'{clsName} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(clsID, bgr=True))
                count[clsName] = count.get(clsName, 0) + 1
        return "Detect: " + '\t'.join([f"{k}:{v}" for k, v in count.items()]), annotator.result(), {
            "bbs": [{
                "xyxy": xyxy,
                "conf": conf,
                "cls": cls,
                "name": label
            } for xyxy, _, conf, cls, label in bboxes]
        }
