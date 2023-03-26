from typing import *

import numpy as np

from utils.plugin import PluginBase

"""
插件模板
"""

ENABLED = False                 # 是否启用该插件

# CONFIG
SOME_CONFIG_HERE = None

# CONST
SOME_CONST_HERE = None

class Plugin(PluginBase):
    def __init__(self):
        super().__init__(
            enabled=ENABLED,
            sequence=0,         # 插件运行顺序，越小越靠前
            name="插件模板",     # 插件名字；缺省值文件名
            fatal=False,        # True则插件运行中出错会结束整个目标检测程序，False则会忽略错误（仅打印错误信息）；缺省值False
        )
        
        # 插件初始化
        # YOUR CODE HERE
        pass

    def run(self, img: np.ndarray, bboxes: List[Tuple[Tuple, Tuple, float, int, str]], **kwargs) -> Tuple[str, np.ndarray, Optional[Dict[str, Any]]]:
        # 每次帧执行一次
        layer = np.zeros(shape=img.shape, dtype=img.dtype)
        for xyxy, xywh, conf, clsID, clsName in bboxes:
            # DO SOMETHING
            pass
        
        # return debug信息:str, 可视化图层:np.ndarray, 输出信息:dict(optional)
        return "YOUR DEBUG MESSAGE HERE", layer, None
