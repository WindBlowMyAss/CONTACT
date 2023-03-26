# 插件

为了避免以后基于yolo的目标检测功能越来越多导致反复修改`detect.py`文件，方便后续添加修改功能，开发了插件系统。

## 插件的定义
插件是一个python模块(.py文件)且：
1. 必须包含一个布尔类型的成员变量`ENABLED`
2. 必须包含一个名为`Plugin`的类且继承于`utils.plugin.PluginBase`
3. 该`Plugin`类必须包含成员变量`enabled(bool)`, `sequence(int)`, `name(str)`, `fatal(bool)`
4. 该`Plugin`类必须包含`run`方法, 函数签名为
`def run(self, img: np.ndarray, bboxes: List[Tuple[List, List, float, int, str]], **kwargs) -> Tuple[str, np.ndarray]:`

## 插件的执行流程
`main.py`启动后会自动加载`yolov5/plugins`文件夹下的所有`ENABLED==True`的插件并实例化:
```python
plugin = importlib.import_module(module)
plugins.append(plugin.Plugin())
// ...
sorted(plugins, key=lambda plugin: plugin.sequence)
```
yolov5每完成一帧检测，会按照插件的`sequence`顺序(升序)运行一次`run`方法，传入参数为这一帧的图像`img`, 检测结果`bboxes`, 共享数据`sharedData`，返回用于打印在控制台的消息`printMsg`, 标记图层`annoLayer`，额外数据`savedResult`:
```python
printMsg, annoLayer, *savedResult= plugin.run(im0, bboxes, **sharedData)
```
其中`sharedData`是前面插件返回的额外数据(`savedResult`的合集), `printMsg`会打印到日志(控制台)，`annoLayer`是一个与帧图像尺寸相同的图层，在预览时会蒙在原始图像上，`savedResult`传递给接下来的插件并通过`Socket`发送给其他程序。

## 编写模板
模板文件`template.py`编写新的插件时可以复制一份该文件并在此基础上修改
```python
from typing import *

import numpy as np

from utils.plugin import PluginBase

"""
插件模板
"""


ENABLED = False                 # 是否启用该插件


class Plugin(PluginBase):
    def __init__(self):
        super().__init__(
            enabled=ENABLED,
            sequence=0,         # 插件运行顺序，越小越靠前
            name="插件模板",     # 插件名字；缺省值文件名
            fatal=False,        # True则插件运行中出错会结束整个目标检测程序，False则会忽略错误（仅打印错误信息）；缺省值False
        )
        
        # 插件初始化（定义配置等）
        # YOUR CODE HERE
        self.SOME_CONFIG = "SOME VALUE"

    def run(self, 
        img: np.ndarray, 
        bboxes: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], float, int, str]], 
        **kwargs) -> Tuple[str, np.ndarray, Optional[Dict[str, Any]]]::
        # 该方法每帧被调用一次
        # img: 帧图像, 类型np.ndarray
        # bboxes: 目标检测结果，类型list:
        #   bboxes[i][0]: (tuple)第i个bbox的xyxy
        #   bboxes[i][1]: (tuple)第i个bbox的xywh
        #   bboxes[i][2]: (float)第i个bbox的置信度
        #   bboxes[i][3]: (int)第i个bbox的类别ID
        #   bboxes[i][3]: (str)第i个bbox的类别标签
        layer = np.zeros(shape=img.shape, dtype=img.dtype)
        for xyxy, xywh, conf, clsID, clsName in bboxes:
            # DO SOMETHING
            pass
        
        # 返回值
        #  (str) 打印到控制台的字符串
        #  (np.ndarray) 与img0相同尺寸的标注图层
        #  (None or Dict[str, Any]) 额外数据，会传递给之后执行的插件并通过Socket传输给其他程序
        return "MESSAGE HERE", layer
```