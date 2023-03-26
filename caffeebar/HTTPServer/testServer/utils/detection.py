from typing import *
# from utils import multicomm
# from PIL import Image
# import cv2
# import numpy as np
import time


def get_detections(id: int, imageType: str = "b64") -> Tuple[bool, str, Dict[str, Any], int]:
    """获取目标检测结果

    Args:
        id (int): 摄像机id
        imageType (str, optional): 返回图片的类型. Defaults to "b64".

    Returns:
        Tuple[bool, str, AttrData, int]: status, img, data, timestamp
    """
    # detResult = multicomm.get("det").get()
    detResult = {
        "timestamp": int(time.time()*1000),
        'tables': [
            {'id': '2', 'num': '0', 'cr_status': '0'},
            {'id': '1', 'num': '0', 'cr_status': '0'},
            {'id': '5', 'num': '0', 'cr_status': '0'},
            {'id': '6', 'num': '0', 'cr_status': '0'},
            {'id': '3', 'num': '0', 'cr_status': '0'},
            {'id': '4', 'num': '0', 'cr_status': '0'},
            {'id': '9', 'num': '0', 'cr_status': '0'},
            {'id': '7', 'num': '0', 'cr_status': '0'},
            {'id': '8', 'num': '0', 'cr_status': '0'}
        ], 
        'bbs': [], 
        'density': [
            {'id': '2', 'de_status': '0'},
            {'id': '1', 'de_status': '0'},
            {'id': '5', 'de_status': '0'},
            {'id': '6', 'de_status': '0'},
            {'id': '3', 'de_status': '0'},
            {'id': '4', 'de_status': '0'},
            {'id': '9', 'de_status': '0'},
            {'id': '7', 'de_status': '0'},
            {'id': '8', 'de_status': '0'}
        ]
    }
    try:
        with open("desk.status", "r", encoding="utf8") as fp:
            ss = [x.strip() for x in fp.readlines()]
        for table in detResult["tables"]:
            if table["id"] in ss:
                table["cr_status"] = '1'
        print("{} has been modified".format(' '.join(ss)))
    except FileNotFoundError as e:
        print(e)

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
