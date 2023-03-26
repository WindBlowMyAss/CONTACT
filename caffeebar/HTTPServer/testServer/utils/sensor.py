from typing import *
import time

def get_multi() -> Tuple[bool, Dict[str, Any], int]:
    """获取综合传感器的读数

    Returns:
        Tuple[bool, Dict[str, Any], int]: status, data, timestamp
    """
    # TODO: 对接传感器
    return True, {"temp": 24.5}, int(time.time()*1000)