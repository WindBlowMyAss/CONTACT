from typing import *
from utils import get_now_timestamp

def get_robo_status() -> Tuple[bool, Dict[str, Any], int]:
    """获取机器人状态

    Returns:
        Tuple[bool, Dict[str, Any], int]: status, data, timestamp
    """
    # TODO: 对接机器人
    status = True
    robos = [
        {
            "id": 0,
            "type": 0,
            "status": 1,
            "pos": [123.5, 234.6],
            "path": [112.0, 255, 333, 444]
        },
    ]
    timestamp = get_now_timestamp()
    return status, robos, timestamp