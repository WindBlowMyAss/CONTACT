from typing import *


def check_cam_id(id:Union[str, int]) -> bool:
    """检查摄像头id是否合法

    Args:
        id (int): _description_

    Returns:
        bool: _description_
    """
    if isinstance(id, str):
        if id.isdigit():
            id = int(id)
        else:
            return False
    elif isinstance(id, int):
        pass
    else:
        return False
    return 0 <= id <= 7


def getCamStatus(id:int) -> Tuple[bool, str]:
    """获取cam的状态

    Args:
        id (int): 

    Returns:
        Tuple[bool, str]: bool, rtsp url
    """
    #id = str(id)
    #return True, "rtsp://admin:HK88888888@192.168.1.{}:554".format({
    #    "0": 214,
    #}[id])
    return True, "rtmp://cn-cd-dx-3.natfrp.cloud:27032/hik01"
