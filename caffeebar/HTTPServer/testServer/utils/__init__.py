import time
from typing import *


def get_now_timestamp() -> str:
    """获取当前时间戳

    Returns:
        int: 
    """
    now = time.time()
    return str(int(now*1000))


def stringify(data) -> Dict[str, Any]:
    """保证可以JSON序列化
    """
    if isinstance(data, dict):
        return {k: stringify(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [stringify(x) for x in data]
    else:
        return str(data)


def table_id_map(id: int) -> Tuple[int]:
    """我们的桌子ID与其他课题组ID的映射

    Args:
        id (int): _description_

    Returns:
        int: _description_
    """
    assert 0 <= id <= 5
    return {
        0: [2],
        1: [1],
        2: [5, 6],
        3: [3, 4],
        4: [9],
        5: [7, 8]
    }[id]
