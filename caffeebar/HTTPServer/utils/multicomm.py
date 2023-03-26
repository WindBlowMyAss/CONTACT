from utils.SocketComm import AsynchronousReciever, AsynchronousBase, AsynchronousSender
from typing import *
from multiprocessing import Process


pool: Dict[str, AsynchronousBase] = None

def init() -> None:
    global pool
    pool = dict()

def register(name:str, inst:AsynchronousBase=None, host:str=None, port:int=None, sor:str=None, **kwargs) -> AsynchronousBase:
    """注册非阻塞sender/reciever
    
    Args:
        name (str, optional): _description_.
        inst (AsynchronousBase, optional): sender/reciever实例. Defaults to None.
        host (str, optional): _description_. Defaults to None.
        port (int, optional): _description_. Defaults to None.
    """
    global pool
    assert name not in pool, f"{name} already exists"
    if inst is not None:
        if host is not None or port is not None:
            print("WARNING: `host` and `port` has been ignored while `inst` is given.")
    else:
        assert host is not None and port is not None, "`host` and `port` must be given while `inst` is not given"
        assert sor.lower() in ("reciever", "sender", "recv", "send"), f'`sor` must be one of ("recieve", "sender") not {sor}'
        if sor in ("reciever", "recv"):
            inst = AsynchronousReciever(host=host, port=port, **kwargs)
        else:
            inst = AsynchronousSender(host=host, port=port, **kwargs)
        pool[name] = inst
    return inst
        
def get(name:str) -> Process:
    global pool
    return pool[name]

def kill(name:str) -> bool:
    global pool
    assert name in pool
    if isinstance(pool[name], AsynchronousReciever):
        pool[name].kill()
    return True