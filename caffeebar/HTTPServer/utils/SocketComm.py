import pickle
import socket
import struct
from multiprocessing import Process, Queue
from typing import *


class SocketBase:
    host: str
    port: int
    con: socket.socket
    
    def __init__(self, host: str, port:int) -> None:
        """_summary_

        Args:
            host (str): _description_
            port (int): _description_
        """
        self.host = host
        self.port = port
        self.con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    

class SocketReciever(SocketBase):
    """
    Usage:
        for data in SocketReciever(host, port):
            print(data)
    """
    client: socket.socket = None
    clientAddr: str = None
    infinite: bool
    
    def __init__(self, host: str, port: int, n:int=5, infinite:bool=False) -> None:
        """
            infinite (bool, optional): True 时，即使sender断开连接也继续监听. Defaults to False.
        """
        super().__init__(host, port)
        self.infinite = infinite
        self.con.bind((self.host, self.port))
        self.con.listen(n)
        
    def __iter__(self):
        return self
    
    def __next__(self) -> Any:
        while True:
            if self.client is None:
                self.accept()
        
            try:
                result = self.recvMsg()
                if result is None:
                    raise ConnectionError()
                else:
                    return result
            except ConnectionError:
                self.client = None
                self.clientAddr = None
                if not self.infinite:
                    raise StopIteration()
    
    def _recvall(self, n) -> Any:
        """Helper function to recv n bytes or return None if EOF is hit

        Args:
            n (_type_): _description_

        Returns:
            Any: _description_
        """
        data = bytearray()
        while len(data) < n:
            packet = self.client.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
        
    def recvMsg(self) -> Any:
        # Read message length and unpack it into an integer
        raw_msglen = self._recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        msg = self._recvall(msglen)
        return pickle.loads(msg)
        
    def accept(self) -> None:
        self.client, self.clientAddr = self.con.accept()


class SocketSender(SocketBase):
    """
    Usage:
        sender = SocketSender(host, port)
        while not sender.connect():
            pass    
        print(f"socket://{sender.host}:{sender.port}\tconnected.")
        sender.send(somedata)    
    """
    connected: bool
    
    def __init__(self, host: str, port: int, initConnect:bool=False) -> None:
        super().__init__(host, port)
        self.connected = False
        if initConnect:
            assert self.connect()
    
    def connect(self) -> bool:
        """try to connect server

        Returns:
            bool: True is success
        """
        try:
            self.con.connect((self.host, self.port))
        except ConnectionError:
            self.connected = False
        else:
            self.connected = True
        return self.connected
    
    def sendMsg(self, data:Any):
        """

        Args:
            data (Any): _description_
        """
        msg = pickle.dumps(data)
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        self.con.sendall(msg)


class AsynchronousBase:
    pass

class AsynchronousSender(AsynchronousBase):
    """非阻塞式sender
    """
    socketSender: SocketSender
    
    def __init__(self, host:str, port:int) -> None:
        """
        Args:
            host (str): socket通信的host
            port (int): socket通信的端口号
        """
        self.socketSender = SocketSender(host, port)
    
    def send(self, data:Any) -> bool:
        """发送数据

        Args:
            data (Any): 数据

        Returns:
            bool: 是否成功发送
        """
        if not self.socketSender.connected:
            connected = self.socketSender.connect()
            if not connected:
                return False
        
        self.socketSender.sendMsg(data)
        return True
    

class AsynchronousReciever(AsynchronousBase):
    """非阻塞式reciever
    """
    # socketReciever: SocketReciever
    timeout: float
    _queue: Queue
    
    def __init__(self, host: str, port: int, n:int=5, qsize:int=4, timeout:float=0.5) -> None:
        """
        Args:
            host (str): 
            port (int): 
            n (int, optional): 
            qsize (int, optional): 数据队列长度，一般>=2即可. Defaults to 4.
            timeout (float, optional): 超时，单位秒；队列不空的情况下0.5s还未取得队列中的数据，则报EmptyError. Defaults to 0.5.
        """
        self._queue = Queue(qsize)
        self._process = Process(target=self._recvForever, args=(host, port, n))
        self._process.start()
        self.timeout = timeout
        
    def _recvForever(self, host:str, port:int, n:int) -> None:
        """接受Socket消息的子线程

        Args:
            host (str): 
            port (int): 
            n (int): 
        """
        for data in SocketReciever(host, port, n, infinite=True):
            # 清空未被接受的数据
            while not self._queue.empty():
                self._queue.get()
            # 最新数据入队
            # self._queue.put(pickle.dumps(data))
            self._queue.put(data)
    
    def get(self) -> Optional[Any]:
        """取得最新数据

        Returns:
            Optional[Any]: 队列中的最新数据，队列为空时返回None
        """
        if self._queue.empty():
            return None
        else:
            return self._queue.get(timeout=self.timeout)
    
    def kill(self) -> None:
        self._process.kill()
        