from tempfile import TemporaryDirectory
from typing import *
import os
import time
import json
from PIL import Image
import numpy as np
import cv2 


class KeptBase:
    Q: List[str]
    qsize: int
    _tempdirObj: TemporaryDirectory
    savepath: str
    ext: str
    
    def __init__(self, qsize:int, savepath:str=None, ext:str=""):
        """
        Args:
            qsize (int): 队列长度
            savepath (str, optional): 数据保存的位置, 
                如果为None则保存在/tmp并且会在析构时清除保存的数据. 
                Defaults to None.
        """
        self.Q = []
        self.qsize = qsize
        self.ext = f".{ext}" if len(ext) > 0 else ""
        if savepath is not None:
            if os.path.exists(savepath):
                # 文件夹已存在，则将已有的文件移入.old下级文件夹
                assert os.path.isdir(savepath), f"{savepath} exists and is not a dir"
                os.makedirs(os.path.join(savepath, ".old"), exist_ok=True)
                for each in os.listdir(os.path.join(savepath)):
                    if each == ".old":
                        continue
                    os.rename(os.path.join(savepath, each), os.path.join(savepath, ".old", each))
                print(f"WARNING: {savepath} exists, files in which are moved into '.old' folder")
            else:
                os.makedirs(savepath)
            self._tempdirObj = None
            self.savepath = savepath
        else:
            self._tempdirObj = TemporaryDirectory()
            self.savepath = self._tempdirObj.name
    
    def _dump(self, filepath:str, data:Any) -> None:
        with open(filepath, 'wb') as fp:
            fp.write(data)
        
    def put(self, data:Any) -> str:
        """向队列中放入数据，超过 self.qsize 量的数据会被丢弃

        Args:
            data (bytes): 要放入的数据
            ext (str): 文件后缀，None则自动(dict转json， 否则无后缀)

        Returns:
            str: 保存下来的文件名
        """
        if len(self.Q) >= self.qsize:
            filename = self.Q.pop(0)
            try:
                os.remove(os.path.join(self.savepath, filename))
            except FileNotFoundError:
                pass

        filename = f"{int(time.time()*1000)}{self.ext}"
        filepath = os.path.join(self.savepath, filename)
        self._dump(filepath, data)
        self.Q.append(filename)
        return filepath
    
    def _load(self, filepath:str) -> Any:
        with open(filepath, "rb") as fp:
            data = fp.read()
        return data
    
    def get(self) -> Optional[Any]:
        """取得最新的数据

        Returns:
        """
        if len(self.Q) > 0:
            while len(self.Q) > 0:
                filepath = os.path.join(self.savepath, self.Q[-1])
                if os.path.exists(filepath):
                    break
                else:
                    self.Q.pop()
            else:
                return None
            
            return self._load(filepath)
            
        else:
            return None

    def __del__(self):
        if self._tempdirObj is not None:
            self._tempdirObj.cleanup()
           
                
class JsonKept(KeptBase):
    def __init__(self, qsize: int, savepath: str = None):
        super().__init__(qsize, savepath, "json")
        
    def _dump(self, filepath: str, data: Dict[str, Any]) -> None:
        with open(filepath, "w", encoding="utf8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)
            
    def _load(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf8") as fp:
            data = json.load(fp)
        return data
    
    
class ImageKept(KeptBase):
    def __init__(self, qsize: int, savepath: str = None, ext: str = "jpg", fmt:str="BGR"):
        super().__init__(qsize, savepath, ext)
        assert fmt.lower() in ("bgr", "rgb")
        self.fmt = fmt.lower()
    
    def _dump(self, filepath: str, data: np.ndarray) -> None:
        if self.fmt == "bgr":
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(data)
        img.save(filepath)
    
    def _load(self, filepath: str) -> np.ndarray:
        img = Image.open(filepath)
        if self.fmt == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return np.asarray(img)