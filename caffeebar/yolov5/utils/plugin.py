from abc import ABC, abstractmethod
from typing import *

import numpy as np


class PluginBase(ABC):
    enabled: bool
    sequence: int
    fatal: bool
    name: str

    def __init__(self, enabled: bool = True, sequence: int = 0, name:str = None, fatal:bool=False, **kwargs):
        self.enabled = enabled
        self.sequence = sequence
        self.fatal = fatal
        self.name = name or __file__[:-3]

    @abstractmethod
    def run(self, img: np.ndarray, bboxes: List[Tuple[List, List, float, int, str]], **kwargs) -> Tuple[str, np.ndarray]:
        pass
    
    def __str__(self) -> str:
        return self.name