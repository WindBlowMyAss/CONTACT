# from multiprocessing import Queue, Process
from threading import Thread, Lock
from queue import Queue
import cv2
from cv2 import VideoCapture
from utils.augmentations import letterbox
import numpy as np
import time


class AsyncLoader:
    _queue: Queue

    def __init__(self, src='streams.txt', img_size=640, stride=32, auto=True):
        self.sources = src
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.cap = VideoCapture(src)
        self._queue = Queue()
        self._process = Thread(target=self.update, daemon=True)
        self.count = 0

        assert self.cap.isOpened(), f'{src} Failed to open'
        # w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, img = self.cap.read()  # guarantee first frame
        assert ret
        self._process.start()

        # # check for common shapes
        # s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        # self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        # if not self.rect:
        #     print('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                while not self._queue.empty():
                    self._queue.get()
                self._queue.put(frame)
            time.sleep(0.01)

    def __iter__(self):
        return self

    # def __next__(self):
    #     self.count += 1
    #     img = self._queue.get()
    #     # Letterbox
    #     img0 = [img.copy()]
    #     img = [letterbox(x, self.img_size, stride=self.stride, auto=True)[0] for x in img0]

    #     # Stack
    #     img = np.stack(img, 0)

    #     # Convert
    #     img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    #     img = np.ascontiguousarray(img)

    #     return self.sources, img, img0, None, ''

    def __next__(self):
        self.count += 1
        # Letterbox
        img0 = self._queue.get().copy()
        img = letterbox(img0, self.img_size, stride=self.stride, auto=True)[0]

        # # Stack
        # img = np.stack(img, 0)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        # return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
        return 1
