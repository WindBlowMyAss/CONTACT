import cv2
# import face_recognition
from utils.SocketComm import SocketReciever

def main(recv):
    # cap = cv2.VideoCapture("rtsp://admin:HK88888888@192.168.64.240:554")
    cv2.namedWindow("win")
    # while True:
    for frame in recv:
        # frame = recv.get()
        if frame is not None:
            frame = cv2.resize(frame, (720, 405))
            anno = frame.copy()
            # face_locations = face_recognition.face_locations(frame)
            # print("I found {} face(s) in this photograph.".format(len(face_locations)))
            # for face_location in face_locations:
            #     top, right, bottom, left = face_location
            #     anno = cv2.rectangle(anno, (left, top), (right, bottom), (0,0,255), 3)
            cv2.imshow("win", anno)

        if ord('q') == cv2.waitKey(1):
            break

if __name__ == "__main__":
    # recv = AsynchronousReciever("127.0.0.1", port=5556)
    recv = SocketReciever("0.0.0.0", 5556, infinite=True)
    main(recv)
    