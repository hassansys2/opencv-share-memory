import numpy as np
import cv2
import mmap

width = 640
height = 480
channels = 3
frame_size = width * height * channels

shared_memory = mmap.mmap(-1, frame_size, "FrameBuffer")

while True:
    shared_memory.seek(0)
    frame_bytes = shared_memory.read(frame_size)
    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, channels))

    cv2.namedWindow('Checkerboard-Python', cv2.WINDOW_NORMAL)
    cv2.imshow('Checkerboard-Python', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

shared_memory.close()
cv2.destroyAllWindows()
