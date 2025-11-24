import cv2

def build_pyramid(frame):
    level0 = frame
    level1 = cv2.pyrDown(level0)
    level2 = cv2.pyrDown(level1)
    return [level0, level1, level2]
