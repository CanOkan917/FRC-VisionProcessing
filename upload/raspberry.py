import cv2
import numpy as np
from networktables import NetworkTables

NetworkTables.initialize(server='roborio-3390-frc.local')
table = NetworkTables.getTable("Vision")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_EXPOSURE, -50.0)
font = cv2.FONT_HERSHEY_SIMPLEX
x_ = 0
y_ = 0
t = 0

while True:
    ret, frame = cam.read()

    height, width = frame.shape[:2]

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    vision_frame = cv2.inRange(hsv_frame, (50, 50, 25), (120, 255, 255))
    vision_frame = cv2.morphologyEx(vision_frame, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(vision_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            rectangle_pos = (center[0] + int(radius), center[1] - int(radius / 2)), (center[0] - int(radius), center[1] + int(radius / 2))

            if radius > 10: 
                middle_point = (int(width / 2), int(height / 2))
                pid_cord = center[0] - middle_point[0], center[1] - middle_point[1]
                x_ = pid_cord[0]
                y_ = pid_cord[1]
                t = 1
                print(f'target: x: {pid_cord[0]} | y: {pid_cord[1]}')
            else:
                x_ = 0
                y_ = 0
                t = 0
        else:
            x_ = 0
            y_ = 0
            t = 0
    else:
        x_ = 0
        y_ = 0
        t = 0
    table.putNumber("vision_X", x_)
    table.putNumber("vision_Y", y_)
    table.putNumber('vision_T', t)
