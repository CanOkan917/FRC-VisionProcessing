import cv2
import numpy as np

# img = cv2.imread("img/img.png")
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_EXPOSURE, -25.0)
font = cv2.FONT_HERSHEY_SIMPLEX
x = 0
y = 0

while True:
    ret, frame = cam.read()

    height, width = frame.shape[:2]

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    vision_frame = cv2.inRange(hsv_frame, (50, 50, 20), (120, 255, 255))
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
                cv2.rectangle(frame, rectangle_pos[0], rectangle_pos[1], (0, 255, 0), 2)
                # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                middle_point = (int(width / 2), int(height / 2))
                pid_cord = center[0] - middle_point[0], center[1] - middle_point[1]
                cv2.line(frame, middle_point, center, (0, 255, 255), 2)
                cv2.putText(frame, f'target: x: {pid_cord[0]} | y: {pid_cord[1]}', (10, height - 15), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # print(f'target: x: {pid_cord[0]} | y: {pid_cord[1]}')
                print(f'h: {height} | w: {width}')
    else:
        x = 0
        y = 0

    # crosshair
    cv2.line(frame, (int(width / 2), int((height / 2) + 10)), (int(width / 2), int((height / 2) - 10)), (0, 255, 0), 2)
    cv2.line(frame, (int((width / 2) + 10), int(height / 2)), (int((width / 2) - 10), int(height / 2)), (0, 255, 0), 2)

    cv2.imshow('vision detect', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()