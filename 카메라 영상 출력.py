import cv2
import numpy as np
import math

def Binary(target, title):
    binSuccess, dst = cv2.threshold(target, 80, 255, cv2.THRESH_BINARY)
    if binSuccess:
        cv2.imshow(title, dst)

cap = cv2.VideoCapture(0)

origWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
origHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, origWidth / 2)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, origHeight / 2)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("GrayScale", gray)

        box = cv2.blur(gray, (3, 3))
        cv2.imshow("Box", box)
        Binary(box, "Box Binary")

        gau = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imshow("Gaussian", gau)
        Binary(gau, "Gaussian Binary")

        med = cv2.medianBlur(gray, 3)
        cv2.imshow("Median", med)
        Binary(med, "Median Binary")

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        cv2.imshow("Edge Detection (Canny)", edges)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
        if lines is not None:
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)
            
        cv2.imshow("Camera Window", frame)

        

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()