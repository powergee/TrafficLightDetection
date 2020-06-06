import cv2
import argparse
import imutils
import numpy as np

def onChange(x):
    pass

def labelPolygon(c):
    shape = None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    #if len(approx) == 3:
    #    shape == "triangle"

    #elif len(approx) == 4:
    #    (x, y, w, h) = cv2.boundingRect(approx)
    #    ar = w / float(h)
    #    shape = "sqaure" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    if len(approx) > 6:
        shape = "circle"

    return shape

def bitwise_and(*argv):
    r = np.bitwise_and(argv[0],argv[1])
    for i in argv[2:]:
        r = np.bitwise_and(r,argv[1])
    return r

def maskImage(frame, h, error):
    # blurred = cv2.medianBlur(frame, 9)
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lowH = h - error
    # highH = h + error

    # lower = (lowH, 75, 200)
    # upper = (highH, 255, 255)
    # gray = cv2.inRange(hsvImage, lower, upper)
    # thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    mask = bitwise_and(
        np.bitwise_or(
            hsvImage[:,:,0]<error,
            hsvImage[:,:,0]>180-error
        ),
        hsvImage[:,:,1]>180,
        hsvImage[:,:,2]>128
    )    

    gray = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    gray[mask]=255
    # gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    return gray
        
    #return cv2.bitwise_and(frame, frame, mask=masked)

def findAllShape(gray, low, high, original, BGR):
    cnts, hierachy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ratio = 1 # ratio = image.shape[0] / float(resized.shape[0])

    for c in cnts:
        M = cv2.moments(c)

        if low <= M["m00"] <= high:
            cX = int(M["m10"] / M["m00"] * ratio)
            cY = int(M["m01"] / M["m00"] * ratio)
            shape = labelPolygon(c)

            if shape != None:
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(original, [c], -1, BGR, 2)
                cv2.putText(original, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Result")
cv2.createTrackbar('Mininum Area', "Result", 10, 100000, onChange)
cv2.createTrackbar('Maxinum Area', "Result", 10, 100000, onChange)

cv2.setTrackbarPos('Mininum Area', 'Result', 10)
cv2.setTrackbarPos('Maxinum Area', 'Result', 100000)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        low = cv2.getTrackbarPos('Mininum Area', 'Result')
        high = cv2.getTrackbarPos('Maxinum Area', 'Result')

        masked = maskImage(frame, 0, 15)
        cv2.imshow("Found Red", masked)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.blur(gray, (5, 5))
        # thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        cv2.threshold(thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,thresh)
        cv2.imshow("Binary", thresh)

        findAllShape(masked, low, high, frame, (0, 0, 255))
        findAllShape(thresh, low, high, frame, (0, 255, 0))
        cv2.imshow("Result", frame)

    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break
