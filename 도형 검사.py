import cv2
import argparse
import imutils
import numpy as np

def onChange(x):
    pass

def isConvex(c, area):
    hull = cv2.convexHull(c)
    hArea = cv2.moments(hull)["m00"]
    return abs(hArea - area) / area <= 0.1

def labelPolygon(c, area):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 7:
        center = sum(approx) / 7
        leftCount = rightCount = 0
        
        for i in range(0, 7):
            if (approx[i] - center)[0, 0] >= 0:
                rightCount = rightCount + 1
            else:
                leftCount = leftCount + 1

        if leftCount > rightCount:
            return "Left"
        else:
            return "Right"


    if len(approx) > 7 and isConvex(c, area):
        return "Circle"

    return None

def multipleAnd(*argv):
    r = np.bitwise_and(argv[0],argv[1])
    for i in argv[2:]:
        r = np.bitwise_and(r,argv[1])
    return r

def maskImage(frame, h, error, sMin, vMin):
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowH = h - error if h - error >= 0 else 180 + (h - error)
    highH = h + error if h + error <= 180 else h + error - 180

    hMat = None
    if lowH <= highH:
        hMat = np.bitwise_and(lowH <= hsvImage[:,:,0], hsvImage[:,:,0] <= highH)
    else:
        hMat = np.bitwise_or(lowH <= hsvImage[:,:,0], hsvImage[:,:,0] <= highH)

    mask = multipleAnd(
        hMat,
        hsvImage[:,:,1]>sMin,
        hsvImage[:,:,2]>vMin
    )    

    gray = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    gray[mask]=255
    return gray

def findShapes(shapeStr, gray, low, high, original, BGR):
    cnts, hierachy = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ratio = 1 # ratio = image.shape[0] / float(resized.shape[0])
    found = 0

    for c in cnts:
        M = cv2.moments(c)

        if M["m00"] != 0 and low <= M["m00"] <= high:
            cX = int(M["m10"] / M["m00"] * ratio)
            cY = int(M["m01"] / M["m00"] * ratio)
            shape = labelPolygon(c, M["m00"])

            if shape == shapeStr:
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(original, [c], -1, BGR, 2)
                cv2.putText(original, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                found = found + 1

    return found

def putTextAtCenter(frame, text, color):
    textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]

    x = (frame.shape[1] - textSize[0]) / 2
    y = (frame.shape[0] - textSize[1]) / 2

    cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cv2.namedWindow("Result")
cv2.createTrackbar('Minimum Area', "Result", 10, 100000, onChange)
cv2.createTrackbar('Maximum Area', "Result", 10, 100000, onChange)

cv2.setTrackbarPos('Minimum Area', 'Result', 1000)
cv2.setTrackbarPos('Maximum Area', 'Result', 100000)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        low = cv2.getTrackbarPos('Minimum Area', 'Result')
        high = cv2.getTrackbarPos('Maximum Area', 'Result')

        redMasked = maskImage(frame, 0, 15, 180, 128)
        yellowMasked = maskImage(frame, 30, 15, 120, 60)
        greenMasked = maskImage(frame, 60, 15, 90, 60)
        greenInverse = 255 - greenMasked

        cv2.imshow("Found Red", redMasked)
        cv2.imshow("Found Yellow", yellowMasked)
        cv2.imshow("Found Green", greenMasked)

        redCount = findShapes("Circle", redMasked, low, high, frame, (0, 0, 255))
        yellowCount = findShapes("Circle", yellowMasked, low, high, frame, (131, 232, 252))
        leftCount = findShapes("Left", greenInverse, low, high, frame, (0, 255, 0))
        rightCount = findShapes("Right", greenInverse, low, high, frame, (0, 255, 0))
        greenCount = findShapes("Circle", greenMasked, low, high, frame, (0, 255, 0))

        if redCount > 0:
            putTextAtCenter(frame, "Red Light!", (0, 0, 255))
        if yellowCount > 0:
            putTextAtCenter(frame, "Yellow Light!", (131, 232, 252))
        if leftCount > 0:
            putTextAtCenter(frame, "Left Direction!", (0, 255, 0))
        elif rightCount > 0:
            putTextAtCenter(frame, "Right Direction!", (0, 255, 0))
        elif greenCount > 0:
            putTextAtCenter(frame, "Green Light!", (0, 255, 0))
        
        cv2.imshow("Result", frame)

    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break
