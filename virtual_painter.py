import numpy as np
import time
import cv2
import os
import HandTrackingModule as htm

###################################
brushThickness = 15
eraserThickness = 50
drawColor = (0,0,255)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
###################################



# Adding all the interface images in a list
folderPath = "Interface"
myList = os.listdir(folderPath)

overlayList = []
for imagePath in myList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)


header = overlayList[0]

# cv2 setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


# HandTracking module setup
detector = htm.handDetector(min_detection_confidence=0.85)

while True:
    ret, frame = cap.read()
    # flipping canvas so as to write properly (mirroring)
    frame = cv2.flip(frame, 1)


    # finding hand landmarks
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    
    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle finger
        x1, y1 = lmList[8][1], lmList[8][2]                    # index finger
        x2, y2 = lmList[12][1], lmList[12][2]                  # middle finger


        # checking which fingers are up
        fingers = detector.fingerUp()
        # print(fingers)


        # Selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")

            # checking for click
            if(y1 < 125):
                if(250 < x1 < 450):
                    header = overlayList[0]
                    drawColor = (0,0,255)
                elif(550 < x1 < 750):
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif(800 < x1 < 950):
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif(1050 < x1 < 1200):
                    header = overlayList[3]
                    drawColor = (0,0,0)

            cv2.rectangle(frame, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)


        # Drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")
            
            if xp==0 and yp==0:
                xp, yp = x1, y1
            

            if drawColor == (0,0,0):
                cv2.line(frame, (xp, yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), drawColor, eraserThickness)
            else:
                cv2.line(frame, (xp, yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), drawColor, brushThickness)


            xp, yp = x1, y1


    # overlaying our header image(default interface) into the screen
    frame[0:125, 0:1280] = header
    frame = cv2.addWeighted(frame, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("frame" , frame)
    # cv2.imshow("Canvas" , imgCanvas)



    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
