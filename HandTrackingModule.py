import cv2
import mediapipe as mp
import time



class handDetector():
    '''constructor'''
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence


        # mediapipe module for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode = self.static_image_mode, max_num_hands = self.max_num_hands, min_detection_confidence = self.min_detection_confidence, min_tracking_confidence = self.min_tracking_confidence)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.tipIds = [8, 12, 16, 20]
    
    def findHands(self, frame, draw=True):
        '''detects the hand in the frame and reflect it in the frame itself if draw=True'''
        # converting image from BGR to RGB and using mediapipe to process on this converted image
        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imageRGB)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for hand in self.result.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

        return frame

    
    def findPosition(self, frame, handNo=0, draw=True):
        '''returns the positions of different hand landmarks in the form of a 2D list'''
        self.lmList = []

        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])


                if(draw):
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


        return self.lmList


    def fingerUp(self):
        '''return the finger up'''

        fingers = []
      
        # for thumb
        if self.lmList[4][1] > self.lmList[4-1][1]:
            fingers.append(1)                                      # checking for index finger
        else:
            fingers.append(0)

        for tipId in self.tipIds:
            if self.lmList[tipId][2] < self.lmList[tipId-2][2]:
                fingers.append(1)                                  # checking for index finger
            else:
                fingers.append(0)

        return fingers



def main():
    # cv2 object
    cap = cv2.VideoCapture(0) 


    # frame rate
    prev_time = 0
    curr_time = 0



    # calling class
    detector = handDetector()
    


    while(True): 
        
        # Capture the video frame by frame 
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame,draw=False)
        if len(lmList) != 0:
            print(lmList[8])


        # frame rate display
        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time


        pos = (10, 70)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 255, 0)
        thickness = 1
        cv2.putText(frame, str(int(fps)), pos, font, fontScale, color, thickness)

        # Display the resulting frame 
        cv2.imshow('frame', frame) 

        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break



if __name__ == '__main__':
    main()