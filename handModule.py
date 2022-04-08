import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode= False, maxHands =2, detectionCon=0.5, trackConfidence=0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,1,self.detectionCon,self.trackConfidence)

        self.mpDrawn = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if ( self.results.multi_hand_landmarks):
            # print("Achou uma mão {}".format(count))
            # count += 1
            for handLms in self.results.multi_hand_landmarks:

                if(draw):
                  self.mpDrawn.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img,handNum=0, draw=True):

        lmList = []
        if  self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([cx,cy])
        return lmList
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    count = 0

    detector = handDetector()
    while True:
        success, img = cap.read()

        img = detector.findHands(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        print("ue")
        print(detector.findPosition(img))
        cv2.putText(img,str(int(fps)),(10,40),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__ == "__main__" :
    main()