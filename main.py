import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
from handModule import *
import tensorflow as tf
from pickle import load

key_relation = {72: 'h', 79: 'o', 77: 'm', 70: 'f', 66: 'b'}
# h 72 = hang loose
# o 79 = open hand
# f 70 = closed hand (fist)
# m 77 = middle finger
# b 66 = like


def idetifyHand(handData,model,scaler):
    array = np.array(handData)
    sample = array.reshape(-1)
    predict_result = model.predict(np.array([sample]))[0]

    y_classes = predict_result.argmax(axis=-1)

    for e in predict_result:
        if e > 1 or e < 0:
            return
    print(y_classes)
    print(predict_result)

    if(predict_result[0]> predict_result[1] and predict_result[0] >= 0.9):
        print("HANG LOOSE IDENTIFICADO")
    elif(predict_result[1]> predict_result[0] and predict_result[1] >= 0.9):
        print("MAO ABERTA IDENTIFICADA")
    else:
        print("-----")


def putOnDataset(data_t, handData, key):


    keys = key_relation.keys()

    if key in keys:
       #data.concat(handData, inplace=True)
        array = np.array(handData)

        sample = array.reshape(-1)
        sample = np.append(sample, key_relation[key])

        data_t.append(sample)
        print("Done: {}".format( key_relation[key]))


    # else:
    #     print("no")



model =  tf.keras.models.load_model("model.hdf5")
scaler = load(open('scaler.pkl', 'rb'))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = handDetector(maxHands=1,detectionCon=0.7)

data_arr = []
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    img = cv2.flip(img, 1)


    handPosition = detector.findPosition(img)

    key = cv2.waitKey(5)

    if key == 27:  # ESC
        break

    if(key != -1):
        putOnDataset(data_arr, handPosition, key)

    if(len(handPosition)>0):
        idetifyHand(handPosition, model, scaler)


    #print(detector.findPosition(img))

    cv2.imshow("Image", img)
    # cv2.waitKey(1)


new_data = pd.DataFrame(data_arr)

new_data.to_csv("HandData3.csv")



#taskkill /F /IM chrome.exe



