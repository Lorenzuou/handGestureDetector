# handGestureDetector

This repository contains a tensorflow neural sequential model for learning hand gesture based on hand points position data originated from Google's mediapipe package, which includes hand detector API. 

It can be trained to detect a hand gesture by creating a dataset made from these hand gestures. To create that, it is necessary to use a camera and press a keyboard related to the especifc hand gesture. The model will then be trained using the samples generated and try to estimate the hand gestures, outputting a probability for each sample related to the likelihood of the points represent a hand gesture 
