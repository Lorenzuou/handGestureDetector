
import tensorflow as tf
import pandas as pd
from pickle import load

import numpy as np




model = tf.keras.models.load_model("model.hdf5")



# data = pd.read_csv("handData.csv")

sample = [364,204,338,156,300,114,270,80,261,47,277,120,221,143,237,151,257,149,276,154,221,171,237,178,256,177,278,190,222,200,236,206,256,205,280,224,223,237,190,247,161,252]
scaler = load(open('scaler.pkl', 'rb'))
sample = scaler.transform([sample])
predict_result = model.predict(np.array([sample[0]]))

print(predict_result)