from keras.models import load_model
from time import sleep
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


classifier =load_model('./inception_v3_model.h5')

class_labels = ['Bacterialspot','Lateblight','healthy']

cap = cv2.VideoCapture(2)



while True:
    #frame=cv2.imread("h.JPG")
    ret, frame = cap.read()
    labels = []
    
    
    roi_gray = cv2.resize(frame,(299,299),interpolation=cv2.INTER_AREA)


    if np.sum([roi_gray])!=0:
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

        preds = classifier.predict(roi)[0]
        print("\nprediction = ",preds)
        label=class_labels[preds.argmax()]
        print("\nprediction max = ",preds.argmax())
        print("\nlabel = ",label)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

























