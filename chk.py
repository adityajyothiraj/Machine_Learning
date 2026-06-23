# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
import os

# Path to label map file
PATH_TO_LABELS = os.path.join('labelmap.txt')

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

# load model
interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0) 
while True:
	# capture image
	ret, img_org = cap.read()
#	cv2.imshow('image', img_org)
	key = cv2.waitKey(1)
	if key == 27: # ESC
		break

	# prepara input image
	img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (300, 300))
	img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
	img = img.astype(np.uint8)
		
	# set input tensor
	interpreter.set_tensor(input_details[0]['index'], img)

	# run
	interpreter.invoke()

	# get outpu tensor
	boxes = interpreter.get_tensor(output_details[0]['index'])
	classes = interpreter.get_tensor(output_details[1]['index'])
	scores = interpreter.get_tensor(output_details[2]['index'])

	
	for i in range(boxes.shape[1]):
         if scores[0, i] > 0.5:
              box = boxes[0, i, :]

              x0 = int(box[1] * img_org.shape[1])
              y0 = int(box[0] * img_org.shape[0])
              x1 = int(box[3] * img_org.shape[1])
              y1 = int(box[2] * img_org.shape[0])

              cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)

              object_name = labels[int(classes[0, i])]

              cv2.putText(
              img_org,
              object_name,
              (x0, y0),
              cv2.FONT_HERSHEY_SIMPLEX,
              1,
              (255, 255, 255),
              2)
              print(object_name)
            
	
	#cv2.imwrite('output.jpg', img_org)
cv2.imshow('image', img_org)
		
cap.release()
cv2.destroyAllWindows()

