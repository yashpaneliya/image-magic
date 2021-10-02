import cv2
import numpy as np
# import object_size
from tensorflow.keras.models import load_model
import cv2 as cv
import imutils
from keras.preprocessing.image import load_img, img_to_array
import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from tensorflow.keras import Model
from keras import optimizers
# from classification_models.tfkeras import Classifiers
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

from tensorflow import keras

# Display
# from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import IPython
# import vis
from tensorflow.keras.applications import imagenet_utils


def auto_canny(image, sigma=0.33):
# 	# compute the median of the single channel pixel intensities
 	v = np.median(image)
 	# apply automatic Canny edge detection using the computed median
 	lower = int(max(0, (1.0 - sigma) * v))
 	upper = int(min(255, (1.0 + sigma) * v))
 	edged = cv2.Canny(image, lower, upper)
 	# return the edged image
return edged

 def change_brightness(img, value=30):
     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
     h, s, v = cv2.split(hsv)
     v = cv2.add(v,value)
     v[v > 255] = 255
     v[v < 0] = 0
     final_hsv = cv2.merge((h, s, v))
     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
return img

def get_closed_image(image):
	clahe = cv2.createCLAHE(clipLimit=5)
	image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	final_img = clahe.apply(image_bw) + 20
	blur = cv2.GaussianBlur(final_img, (5, 5), 0)
	retValue, threshImg = cv2.threshold(blur, 220, 190, cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
	closed = cv2.morphologyEx(threshImg, cv2.MORPH_CLOSE, kernel)
	closed = cv2.erode(closed, None, iterations=10)
	closed = cv2.dilate(closed, None, iterations=10)
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0.7)
	# (T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
	# nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
	# sizes = stats[1:, -1]; nb_components = nb_components - 1
	# min_size = 1030
	# # min_size = 500
	#
	# img2 = np.zeros((output.shape))
	# for i in range(0, nb_components):
	#     if sizes[i] >= min_size:
	#         img2[output == i + 1] = 255
	# (T, threshInv) = cv2.threshold(gray, 155, 255,cv2.THRESH_BINARY_INV)
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
	# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	# closed = cv2.erode(closed, None, iterations = 14)
	# closed = cv2.dilate(closed, None, iterations = 14)
	return closed

def localizationOfTumor(img_path):
	image = cv2.imread(img_path)
	dim=(500,590)
	image=cv2.resize(image, dim)
	closed = get_closed_image(image)
	if np.count_nonzero(closed) > 1:
		cv2.imwrite("./ProcessedImage/closed.png", closed)
		contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# For each contour, find the convex hull and draw it
		# on the original image.
		contours_poly = [None] * len(contours)
		boundRect = [None] * len(contours)
		centers = [None] * len(contours)
		radius = [None] * len(contours)

		for i, c in enumerate(contours):
			contours_poly[i] = cv2.approxPolyDP(c, 3, True)
			boundRect[i] = cv2.boundingRect(contours_poly[i])
			centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

		for i in range(len(contours)):
			color = (0, 0, 255)
			# cv2.drawContours(img, contours_poly, i, color,1)
			# t[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
			cv2.circle(image, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 1)
		# edged = cv2.Canny(image, 100, 200)
		# canny = auto_canny(closed)
		# (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
		# cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
		# cv2.imshow("gray Image",image)
		cv2.imwrite("./ProcessedImage/image.png", image)

	else:
		cv2.imwrite("./ProcessedImage/closed.png", closed)
		contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# For each contour, find the convex hull and draw it
		# on the original image.
		contours_poly = [None] * len(contours)
		boundRect = [None] * len(contours)
		centers = [None] * len(contours)
		radius = [None] * len(contours)

		for i, c in enumerate(contours):
			contours_poly[i] = cv2.approxPolyDP(c, 3, True)
			boundRect[i] = cv2.boundingRect(contours_poly[i])
			centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

		for i in range(len(contours)):
			color = (0, 0, 255)
			# cv2.drawContours(img, contours_poly, i, color,1)
			# t[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
			cv2.circle(image, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 1)
		# edged = cv2.Canny(image, 100, 200)
		# canny = auto_canny(closed)
		# (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
		# cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
		# cv2.imshow("gray Image",image)
		cv2.imwrite("./ProcessedImage/image.png", image)

	cv2.waitKey(0) 
#
# def loadModel(model_path, nb_classes):
# 	# hyper parameters for model
# 	nb_classes = nb_classes   # number of classes
# 	img_width, img_height = 512, 512  # change based on the shape/structure of your images
# 	img_size = 512
# 	learn_rate = 0.0001  # sgd learning rate
#
# 	seresnet152, _ = Classifiers.get('seresnet152')
# 	base = seresnet152(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
# 	x = base.output
# 	x = layers.GlobalAveragePooling2D()(layers.Dropout(0.16)(x))
# 	x = layers.Dropout(0.3)(x)
# 	preds = layers.Dense(nb_classes, 'sigmoid')(x)
# 	model=Model(inputs=base.input,outputs=preds)
# 	loss= tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0)
# 	model.compile(optimizers.Adam(lr=learn_rate),loss=loss,metrics=[tf.keras.metrics.AUC(multi_label=True)])
#
#
# 	model.load_weights(model_path)
# 	return model
#
# def DetectTumor(img_path):
# 	# model = load_model('./brain_tumor_detector.h5')
# 	# model = load_model('./2019-8-6_resnet50.h5')
# 	model = loadModel(model_path = "./SeResNetBTYes_NoWeights.h5", nb_classes=2)
# 	x = load_img(img_path, target_size=(512, 512))
# 	x = img_to_array(x)
# 	x = np.expand_dims(x, axis=0)
# 	a = model.predict(x)
# 	a = a*100   ## Lets equalite all predictions on one stage which can be readable.
# 	if(a[0][0] >= 75):
# 		return "no", a[0][0]
# 	elif(a[0][1] >= 35):
# 		return "yes", a[0][1]
# 	else:
# 		return "Model is Confused"
#
# def PredictTumor(img_path):
# 	# image = gradCam(img_path, model)
# 	x = load_img(img_path, target_size=(512, 512))
# 	x = img_to_array(x)
# 	x = np.expand_dims(x, axis=0)
# 	model = loadModel(model_path = "./BrainTumor_SeResNet3_Weights.h5", nb_classes=3)
# 	a = model.predict(x)
# 	a = a*100   ## Lets equalite all predictions on one stage which can be readable.
# 	if(np.argmax(a) == 0 and a[0][0] >= 40):
# 	    print("Glioma Cancer Detected with {} confidence.".format(a[0][0]))
# 	elif(np.argmax(a) == 1 and a[0][1] >= 40):
# 	    print("Meningioma Cancer Detected with {} confidence.".format(a[0][1]))
# 	elif(np.argmax(a) == 2 and a[0][2] >= 40):
# 	    print("Pituitary Cancer Detected with {} confidence.".format(a[0][2]))
# 	else:
# 	    print("Model is Confused")
#
# def main(img_path):
# 	tumorOrNot, confidence = DetectTumor(img_path=img_path)
#
# 	if tumorOrNot == "yes":
# 		print("Tumor Found in {} with {} Confidence.".format(img_path, confidence))
# 		PredictTumor(img_path)
# 		print('Localization/Marking Of Tumor.')
# 		localizationOfTumor(img_path)
# 		print('Successfully Completed.')
# 	elif tumorOrNot == "no":
# 		print("No Tumor Detected in {} with {} Confidence.".format(img_path, confidence))
# 	else:
# 		print("Model is Confused")

img_path = "BT10.jpeg"
# img_path = "Meningioma.png"
# main(img_path)
localizationOfTumor(img_path)
