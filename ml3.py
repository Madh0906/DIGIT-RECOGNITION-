from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
import joblib

image_size=28
no_of_different_labels=10
image_pixels=image_size*image_size
data_path="mnist/"
train_data=np.loadtxt("test/mnist_train.csv",delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",delimiter=",")


trainLabels=train_data[:,:1]
trainLabels=trainLabels.reshape(1,-1).flatten()
trainLabels=trainLabels.astype(int)
trainData=train_data[:,1:]
testLabels=test_data[:,:1]
testLabels=testLabels.reshape(1,-1).flatten()
testLabels=testLabels.astype(int)
testData=test_data[:,1:]

(trainData,valData,trainLabels,valLabels)=train_test_split(trainData,trainLabels,test_size=0.1,random_state=84)
valLabels=valLabels.astype(int)
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

kVals = range(1, 30, 2)
accuracies = []

for k in range(1, 30, 2):
          #train the k-Nearest Neighbor classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(trainData, trainLabels)
          # evaluate the model and update the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (5, score * 100))
    accuracies.append(score)
# find the value of k that has the largest accuracy

i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
accuracies[i] * 100))

# re-train our classifier using the best k value and predict the labels of the
# test data

model = KNeighborsClassifier(n_neighbors=1)
model.fit(trainData, trainLabels)
predictions = model.predict(testData)
score = model.score(testData, testLabels)
print("k=%d, accuracy=%.2f%%" % (5, score * 100))

print(predictions[1])

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits

print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

score = model.score(testData, testLabels)
print("k=%d, accuracy=%.2f%%" % (1, score * 100))

print ("Confusion matrix")
print(confusion_matrix(testLabels,predictions))

# loop over a few random digits

for i in np.random.randint(0, high=len(testLabels), size=(5,)):
         # grab the image and classify it
         image = testData[i]
         print(image)
         print(testData[i])
         prediction = model.predict([image])[0]
         
         imgdata = np.array(image, dtype='float')
         pixels = imgdata.reshape((28,28))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(prediction,(5,5),bbox={'facecolor':'white'},fontsize=16)
         print("i think tha digit is : {}".format(prediction))
         cv2.imshow("image", image)
         plt.show()
         cv2.waitKey(0)
joblib.dump(model, 'digit_model5.joblib')

