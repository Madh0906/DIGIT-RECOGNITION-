import numpy as np
import cv2
import os
import joblib
import shutil
model = joblib.load('iris_classifier_knn5.joblib')

IMG_DIR = r'C:\Users\SAI MADHU\Desktop\test\TEST2'
IMG_DIR1 = r'C:\Users\SAI MADHU\Desktop\test'
a=os.listdir(IMG_DIR)
#for i in a:
        #b=os.listdir(os.path.join(IMG_DIR,i))
        #print(b)
for img in a:
        fd=os.path.join(IMG_DIR,img)
        b=img
        #print(fd)
        img_array = cv2.imread(fd, cv2.IMREAD_GRAYSCALE)
        img_array = img_array.flatten()
        #print(len(img_array))
        img_array  = img_array.reshape(-1, 1).T
        res = model.predict(img_array)
        print(img,":",res[0])
        des=os.path.join(IMG_DIR1,str(res[0]),b)
        shutil.copy(fd,des)
print('done')
