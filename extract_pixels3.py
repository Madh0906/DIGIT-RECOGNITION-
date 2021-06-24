import numpy as np
import cv2
import os

IMG_DIR = r'C:\Users\SAI MADHU\Desktop\py\Hand'
a=os.listdir(IMG_DIR)
for i in a:
        #b=os.listdir(os.path.join(IMG_DIR,i))
        #print(b)
        #for img in b:
                fd=os.path.join(IMG_DIR,i,img)
                #print(fd)
                img_array = cv2.resize(cv2.imread(fd, cv2.IMREAD_GRAYSCALE),(28,28))
                img_array = (img_array.flatten())
                print(len(img_array))
                img_array  = img_array.reshape(-1, 1).T
                with open(i+i+i+'.csv', 'ab') as f:
                        np.savetxt(f, img_array, delimiter=",")
print('done')
