import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
#adding a ssl verification context
# if (not os.environ.get('PYTHONHTTPSVERIFY', '')and getattr(ssl,'create unverified context', None)):
#     ssl._create_default_https_context=ssl._create_unverified_context 
X=np.load("image.npz")["arr_0"]
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

nclass=len(classes)

Xtrain, Xtest, ytrain,ytest=train_test_split(X,y, random_state=1, test_size=2500, train_size=7500)
Xtrain_scaled=Xtrain/255.0
Xtest_scaled = Xtest/255.0

LogReg=LogisticRegression(solver='saga', multi_class='multinomial')
LogReg.fit(Xtrain_scaled, ytrain)
ypred=LogReg.predict(Xtest_scaled)

accuracy=accuracy_score(ytest, ypred)
print(accuracy)

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dimension=gray.shape
    height=dimension[0]
    width=dimension[1]
    upper_left=(int(width/2-56), int(height/2-56))
    bottom_right=(int(width/2+56), int(height/2+56))
    cv2.rectangle(frame, upper_left, bottom_right, (0,255,0), 2)
    roi=gray[upper_left[1]:bottom_right[1], upper_left[0]: bottom_right[0]]
    im_pil=Image.fromarray(roi)
    image_bw=im_pil.convert('L')
    image_bw_resized=image_bw.resize((22,30), Image.ANTIALIAS)
    image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
    pixel_filter=20
    min_pixel=np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-min_pixel,0,255)
    max_pixel=np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_pred=LogReg.predict(test_sample)
    print("predixted class is", test_pred)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

