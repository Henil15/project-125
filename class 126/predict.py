
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import PIL.ImageOps
from PIL import Image

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

def getPrediction(image):
    impil = Image.open(image)
    imsca = impil.convert("L")
    imageresizing = imsca.resize((28 , 28) , Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(imageresizing , pixelfilter)
    imginvertedscale = np.clip(imageresizing - minpixel , 0,255)
    maxpixel = np.max(imageresizing)
    imginvertedscale = np.asarray(imginvertedscale)/maxpixel
    testsample = np.array(imginvertedscale).reshape(1 , 784)
    testpred = clf.predict(testsample)
    return testpred[0]
    
