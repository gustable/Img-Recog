# Start the timer
import time
start_time = time.time()

#First Run Alex's Code to flatten and bunch the data
from dataset import *
os.environ["CIFIR_DATA_DIR"] = "/home/gp/Desktop/cifar-ten_pics/testing/"
samples = load(5000)
os.environ["CIFIR_DATA_DIR"] = "/home/gp/Desktop/cifar-ten_pics/testing/test/"
testData = load(1000)

#Then run an LR 
import numpy as np
import pylab as pl
from sklearn import linear_model, datasets

# import some data to play with
X = samples.data
Y = samples.target
Xtest = testData.data
Ytest = testData.target

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Then we test the fit on another dataset
logreg.predict(Xtest)
# and check our accuracy
print logreg.score(Xtest,Ytest) * 100, "% Accuracy"
# and Print the time taken to run the program
print time.time() - start_time, "seconds"

"""
gp@Pontifigore-Mint ~/Desktop/cifar-ten_pics $ python LRrotation.py
0.97
8.07875299454 seconds
gp@Pontifigore-Mint ~/Desktop/cifar-ten_pics $ python LRrotation.py
97.0 % Accuracy
1.23288083076 seconds
gp@Pontifigore-Mint ~/Desktop/cifar-ten_pics $ python LRrotation.py
97.0 % Accuracy
1.22457909584 seconds
gp@Pontifigore-Mint ~/Desktop/cifar-ten_pics $ python LRrotation.py
99.2 % Accuracy
24.152477026 seconds
gp@Pontifigore-Mint ~/Desktop/cifar-ten_pics $ 

"""
