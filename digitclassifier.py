import matplotlib.pyplot as plt
from pylab import imread
from sklearn import datasets, svm, metrics
from scipy.misc import imread, imresize
import numpy as np
from PIL import Image
digits = datasets.load_digits()
#using MNIST data digits. 
n_samples = len(digits.images)
#print n_samples
data = digits.images.reshape((n_samples,-1))
print data
#reshaping data to be samples,features.
classifier = svm.SVC(gamma=0.001)
#training on first half of data.
classifier.fit(data[:n_samples/2], digits.target[:n_samples/2])
expected = digits.target[n_samples/2:]
predicted = classifier.predict(data[n_samples/2:])
col = Image.open("photo3.jpg")
gray = col.convert('L')
bw = np.asarray(gray).copy()
bw[bw < 128] = 0
bw[bw >= 128] = 255
imfile = Image.fromarray(bw)
imfile.save("bw.png")
img = imread("bw.png")
img = imresize(img, (64, 64))
predicted_image = classifier.predict(img)
print(predicted_image)
