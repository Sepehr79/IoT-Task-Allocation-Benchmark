from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

clf.fit(X_train, y_train)


import psutil
import time
import os

freq = psutil.cpu_freq()
process = psutil.Process(os.getpid())
print(f"Current: {freq.current:.2f} MHz")
print(f"Minimum: {freq.min:.2f} MHz")
print(f"Maximum: {freq.max:.2f} MHz")
print("#########################################################")


def measure_performance(function, description, N = 10000):
    process.cpu_percent(None)
    start_wall = time.perf_counter()
    for _ in range(N):
        function()
    wall = time.perf_counter() - start_wall
    usage = process.cpu_percent(None)
    mean_time = wall/N
    required_capacity = freq.current*1e6*mean_time*(usage/100)
    print(f"Task     : {description}")
    print(f"CPU Time : {mean_time:.6f} s")
    print(f"Usage    : {usage:.2f} %")     
    print(f"Estimated Required Processing Capacity: {required_capacity:.0f}")
    print("#########################################################")


sample = X_test[0].reshape(1,-1)
predict = clf.predict
measure_performance(lambda: predict(sample), "Number Prediction")


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

mobileNet = MobileNetV2(weights=None)

image = np.random.rand(1,224,224,3).astype(np.float32)
image = preprocess_input(image)
predict = mobileNet.predict
measure_performance(lambda: predict(image), "Image Classification", N=10)





