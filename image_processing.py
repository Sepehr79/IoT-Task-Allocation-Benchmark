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

process = psutil.Process(os.getpid()) 

def measure_performance(function, description, N = 10000):
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    for _ in range(N):
        function()
    wall = time.perf_counter() - start_wall
    cpu = time.process_time() - start_cpu
    mean_time = cpu/N
    usage = cpu/wall
    required_capacity = 2*10**9*usage*mean_time
    print(f"Task     : {description}")
    print(f"CPU Time : {cpu/N:.6f} s")
    print(f"Usage    : {100 * usage:.2f} %")     
    print(f"Estimated Required Processing Capacity: {required_capacity:.0f}")
    print("#########################################################")


sample = X_test[0].reshape(1,-1)
predict = clf.predict
measure_performance(lambda: predict(sample), "Number Prediction")




