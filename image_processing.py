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


from contextlib import contextmanager
from timeit import default_timer
import psutil
import time
import os

process = psutil.Process(os.getpid()) 

N = 10000

@contextmanager
def elapsed_timer():
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    yield
    wall = time.perf_counter() - start_wall
    cpu = time.process_time() - start_cpu
    print(f"CPU time : {cpu/N:.6f} s")
    print(f"Usage    : {100 * cpu/wall:.2f} %")

sample = X_test[0].reshape(1,-1)
with elapsed_timer():
    for _ in range(N):
        predicted = clf.predict(sample) 



