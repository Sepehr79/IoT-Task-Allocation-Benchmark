import psutil
import time

freq = psutil.cpu_freq()
print("Frequency Status")
print(f"Current: {freq.current:.2f} MHz")
print("#########################################################")


def measure_performance(function, description, N = 10000):
    start_wall = time.process_time()
    for _ in range(N):
        function()
    elapsed = time.process_time() - start_wall
    mean_time = elapsed/N
    required_capacity = freq.current*1e6*mean_time
    print(f"Task: {description}")
    print(f"Elapsed Time: {mean_time:.6f} s")
    print(f"Estimated Required Processing Capacity: {required_capacity:,.0f}")
    print("#########################################################")



from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    EfficientNetB0,
    DenseNet121,
    VGG16,
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import random

folder_path = "dataset"
valid_extensions = (".jpg", ".jpeg", ".png")

image_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith(valid_extensions)
]

random_image = random.choice(image_files)
img = image.load_img(random_image, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

mobileNet = MobileNetV2(weights=None)
resnet = ResNet50(weights=None)
efficientNet = EfficientNetB0(weights=None)
denseNet = DenseNet121(weights=None)
vgg16 = VGG16(weights=None)

mobileNet.predict(x, verbose=0)
measure_performance(lambda: mobileNet.predict(x, verbose=0), "MobileNet Image Classification", N=10)

resnet.predict(x, verbose=0)
measure_performance(lambda: resnet.predict(x, verbose=0), "ResNet50 Image Classification", N=10)

efficientNet.predict(x, verbose=0)
measure_performance(lambda: efficientNet.predict(x, verbose=0), "EfficientNetB0 Image Classification", N=10)

denseNet.predict(x, verbose=0)
measure_performance(lambda: denseNet.predict(x, verbose=0), "DenseNet121 Image Classification", N=10)

vgg16.predict(x, verbose=0)
measure_performance(lambda: vgg16.predict(x, verbose=0), "VGG16 Image Classification", N=10)


file_sizes = np.array([
    os.path.getsize(img_file) / 1024
    for img_file in image_files
])

print(f"Minimum: {np.min(file_sizes):.2f} KB")
print(f"Maximum: {np.max(file_sizes):.2f} KB")
print(f"Mean:    {np.mean(file_sizes):.2f} KB")
print(f"Median:  {np.median(file_sizes):.2f} KB")
print(f"Std:     {np.std(file_sizes):.2f} KB")