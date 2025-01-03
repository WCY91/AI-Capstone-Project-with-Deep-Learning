import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

data_generator = ImageDataGenerator()
dataset_dir = './concrete_data_week2'
positive = 'Positive'
positive_file_path = os.path.join(dataset_dir,positive)
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
print(len(positive_files))
image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
)

first_batch = next(image_generator)  # Use next(image_generator)
first_batch_images = first_batch[0]
first_batch_labels = first_batch[1]

data_generator = ImageDataGenerator(
    rescale=1./255
)

image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
)

first_batch = next(image_generator)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))  # Define figure and axes

ind = 0

second_batch = next(image_generator)
third_batch = next(image_generator)
third_batch_images = third_batch[0]
for ax1 in axs:
    for ax2 in ax1:
        image_data = (third_batch_images[ind] * 255).astype(np.uint8)  # Rescale back to 0-255 for visualization
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('First Batch of Concrete Images')
plt.show()

forth_batch = next(image_generator)
forth_count = Counter(np.argmax(forth_batch[1], axis=1))
print(forth_count)

fifth_batch = next(image_generator)
fifth_count = Counter(np.argmax(fifth_batch[1], axis=1))
print(fifth_count)
