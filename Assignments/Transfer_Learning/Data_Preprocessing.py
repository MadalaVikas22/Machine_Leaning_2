# Importing the necessary Libraries and Packages
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define constants and paths
IMAGE_SIZE = (255, 255)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = '/ML_2/RS_data/data'

# Create an empty dataframe to store image paths and labels
data = pd.DataFrame(columns=['image_path', 'label'])

# Define the labels/classes
labels = {'/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/RS_data/data/cloudy': 'Cloudy',
          '/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/RS_data/data/desert': 'Desert',
          '/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/RS_data/data/green_area': 'Green_Area',
          '/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/RS_data/data/water': 'Water'}

# Populate the dataframe with image paths and labels
for folder in labels:
    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        label = labels[folder]
        data = pd.concat([data, pd.DataFrame({'image_path': [image_path], 'label': [label]})], ignore_index=True)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Pre-process the data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=45,
                                   vertical_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    x_col="image_path",
                                                    y_col="label",
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical")

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                  x_col="image_path",
                                                  y_col="label",
                                                  target_size=IMAGE_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode="categorical")

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(255, 255, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Make predictions
predictions = model.predict(test_generator)
##########