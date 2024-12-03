# # Importing the necessary Libraries and Packages
# import os
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import BernoulliRBM
# from keras.preprocessing.image import ImageDataGenerator
# from sklearn.pipeline import Pipeline
# from keras.utils import to_categorical
#
# # Define constants and paths
# IMAGE_SIZE = (255, 255)
# BATCH_SIZE = 32
# EPOCHS = 10
# DATA_DIR = '/ML_2/RS_data/data'
#
# # Create an empty dataframe to store image paths and labels
# data = pd.DataFrame(columns=['image_path', 'label'])
#
# # Define the labels/classes
# labels = {'/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/RS_data/data/cloudy': 'Cloudy',
#           '/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/RS_data/data/desert': 'Desert',
#           '/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/RS_data/data/green_area': 'Green_Area',
#           '/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/RS_data/data/water': 'Water'}
#
# # Populate the dataframe with image paths and labels
# for folder in labels:
#     for image_name in os.listdir(folder):
#         image_path = os.path.join(folder, image_name)
#         label = labels[folder]
#         data = pd.concat([data, pd.DataFrame({'image_path': [image_path], 'label': [label]})], ignore_index=True)
#
# # Split the dataset into training and testing sets
# train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
#
# # Pre-process the data using ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    rotation_range=45,
#                                    vertical_flip=True,
#                                    fill_mode='nearest')
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
#                                                     x_col="image_path",
#                                                     y_col="label",
#                                                     target_size=IMAGE_SIZE,
#                                                     batch_size=BATCH_SIZE,
#                                                     class_mode="categorical")
#
# test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
#                                                   x_col="image_path",
#                                                   y_col="label",
#                                                   target_size=IMAGE_SIZE,
#                                                   batch_size=BATCH_SIZE,
#                                                   class_mode="categorical")
#
# # Convert labels to categorical format
# train_y = to_categorical(train_generator.labels)
# test_y = to_categorical(test_generator.labels)
#
# # Define the model architecture
# rbm = BernoulliRBM(n_components=50, learning_rate=0.01, batch_size=BATCH_SIZE, n_iter=EPOCHS, verbose=1, random_state=42)
# model = Pipeline(steps=[('rbm', rbm)])
#
# # Fit the model
# model.fit(train_generator, train_y)
#
# # Evaluate the model
# loss = model.score_samples(test_generator)
# print(f"Negative log likelihood: {loss}")
#
# # Make predictions
# predictions = model.predict(test_generator)

#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# Load the dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)

# Preprocess the data by scaling it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the RBM model
rbm = BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=5, verbose=1)
# Initialize the logistic regression model
logistic = LogisticRegression(max_iter=1000)
# Create a pipeline that first extracts features using the RBM and then classifies with logistic regression
dbn_pipeline = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
# Train the DBN
dbn_pipeline.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
dbn_score = dbn_pipeline.score(X_test_scaled, y_test)
print(f"DBN Classification score: {dbn_score}")
