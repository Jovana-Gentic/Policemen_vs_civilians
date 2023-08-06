import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score
from matplotlib import pyplot as plt
from data import create_dataset, get_positives_and_negatives
from utils import plot_model_probabilities

# Loading model
checkpoint_filepath = '../model_checkpoint/model11'
model = tf.keras.models.load_model(checkpoint_filepath)

#Creating dataset
_, val_data, _= create_dataset()
pos_inputs, neg_inputs, inputs, labels = get_positives_and_negatives(val_data)

plot_model_probabilities(model, pos_inputs, neg_inputs)
    
# Computing metrics
threshold = float(input("Enter model threshold:"))
accuracy = accuracy_score(labels, (model.predict(inputs)>threshold).astype(int))
recall = recall_score(labels, (model.predict(inputs)>threshold).astype(int))
precision = precision_score(labels, (model.predict(inputs)>threshold).astype(int))
print(f"Accuracy is {accuracy}")
print(f"Recall is {recall}")
print(f"Precision is {precision}")
