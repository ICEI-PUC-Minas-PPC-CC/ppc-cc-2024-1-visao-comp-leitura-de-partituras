import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Função para carregar e pré-processar o dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    classes = os.listdir(dataset_path)
    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, classes

# Função para carregar o modelo e as classes
def prepare_model(model_path, dataset_path):
    model = load_model(model_path)
    _, _, classes = load_dataset(dataset_path)
    return model, classes

model, classes = prepare_model('music_notes_cnn.keras', 'music-notes-dataset/datasets/datasets/Notes')

# Função para pré-processar imagens para classificação
def preprocess_image(image, target_size=(64, 64)):
    # Redimensionar a imagem para o tamanho esperado pelo modelo
    image = cv2.resize(image, target_size)
    # Converter a imagem para float32 e normalizar
    image = image.astype('float32') / 255.0
    # Expandir as dimensões da imagem para (1, altura, largura, canais)
    image = np.expand_dims(image, axis=0)
    return image

# Função para classificar uma imagem usando o modelo treinado
def classify_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class