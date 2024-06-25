import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

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

# Carregar o dataset
dataset_path = 'music-notes-dataset/datasets/datasets/Notes'  # Ajuste o caminho para o seu dataset
images, labels, classes = load_dataset(dataset_path)

# Pré-processar os dados
images = images.reshape(-1, 64, 64, 1)
images = images / 255.0  # Corrigido aqui
labels = to_categorical(labels)

# Dividir o dataset em treino e validação
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Construir o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Salvar o modelo treinado no formato nativo Keras
model.save('music_notes_cnn.keras')

# Plotar precisão e perda
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
