import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

def train_mnist_model():
    # Cargar los datos MNIST
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Mostrar información de los datos
    print(train_data_x.shape)
    print(train_labels_y[1])
    plt.imshow(train_data_x[0])
    print(test_data_x.shape)
    plt.show()

    # Arquitectura de la red
    model = Sequential([
        Input(shape=(28*28,)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compilación
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Resumen de la red
    model.summary()

    # Normalización de los datos
    x_train = train_data_x.reshape(60000, 28*28)
    x_train = x_train.astype('float32') / 255
    y_train = to_categorical(train_labels_y)

    # Normalización de los datos de prueba
    x_test = test_data_x.reshape(10000, 28*28)
    x_test = x_test.astype('float32') / 255
    y_test = to_categorical(test_labels_y)

    # Entrenamiento
    model.fit(x_train, y_train, epochs=5, batch_size=128)

    return model, x_test, y_test  # Devolver el modelo y los datos de prueba para usarlos después

# Asegúrate de que el código solo se ejecuta si este archivo es el principal
if __name__ == "__main__":
    train_mnist_model()
