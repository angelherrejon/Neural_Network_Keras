
# README

## Description
This project demonstrates the implementation of a deep learning model using Keras to classify the MNIST dataset, which contains handwritten digits. The model uses a fully connected (dense) neural network with two hidden layers, and it is trained using the categorical cross-entropy loss function with RMSprop optimizer.

### Key Features:
- Loads the MNIST dataset and normalizes the input images.
- Constructs a neural network with a hidden layer using ReLU activation and an output layer using softmax activation for classification.
- Trains the model for 5 epochs and evaluates it on the test set.

## Dependencies
The following Python libraries are required:
- `numpy`: For numerical operations.
- `keras`: For building and training the neural network.
- `matplotlib`: For visualizing the dataset.

Install the required libraries with:
```bash
pip install numpy keras matplotlib
```

## Code Structure

1. **train_mnist_model()**:
   - Loads and normalizes the MNIST dataset.
   - Defines a neural network architecture with an input layer, one hidden dense layer, and an output layer.
   - Compiles the model with RMSprop optimizer and categorical cross-entropy loss function.
   - Trains the model on the MNIST training data for 5 epochs.
   - Returns the trained model and the test data for evaluation.

2. **Model Architecture**:
   - Input layer with shape `(28*28)` representing flattened MNIST images.
   - Hidden dense layer with 512 units and ReLU activation.
   - Output dense layer with 10 units (for 10 digit classes) and softmax activation.

## Usage

1. **Training the Model**:
   To train the model on the MNIST dataset, call the function `train_mnist_model()`:
   ```python
   model, x_test, y_test = train_mnist_model()
   ```

2. **Evaluating the Model**:
   After training, you can evaluate the model's performance on the test set:
   ```python
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print(f'Test accuracy: {test_acc}')
   ```

## Customization

- **Network Structure**: You can modify the number of hidden layers and units in each layer by editing the `Sequential()` model in the code.
- **Training Parameters**: Change the `epochs`, `batch_size`, or optimizer in the `model.compile()` and `model.fit()` functions to experiment with different training configurations.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as needed.
