# Intelligent Decision Support Systems Course Labs

This repository contains the laboratory works for the **Intelligent Decision Support Systems** course. The labs focus on practical skills in machine learning and neural networks using Python, including gradient descent convergence, multilayer perceptrons, regression models, convolutional neural networks, and transfer learning. All implementations are based on libraries such as NumPy, TensorFlow, Keras, and Matplotlib.

## Prerequisites

- Python 3.x
- Required libraries: `numpy`, `tensorflow`, `keras`, `matplotlib`, `scikit-learn`, `pandas` (install via `pip install -r requirements.txt` if a requirements file is provided)

## Table of Contents

- [Lab 1: Research on the Convergence of Gradient Learning Algorithms](#lab-1-research-on-the-convergence-of-gradient-learning-algorithms)
- [Lab 2: Implementation of Multilayer Perceptron for MNIST Image Classification in Python from Scratch Using NumPy](#lab-2-implementation-of-multilayer-perceptron-for-mnist-image-classification-in-python-from-scratch-using-numpy)
- [Lab 3: Training a Regression Model with Gradient Methods in TensorFlow Python](#lab-3-training-a-regression-model-with-gradient-methods-in-tensorflow-python)
- [Lab 4: Building Convolutional Neural Network Models for Image Classification in TensorFlow](#lab-4-building-convolutional-neural-network-models-for-image-classification-in-tensorflow)
- [Lab 5: Using Transfer Learning Technology for Building Convolutional Neural Network Models in Color Image Classification Tasks](#lab-5-using-transfer-learning-technology-for-building-convolutional-neural-network-models-in-color-image-classification-tasks)

## Lab 1: Research on the Convergence of Gradient Learning Algorithms

### Objective
To investigate the convergence properties of gradient descent algorithms.

### Task Description

<img width="902" height="622" alt="image" src="https://github.com/user-attachments/assets/bf03a9a7-390c-4b29-b750-ca12b642acf7" />

### Implementation Notes
Implement the algorithm in Python using NumPy for matrix operations and Matplotlib for plotting. Focus on analyzing convergence by varying \(\eta\).

## Lab 2: Implementation of Multilayer Perceptron for MNIST Image Classification in Python from Scratch Using NumPy

### Objective
To create a feedforward multilayer neural network using the NumPy library in Python, implement a custom class inheriting from object, realize the backpropagation algorithm to adjust the network weights, and use the multilayer feedforward neural network for classifying MNIST-type images.

### Task Description
(No use of Keras API; everything from scratch with NumPy.)

### Procedure
1. Load a simple set of images according to the variant.
2. Prepare the data for training as necessary, depending on the input set (e.g., normalize the data).
3. If necessary, split the data into training and validation sets in an 80%:20% ratio. Note that the loaded data may already be split into training and validation sets.
4. Build a basic feedforward network with a single hidden layer and a softmax output layer. Implement a custom class inheriting from object for the model. Initialize weights with small random values normally distributed with zero mean and small variance.
5. Implement the backpropagation algorithm for error propagation.
6. Set the loss function as cross-entropy.
7. Train the network weights using the implemented backpropagation algorithm.
8. Build several alternative MLP networks with multiple hidden layers using ReLU or tanh activations. Train their weights using the backpropagation algorithm.
9. For each model, build:
   - Graphs of the loss function changes on the training and validation sets as training progresses (depending on the number of epochs).
   - Graphs of the accuracy changes on the training and validation sets as training progresses.
10. Tune the hyperparameter for learning rate.
11. Use different activation functions for hidden layer neurons: LeakyReLU, Parametric LeakyReLU, ELU. Investigate if the choice of activation function affects:
    - Network training time.
    - Network performance quality (final accuracy value) on the validation set.
    - Prediction time by the network.
12. Display several images incorrectly classified by the model.

### Dataset (Variant 2)
Fashion MNIST from Kaggle.com

## Lab 3: Training a Regression Model with Gradient Methods in TensorFlow Python

### Objective
To learn the basics of the Keras API (tf.keras), use tf.data – Dataset API and function decorators, and create a custom class using tf.Module or tf.keras.Model.

### Procedure
1. Load the initial data.
2. Visualize the data graphically.
3. Initialize parameters: learning rate, number of epochs (=100), and additionally, if specified by the variant, batch size and/or regularization parameter.
4. Initialize the parameter vector.
5. Implement a function that defines the model, for example:
   - Polynomial regression.
6. Define the loss function (according to the variant):
   - MSE with L2 norm regularization.
7. Define the operation to be called at each training iteration:
   - Mini-batch gradient descent algorithm.
8. Train the model.
9. Output the loss function values every 10 epochs.
10. Save checkpoints at regular intervals during training. At the end of training, save the resulting model. Restore the last checkpoint if training was interrupted.
11. Tune the learning rate hyperparameter and, if specified by the variant, the mini-batch size. Examine the shape of the learning curve.
12. Investigate different regularization parameter values and select the best one if the loss function is regularized according to the variant. In this case, the data should be pre-split into training and validation sets.
13. Plot the graph with initial data and the regression line.

### Initial Data
- Dataset (a):
  ```python
  import numpy as np
  X_data = np.linspace(-1, 1, 100)
  num_coef = 2
  coef = [10, 3]
  y_data = 0
  for i in range(num_coef):
      y_data += coef[i] * np.power(X_data, i)
  y_data += np.random.randn(*X_data.shape) * 1.5
  ```
- Dataset (b): `sklearn.datasets.load_diabetes`

## Lab 4: Building Convolutional Neural Network Models for Image Classification in TensorFlow

### Objective
To learn to create convolutional neural network models using the TensorFlow library and Keras API, use a convolutional neural network for classifying black-and-white and color images, compare results with MLP-based models, and use TensorBoard for result visualization.

### Procedure
1. Load two sets of images according to the variant: black-and-white (from the previous lab) and color. If the set is large, select a subset.
2. Prepare the data for training as necessary.
3. Split the data into training, validation, and test subsets.
4. Build and train a basic model with one convolutional layer. Evaluate accuracy and precision on the training and validation sets.
5. Investigate different values of padding and strides parameters in the convolutional layer of the basic model, as well as kernel size, and their impact on model accuracy. Select values on the validation set.
6. Investigate several alternative convolutional model architectures that include:
   - Multiple convolutional layers.
   - Mini-batch normalization layer(s).
   - Dropout layer(s).
   Evaluate model quality on the validation set and select the best architecture using metrics: accuracy, precision, recall, f1-score, AUC.
7. Does adding regularization (dropout, early stopping, different weight initialization methods) affect model accuracy?
8. Display in TensorBoard graphs illustrating training quality assessments on training and validation sets:
   - Graphs of loss function changes on training and validation sets during model training.
   - Graphs of model accuracy changes on training and validation sets during model training.
9. Calculate quality assessments of the selected best model on the test set.
10. Load a test set image and recognize it with the trained models.
11. Compare the built convolutional models and the multilayer perceptron in black-and-white and color image classification tasks.
12. Draw conclusions regarding classification quality based on the built models.

### Dataset (Variant 2)
Rice Image Dataset from Kaggle.com

## Lab 5: Using Transfer Learning Technology for Building Convolutional Neural Network Models in Color Image Classification Tasks

### Objective
To learn to use transfer learning for building convolutional neural network models for color image classification, create convolutional neural network models using the TensorFlow library and Keras API, use a convolutional neural network for classifying color images, and use TensorBoard for result visualization.

### Procedure
1. Load the set of color images from the previous lab according to the variant. If the set is large, select a subset.
2. Prepare the data for training. Perform data augmentation.
3. Split the data into training, validation, and test subsets.
4. Build image classification models based on pre-trained deep convolutional networks using transfer learning:
   - Load pre-trained weights. Import weights obtained during training of selected deep models on the ImageNet dataset.
   - Build one or more top fully connected layers. The last (output) layer should be a fully connected softmax layer with the number of neurons equal to the number of classes in the dataset specified by the variant.
   - Freeze the pre-trained weights. By freezing the variables of the previous model, ensure that only the added top fully connected layer(s) are trained; the previous model values remain unchanged.
   - Perform fine-tuning of the added top layers on your own image set.
   - Tune the parameters of the added top layers on the validation subset.
   Investigate several pre-trained deep models, for example: VGG19, Xception, InceptionV3, ResNet152, DenseNet201, EfficientNetB7.
5. Display in TensorBoard graphs illustrating model training quality assessments:
   - Graphs of loss function changes on training and validation sets during model training.
   - Graphs of model quality metrics changes (accuracy, f1-score, AUC) on training and validation sets during model training.
6. Calculate quality assessments of the selected best model on the test set.
7. Load a test set image and recognize it with the trained models.
8. Draw conclusions regarding classification quality based on the built models. Compare with results from the previous lab.

### Dataset (Variant 2)
Rice Image Dataset from Kaggle.com
