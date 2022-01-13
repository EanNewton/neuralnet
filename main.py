#!/usr/bin/python3

"""
Super Simple Neural Net
Implements an average() function on a 2x2 rectangle
Based on: https://sirupsen.com/napkin/neural-net
"""

# TODO Support far larger rectangles, eg 100x100

# TODO Add biases in addition to the weights. A model doesn't just have weights that are multiplied onto the inputs,
#  but also biases that are added (+) onto the inputs in each layer.

# TODO Rewrite the model to use PyTorch tensors for matrix operations, as described in the previous section.

# TODO Add 1-2 more layers to the model. Try to have them different sizes.

# TODO Change the tensors to run on GPU (see the PyTorch documentation) and see the performance speed up! Increase
#  the size of the training set and rectangles to really be able to tell the difference.
#  Make sure you change Runtime > Change Runtime Type in Collab to run on a GPU.

# TODO This is a difficult step that will likely take a while, but it'll be well worth it:
#  Adapt the code to recognize handwritten letters from the MNIST dataset dataset. You'll need to use pillow to turn
#  the pixels into a large 1-dimensional tensor as the input layer, as well as a non-linear activation function like
#  Sigmoid or ReLU. Use Nielsen's book as a reference if you get stuck, which does exactly this.

import torch

import training


if __name__ == '__main__':
    rectangles, rectangle_average = training.generate()
    hidden_layer = torch.tensor([0.98, 0.4, 0.86, -0.08], requires_grad=True)

    for epoch in range(int(input("Epochs > "))):
        error = training.train(rectangles, rectangle_average, hidden_layer)
        print(f"Epoch: {epoch}, Error: {error}, Layer: {hidden_layer.data}")

    print(f"After: {training.model([0.2, 0.5, 0.4, 0.7], hidden_layer)}")