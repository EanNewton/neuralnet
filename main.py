#!/usr/bin/python3

"""
Super Simple Neural Net
Implements an average() function on a 2x2 rectangle
Based on: https://sirupsen.com/napkin/neural-net
"""

import torch

import training


if __name__ == '__main__':
    rectangles, rectangle_average = training.generate()
    hidden_layer = torch.tensor([0.98, 0.4, 0.86, -0.08], requires_grad=True)

    for epoch in range(int(input("Epochs > "))):
        error = training.train(rectangles, rectangle_average, hidden_layer)
        print(f"Epoch: {epoch}, Error: {error}, Layer: {hidden_layer.data}")

    print(f"After: {training.model([0.2, 0.5, 0.4, 0.7], hidden_layer)}")