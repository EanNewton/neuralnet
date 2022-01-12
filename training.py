import random


def model(rectangle, hidden_layer):
    output_neuron = 0.

    for index, input_neuron in enumerate(rectangle):
        output_neuron += input_neuron * hidden_layer[index]

    return output_neuron


def train(rectangles, rectangle_average, hidden_layer):
    outputs = []

    for rectangle in rectangles:
        output = model(rectangle, hidden_layer)
        outputs.append(output)

    error = mean_squared_error(outputs, rectangle_average)
    error.backward()

    for index, _ in enumerate(hidden_layer):
        learning_rate = 0.1
        hidden_layer.data[index] -= learning_rate * hidden_layer.grad.data[index]

    hidden_layer.grad.zero_()
    return error


def generate():
    rectangles = []
    rectangle_average = []

    for i in range(0, 1000):
        # Generate a 2x2 rectangle [0.1, 0.8, 0.6, 1.0]
        rectangle = [round(random.random(), 1),
                     round(random.random(), 1),
                     round(random.random(), 1),
                     round(random.random(), 1)]
        rectangles.append(rectangle)
        # Take the _actual_ average for our training dataset!
        rectangle_average.append(sum(rectangle) / 4)
    return (rectangles, rectangle_average)


# Take the average of all the differences squared!
# This calculates how "wrong" our predictions are.
# This is called our "loss".
def mean_squared_error(actual, expected):
    error_sum = 0
    for a, b in zip(actual, expected):
        error_sum += (a - b) ** 2
    return error_sum / len(actual)