
import numpy as np

# creates weight_matrix which contains weights of each layer
def create_weights_biases(layers, input, seed):

    # weight_matrix init
    weight_matrix = []
    # weight_matrix = np.empty(len(layers), dtype=object)

    # weights are created based on the input to the layer and the number of nodes in the layer
    # so capturing node count and caputreing layer_index to predict the incoming inputs dimensions
    for layer_index, node_count in enumerate(layers):

        # layer 1 is represented as 0 here for comfortable coding
        # cannot predict weights dimensions of layer 1 as it doesnt have a previous layer_index
        # therefore layer 1 weights completely depend on input dataset
        if layer_index == 0:

            # layer 1 weights dimensions (node_count, number of rows in input +PLUS bias)
            rows = node_count
            input_rows, input_columns = np.shape(input)
            columns = input_rows
            np.random.seed(seed)

            # random weight generation based on random seed provided
            weight_matrix.append(np.random.randn(rows, columns+1))
            # weight_matrix[layer_index] = np.random.randn(rows, columns+1)

        #  for every other layer dimensions o weight matrix can be predicted based on the prev layer nodes
        # and current layer nodes
        else:
            np.random.seed(seed)

            # random weight generation based on random seed provided
            weight_matrix.append(np.random.randn(
                node_count, layers[layer_index-1]+1))
            # weight_matrix[layer_index] = np.random.randn(rows, layers[layer_index-1]+1)

    # returns randomly initialized weights for each layer
    return weight_matrix


def sigmoid(x):
    return 1/(1+np.exp(-x))

# Predicts target values for given input


def predict(weight_matrix, input, layers):

    # input values travel through all the count of number of layers
    for layer_index in range(len(layers)):
        # if i==0:
        #     # X_train=np.transpose(X_train)
        #     X_train = np.concatenate((np.ones((1,X_train[0].size)),X_train))
        #     # print(np.shape(weight_matrix[i]),np.shape(X_train))
        #     net_matrix = np.dot(weight_matrix[i],X_train)
        #     net_matrix= sigmoid(net_matrix)
        #     # print(np.shape(net_matrix))
        #     X_train=net_matrix
        # else:
        # print(np.shape(weight_matrix[i]),np.shape(X_train),weight_matrix[i])

        # we have to add 1's in the first row of the input to multiply with bias
        input = np.concatenate((np.ones((1, input[0].size)), input))

        # weighted sum is calculated for each layer
        # and sigmoid function is applied on the weighted sum
        weighted_sum = np.dot(weight_matrix[layer_index], input)
        net_matrix = sigmoid(weighted_sum)

        # the resultent net_matrix is used as input for the next layer
        input = net_matrix

    # once the input is traveled through all layers the final form is the output that is returned
    return input

# makes a deep copy of weight matrix


def deep_copy(weight_matrix, weight_matrix_copy):
    for layer_index in range(len(weight_matrix)):
        for node_index, node in enumerate(weight_matrix[layer_index]):
            for weight_index, weight in enumerate(node):
                # print(id(weight_matrix),id(weight_matrix_copy))
                current_weight = weight_matrix[layer_index][node_index][weight_index]
                weight_matrix_copy[layer_index][node_index][weight_index] = current_weight
    return weight_matrix_copy

# trains the NN by updating the weights inorder to reduse MSE


def train(weight_matrix, weight_matrix_copy, X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h, seed):

    # tragets each weight uniquely
    for layer_index in range(len(weight_matrix_copy)):
        for node_index, node in enumerate(weight_matrix_copy[layer_index]):
            for weight_index, weight in enumerate(node):
                print(id(weight_matrix), id(weight_matrix_copy))

                current_weight = weight_matrix_copy[layer_index][node_index][weight_index]
                original_weight = weight
                # x+h
                gweight = weight+h
                # x-h
                lweight = original_weight-h

                # X+h insertion into weight_matrix
                weight_matrix_copy[layer_index][node_index][weight_index] = gweight

                # prediction with weight+h
                gprediction = predict(weight_matrix_copy, X_train, layers)

                # X-h insertion into weight_matrix
                weight_matrix_copy[layer_index][node_index][weight_index] = lweight

                # prediction with weight-h
                lprediction = predict(weight_matrix_copy, X_train, layers)

                # centered difference approximation to calculate partial derivatives.
                # (f(x + h)-f(x - h))/2*h
                partial_derivative = ((np.square(
                    gprediction - Y_train)).mean()-(np.square(lprediction - Y_train)).mean())/(2*h)

                # update weights
                new_weight = current_weight-(alpha*partial_derivative)

                # make weight_matrix_copy to its original form and traget next values uniquely
                weight_matrix_copy[layer_index][node_index][weight_index] = original_weight

                # update original weight matrix
                weight_matrix[layer_index][node_index][weight_index] = new_weight
                print(id(weight_matrix), id(weight_matrix_copy))

    # final weight_matrix after updateing all weights uniquely
    return weight_matrix

# Multi_layer_NN istarting point i.e main function


def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h, seed):

    # initialize random weights and biases for each layer
    weights = create_weights_biases(layers, X_train, seed)
    weight_matrix_copy = create_weights_biases(layers, X_train, seed)

    # array to store Mean-squared-error
    mse = []

    # NN gets trained epochs number of times by calling train function
    # for i in range(epochs):
    for i in range(epochs):

        # train function returns updated weights
        # the updated weights are used for next training
        weights = train(weights, weight_matrix_copy, X_train, Y_train, X_test,
                        Y_test, layers, alpha, epochs, h, seed)
        weight_matrix_copy = deep_copy(weights, weight_matrix_copy)

        # predit function predicts Y_test for X_test and stored in predictions
        predictions = predict(weights, X_test, layers)

        # Mean Squared Error is claculated between prediction and Y_test in each epoch and stored in MSE array
        mse.append((np.square(predictions - Y_test)).mean())

    # prediction is made on X_test data after training the NN for epoch number of times
    predictions = predict(weights, X_test, layers)

    # multi_layer_nn returns updated weights after epoch number of times training,
    # MSE after each update of error, final prediction of epoch times training
    return [weights, mse, predictions]
