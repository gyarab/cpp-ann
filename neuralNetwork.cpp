#include "neuralNetwork.h"
#include "initializers.h"
#include "layer.h"
#include "random.h"
#include <algorithm>
#include <bits/stdc++.h>
#include <functional>
#include <iostream>
#include <vector>

NeuralNetwork::NeuralNetwork(int input_size, float learning_rate)
    : input_size(input_size), learning_rate(learning_rate),
      inputs(Matrix::fromVector(std::vector<float>(input_size))) {}

void NeuralNetwork::setInputs(const std::vector<float> &input_vector) {
  inputs = Matrix::fromVector(input_vector);
}

void NeuralNetwork::addLayer(
    int size, std::function<Matrix(Matrix &)> activation,
    std::function<Matrix(Matrix &)> activation_derivative,
    std::function<void(Matrix &, int)> initializer) {
  if (layers.size() < 1)
    layers.push_back(Layer(input_size, size, activation, activation_derivative,
                           initializer));
  else
    layers.push_back(Layer(layers.at(layers.size() - 1).output_size, size,
                           activation, activation_derivative, initializer));
}

void NeuralNetwork::test() {
  std::cout << "Inputs:" << std::endl << "--------------------" << std::endl;
  inputs.print();

  if (layers.size() > 0) {
    std::cout << std::endl << std::endl;
    for (int i = 0; i < layers.size(); i++) {
      std::cout << "Layer " << i + 1 << ':' << std::endl
                << "--------------------" << std::endl;
      layers[i].print();
      if (i != layers.size() - 1)
        std::cout << std::endl << std::endl;
    }
  }
}

Matrix NeuralNetwork::predict() {
  for (int i = 0; i < layers.size(); i++) {
    if (i == 0) {
      layers.at(0).calculate(inputs);
      layers.at(0).activate();
    } else {
      layers.at(i).calculate(layers.at(i - 1).a);
      layers.at(i).activate();
    }
  }

  return layers.at(layers.size() - 1).a;
}

void NeuralNetwork::train(
    const std::vector<std::vector<std::vector<float>>> samples, int epochs) {
  auto shuffled_samples = samples;
  for (int i = 0; i < epochs; i++) {
    std::shuffle(shuffled_samples.begin(), shuffled_samples.end(), global_rng());

    for (auto sample : shuffled_samples) {
      setInputs(sample.at(0));
      predict();
      Matrix targetMatrix = Matrix::fromVector(sample.at(1));
      backpropagate(targetMatrix);
    }
  }
}

void NeuralNetwork::backpropagate(Matrix &targets) {
  Matrix loss = Matrix::subtract(layers.at(layers.size() - 1).a, targets);

  // std::cout << std::endl << "o - t:" << std::endl;
  // loss.print();

  loss.applyFunction([](float n) { return n * n; });

  Matrix dE_dA(0, 0);
  Matrix next_layer_dE_dZ(0, 0);
  Matrix next_layer_delta_w(0, 0);
  Matrix next_layer_delta_b(0, 0);

  for (int i = layers.size() - 1; i >= 0; i--) {

    if (i == layers.size() - 1) {
      dE_dA = Matrix::subtract(layers.at(i).a, targets);
      dE_dA.applyFunction([](float n) { return 2 * n; });
    } else {
      dE_dA = Matrix::multiply(Matrix::transpose(layers.at(i + 1).weights),
                               next_layer_dE_dZ);
    }

    // std::cout << std::endl << "dE_dA:" << std::endl;
    // dE_dA.print();

    Matrix dA_dZ = layers.at(i).d_activate();

    // std::cout << std::endl << "dA_dZ:" << std::endl;
    // dA_dZ.print();

    Matrix dE_dZ = Matrix::multiplyElementwise(dE_dA, dA_dZ);

    // std::cout << std::endl << "dE_dZ:" << std::endl;
    // dE_dZ.print();

    Matrix dZ_dW = i == 0 ? Matrix::transpose(inputs)
                          : Matrix::transpose(layers.at(i - 1).a);
    Matrix dE_dW = Matrix::multiply(dE_dZ, dZ_dW);

    // std::cout << std::endl << "dE_dW:" << std::endl;
    // dE_dW.print();

    Matrix delta_w = dE_dW;
    delta_w.applyFunction([this](float n) { return n * -learning_rate; });

    // std::cout << std::endl << "dW:" << std::endl;
    // delta_w.print();

    Matrix delta_b = dE_dZ;
    delta_b.applyFunction([this](float n) { return n * -learning_rate; });

    // std::cout << std::endl << "dB:" << std::endl;
    // delta_b.print();

    if (i + 1 < layers.size()) {
      layers.at(i + 1).weights =
          Matrix::add(layers.at(i + 1).weights, next_layer_delta_w);
      layers.at(i + 1).biases =
          Matrix::add(layers.at(i + 1).biases, next_layer_delta_b);
    }

    next_layer_dE_dZ = dE_dZ;
    next_layer_delta_b = delta_b;
    next_layer_delta_w = delta_w;
  }

  layers.at(0).weights = Matrix::add(layers.at(0).weights, next_layer_delta_w);
  layers.at(0).biases = Matrix::add(layers.at(0).biases, next_layer_delta_b);
}
