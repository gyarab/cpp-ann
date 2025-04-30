#include "neuralNetwork.h"
#include "initializers.h"
#include "layer.h"
#include "random.h"
#include <algorithm>
#include <bits/stdc++.h>
#include <functional>
#include <iostream>
#include <vector>

NeuralNetwork::NeuralNetwork(int input_size)
    : input_size(input_size),
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
    const std::vector<std::vector<float>> training_inputs,
    const std::vector<std::vector<float>> training_targets, int batch_size,
    int epochs, float learning_rate) {
  this->learning_rate = learning_rate;

  std::vector<std::vector<std::vector<float>>> samples;
  for (int i = 0; i < training_inputs.size(); i++) {
    std::vector<std::vector<float>> sample;
    sample.push_back(training_inputs.at(i));
    sample.push_back(training_targets.at(i));
    samples.push_back(sample);
  }

  for (int epoch = 0; epoch < epochs; epoch++) {
    std::shuffle(samples.begin(), samples.end(), global_rng());

    for (int batch_start_index = 0; batch_start_index < samples.size();
         batch_start_index += batch_size) {

      std::vector<std::vector<std::vector<float>>> batch_samples(
          samples.begin() + batch_start_index,
          samples.begin() +
              std::min(batch_start_index + batch_size, (int)samples.size()));

      Matrix batch_inputs = Matrix(input_size, 1);
      Matrix batch_targets =
          Matrix(layers.at(layers.size() - 1).output_size, 1);
      std::vector<Matrix> batch_outputs;

      for (int i = 0; i < layers.size(); i++) {
        batch_outputs.push_back(Matrix(layers.at(i).output_size, 1));
      }

      for (int sample = 0; sample < batch_samples.size(); sample++) {
        setInputs(batch_samples.at(sample).at(0));
        batch_inputs = Matrix::add(batch_inputs, inputs);
        Matrix targetMatrix =
            Matrix::fromVector(batch_samples.at(sample).at(1));
        batch_targets = Matrix::add(batch_targets, targetMatrix);
        predict();
        for (int i = 0; i < layers.size(); i++) {
          batch_outputs.at(i) =
              Matrix::add(batch_outputs.at(i), layers.at(i).a);
        }
      }

      for (int i = 0; i < layers.size(); i++) {
        batch_outputs.at(i).applyFunction(
            [&batch_samples](float n) { return n / batch_samples.size(); });
      }
      batch_inputs.applyFunction(
          [&batch_samples](float n) { return n / batch_samples.size(); });
      batch_targets.applyFunction(
          [&batch_samples](float n) { return n / batch_samples.size(); });

      backpropagate(batch_inputs, batch_targets, batch_outputs);
    }
  }
}

void NeuralNetwork::backpropagate(Matrix &inputs, Matrix &targets,
                                  std::vector<Matrix> &outputs) {
  Matrix loss = Matrix::subtract(outputs.at(layers.size() - 1), targets);
  loss.applyFunction([](float n) { return n * n; });

  Matrix dE_dA(0, 0);
  Matrix next_layer_dE_dZ(0, 0);
  Matrix next_layer_delta_w(0, 0);
  Matrix next_layer_delta_b(0, 0);

  for (int i = layers.size() - 1; i >= 0; i--) {

    if (i == layers.size() - 1) {
      dE_dA = Matrix::subtract(outputs.at(i), targets);
      dE_dA.applyFunction([](float n) { return 2 * n; });
    } else {
      dE_dA = Matrix::multiply(Matrix::transpose(layers.at(i + 1).weights),
                               next_layer_dE_dZ);
    }

    Matrix dA_dZ = layers.at(i).d_activate();
    Matrix dE_dZ = Matrix::multiplyElementwise(dE_dA, dA_dZ);

    Matrix dZ_dW = i == 0 ? Matrix::transpose(inputs)
                          : Matrix::transpose(outputs.at(i - 1));
    Matrix dE_dW = Matrix::multiply(dE_dZ, dZ_dW);

    Matrix delta_w = dE_dW;
    delta_w.applyFunction([this](float n) { return n * -learning_rate; });

    Matrix delta_b = dE_dZ;
    delta_b.applyFunction([this](float n) { return n * -learning_rate; });

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
