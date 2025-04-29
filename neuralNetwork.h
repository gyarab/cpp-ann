#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <vector>

class NeuralNetwork {
private:
  int input_size;
  float learning_rate;
  Matrix inputs;

  std::vector<Layer> layers;
  void backpropagate(Matrix &targets);

public:
  NeuralNetwork(int input_size, float learning_rate = 0.001f);

  void setInputs(const std::vector<float> &input_vector);
  void addLayer(int size, std::function<Matrix(Matrix &)> activation,
                std::function<Matrix(Matrix &)> activation_derivative,
                std::function<void(Matrix &, int)> initializer);

  Matrix predict();
  void train(const std::vector<std::vector<std::vector<float>>> samples,
             int epochs);

  void setLearningRate(float lr);
  void test();
};

#endif
