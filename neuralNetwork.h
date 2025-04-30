#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <vector>

class NeuralNetwork {
private:
  int input_size;
  float learning_rate = 0.001f;
  Matrix inputs;

  std::vector<Layer> layers;
  void backpropagate(Matrix &inputs, Matrix &targets,
                     std::vector<Matrix> &outputs);

public:
  NeuralNetwork(int input_size);

  void setInputs(const std::vector<float> &input_vector);
  void addLayer(int size, std::function<Matrix(Matrix &)> activation,
                std::function<Matrix(Matrix &)> activation_derivative,
                std::function<void(Matrix &, int)> initializer);

  Matrix predict();

  void train(const std::vector<std::vector<float>> training_inputs,
             const std::vector<std::vector<float>> training_targets,
             int batch_size, int epochs, float learning_rate=0.001f);

  void setLearningRate(float lr);
  void test();
};

#endif
