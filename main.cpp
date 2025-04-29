#include "activations.h"
#include "initializers.h"
#include "neuralNetwork.h"
#include <iostream>

int main(int argc, char *argv[]) {
  NeuralNetwork nn(2, 0.001f);
  nn.addLayer(16, relu, d_relu, he);
  nn.addLayer(16, relu, d_relu, he);
  nn.addLayer(1, none, d_none, randomize);

  const std::vector<std::vector<std::vector<float>>> samples = {
      {{0, 0}, {0}},
      {{0, 1}, {1}},
      {{1, 0}, {1}},
      {{1, 1}, {0}},
  };

  nn.train(samples, 10000);

  nn.test();

  std::cout << std::endl << std::endl << "XOR:" << std::endl;
  nn.setInputs({0, 0});
  nn.predict().print();
  nn.setInputs({0, 1});
  nn.predict().print();
  nn.setInputs({1, 0});
  nn.predict().print();
  nn.setInputs({1, 1});
  nn.predict().print();

  return 0;
}
