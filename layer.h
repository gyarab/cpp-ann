#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include <functional>

struct Layer {
  int input_size;
  int output_size;

  Matrix weights;
  Matrix biases;

  Matrix z;
  Matrix a;

  std::function<Matrix(Matrix &)> activation;
  std::function<Matrix(Matrix &)> activation_derivative;
  std::function<void(Matrix &, int)> initializer;

  Layer(int input_size, int output_size,
        std::function<Matrix(Matrix &)> activation,
        std::function<Matrix(Matrix &)> activation_derivative,
        std::function<void(Matrix &, int)> initializer);

  void calculate(const Matrix &input);
  void activate();
  Matrix d_activate();

  void print();
};

#endif
