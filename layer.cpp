#include "layer.h"
#include <iostream>

Layer::Layer(int input_size, int output_size,
             std::function<Matrix(Matrix &)> activation,
             std::function<Matrix(Matrix &)> activation_derivative,
             std::function<void(Matrix &, int)> initializer)
    : input_size(input_size), output_size(output_size),
      weights(output_size, input_size), biases(output_size, 1),
      activation(activation), activation_derivative(activation_derivative),
      initializer(initializer), z(Matrix(output_size, 1)), a(output_size, 1) {
  initializer(weights, input_size);
}

void Layer::calculate(const Matrix &input) {
  z = Matrix::multiply(weights, input);
  z = Matrix::add(z, biases);
}

void Layer::activate() { a = activation(z); }
Matrix Layer::d_activate() { return activation_derivative(a); }

void Layer::print() {
  std::cout << "Output neurons: " << output_size << std::endl;
  std::cout << "Input neurons: " << input_size << std::endl;

  std::cout << std::endl << "Weighted sums:" << std::endl;
  z.print();

  std::cout << std::endl << "Output values:" << std::endl;
  a.print();

  std::cout << std::endl << "Weights:" << std::endl;
  weights.print();

  std::cout << std::endl << "Biases:" << std::endl;
  biases.print();
}
