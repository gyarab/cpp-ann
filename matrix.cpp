#include "matrix.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) { allocate(); }

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols) {
  allocate();
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      data[i][j] = other.data[i][j];
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this != &other) {
    deallocate();
    rows = other.rows;
    cols = other.cols;
    allocate();
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        data[i][j] = other.data[i][j];
  }
  return *this;
}

Matrix::~Matrix() { deallocate(); }

void Matrix::allocate() {
  data = new float *[rows];
  for (int i = 0; i < rows; ++i)
    data[i] = new float[cols]();
}

void Matrix::deallocate() {
  for (int i = 0; i < rows; ++i)
    delete[] data[i];
  delete[] data;
}

// Matrix Matrix::fromArray(const float *array, int size) {
//   Matrix m(size, 1);
//   for (int i = 0; i < size; ++i)
//     m.data[i][0] = array[i];
//   return m;
// }

Matrix Matrix::fromVector(std::vector<float> vector) {
  Matrix m(vector.size(), 1);
  for (int i = 0; i < vector.size(); i++) {
    m.data[i][0] = vector.at(i);
  }
  return m;
}

int Matrix::getR() const { return rows; }
int Matrix::getC() const { return cols; }

float Matrix::get(int row, int col) const { return data[row][col]; }

void Matrix::set(int row, int col, float value) { data[row][col] = value; }

// void Matrix::randomize(float min, float max) {
//   for (int i = 0; i < rows; ++i)
//     for (int j = 0; j < cols; ++j)
//       data[i][j] = min + static_cast<float>(rand()) / RAND_MAX * (max - min);
// }

void Matrix::applyFunction(std::function<float(float)> func) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      data[i][j] = func(data[i][j]);
}

// void Matrix::multiply(float scalar) {
//   for (int i = 0; i < rows; ++i)
//     for (int j = 0; j < cols; ++j)
//       data[i][j] *= scalar;
// }

Matrix Matrix::multiply(const Matrix &a, const Matrix &b) {
  if (a.cols != b.rows) {
    std::cerr << "Matrix multiply error: incompatible dimensions." << std::endl;
    exit(1);
  }

  Matrix result(a.rows, b.cols);
  for (int i = 0; i < result.rows; ++i)
    for (int j = 0; j < result.cols; ++j)
      for (int k = 0; k < a.cols; ++k)
        result.data[i][j] += a.data[i][k] * b.data[k][j];
  return result;
}

Matrix Matrix::multiplyElementwise(const Matrix &a, const Matrix &b) {
  Matrix result = a;
  for (int row = 0; row < result.rows; row++) {
    for (int col = 0; col < result.cols; col++) {
      result.data[row][col] *= b.data[row][col];
    }
  }
  return result;
}

Matrix Matrix::transpose(const Matrix &m) {
  Matrix result(m.cols, m.rows);
  for (int i = 0; i < m.rows; ++i)
    for (int j = 0; j < m.cols; ++j)
      result.data[j][i] = m.data[i][j];
  return result;
}

Matrix Matrix::add(const Matrix &a, const Matrix &b) {
  if (a.rows != b.rows || a.cols != b.cols) {
    std::cerr << "Matrix add error: incompatible dimensions." << std::endl;
    exit(1);
  }

  Matrix result(a.rows, a.cols);
  for (int i = 0; i < a.rows; ++i)
    for (int j = 0; j < a.cols; ++j)
      result.data[i][j] = a.data[i][j] + b.data[i][j];
  return result;
}

Matrix Matrix::subtract(const Matrix &a, const Matrix &b) {
  if (a.rows != b.rows || a.cols != b.cols) {
    std::cerr << "Matrix subtract error: incompatible dimensions." << std::endl;
    exit(1);
  }

  Matrix result(a.rows, a.cols);
  for (int i = 0; i < a.rows; ++i)
    for (int j = 0; j < a.cols; ++j)
      result.data[i][j] = a.data[i][j] - b.data[i][j];
  return result;
}

void Matrix::print() const {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j)
      std::cout << std::fixed << (data[i][j] > 0 ? std::setprecision(5) : std::setprecision(4)) << data[i][j] << " ";
    std::cout << std::endl;
  }
}
