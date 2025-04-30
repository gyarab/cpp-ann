#include "matrix.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) { allocate(); }

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols) {
  allocate();
  for (int row = 0; row < rows; row++)
    for (int col = 0; col < cols; col++)
      data[row][col] = other.data[row][col];
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this != &other) {
    deallocate();
    rows = other.rows;
    cols = other.cols;
    allocate();
    for (int row = 0; row < rows; row++)
      for (int col = 0; col < cols; col++)
        data[row][col] = other.data[row][col];
  }
  return *this;
}

Matrix::~Matrix() { deallocate(); }

void Matrix::allocate() {
  data = new float *[rows];
  for (int row = 0; row < rows; row++)
    data[row] = new float[cols]();
}

void Matrix::deallocate() {
  for (int row = 0; row < rows; row++)
    delete[] data[row];
  delete[] data;
}

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

void Matrix::applyFunction(std::function<float(float)> func) {
  for (int row = 0; row < rows; row++)
    for (int col = 0; col < cols; col++)
      data[row][col] = func(data[row][col]);
}

Matrix Matrix::multiply(const Matrix &a, const Matrix &b) {
  if (a.cols != b.rows) {
    std::cerr << "Matrix multiply error: incompatible dimensions." << std::endl;
    exit(1);
  }

  Matrix result(a.rows, b.cols);
  for (int row = 0; row < result.rows; row++)
    for (int col = 0; col < result.cols; col++)
      for (int i = 0; i < a.cols; i++)
        result.data[row][col] += a.data[row][i] * b.data[i][col];
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
  for (int row = 0; row < m.rows; row++)
    for (int col = 0; col < m.cols; col++)
      result.data[col][row] = m.data[row][col];
  return result;
}

Matrix Matrix::add(const Matrix &a, const Matrix &b) {
  if (a.rows != b.rows || a.cols != b.cols) {
    std::cerr << "Matrix add error: incompatible dimensions." << std::endl;
    exit(1);
  }

  Matrix result(a.rows, a.cols);
  for (int row = 0; row < a.rows; row++)
    for (int col = 0; col < a.cols; col++)
      result.data[row][col] = a.data[row][col] + b.data[row][col];
  return result;
}

Matrix Matrix::subtract(const Matrix &a, const Matrix &b) {
  if (a.rows != b.rows || a.cols != b.cols) {
    std::cerr << "Matrix subtract error: incompatible dimensions." << std::endl;
    exit(1);
  }

  Matrix result(a.rows, a.cols);
  for (int row = 0; row < a.rows; row++)
    for (int col = 0; col < a.cols; col++)
      result.data[row][col] = a.data[row][col] - b.data[row][col];
  return result;
}

void Matrix::print() const {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++)
      std::cout << std::fixed
                << (data[row][col] > 0 ? std::setprecision(5)
                                       : std::setprecision(4))
                << data[row][col] << " ";
    std::cout << std::endl;
  }
}
