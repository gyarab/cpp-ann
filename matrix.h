#ifndef MATRIX_H
#define MATRIX_H

#include <functional>

class Matrix {
private:
  int rows, cols;
  float **data;

  void allocate();
  void deallocate();

public:
  Matrix(int rows, int cols);
  Matrix(const Matrix &other);
  Matrix &operator=(const Matrix &other);
  ~Matrix();

  // static Matrix fromArray(const float *array, int size);
  static Matrix fromVector(std::vector<float> vector);
  int getR() const;
  int getC() const;
  float get(int row, int col) const;
  void set(int row, int col, float value);

  // void randomize(float min = -1.0f, float max = 1.0f);
  void applyFunction(std::function<float(float)> func);

  static Matrix multiply(const Matrix &a, const Matrix &b);
  static Matrix multiplyElementwise(const Matrix &a, const Matrix &b);

  static Matrix transpose(const Matrix &m);

  static Matrix add(const Matrix &a, const Matrix &b);
  static Matrix subtract(const Matrix &a, const Matrix &b);

  void print() const;
};

#endif
