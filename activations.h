#pragma once
#include "matrix.h"
#include <cmath>

inline Matrix relu(Matrix &z) {
  Matrix a(z);
  a.applyFunction([](float n) { return n <= 0 ? 0 : n; });
  return a;
};

inline Matrix d_relu(Matrix &a) {
  Matrix d(a);
  d.applyFunction([](float n) { return n <= 0 ? 0 : 1; });
  return d;
};

inline Matrix none(Matrix &z) { return z; }

inline Matrix d_none(Matrix &a) {
  Matrix d(a.getR(), a.getC());
  d.applyFunction([](float n) { return 1; });
  return d;
}
