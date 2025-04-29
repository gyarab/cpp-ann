#include "matrix.h"
#include "random.h"
#include <cmath>
#include <cstdlib>
#include <random>

inline void randomize(Matrix &weights, int input_size) {
  float min = -1.0f;
  float max = 1.0f;
  weights.applyFunction([min, max](float n) {
    return min + static_cast<float>(rand()) / RAND_MAX * (max - min);
  });
}

inline void he(Matrix &weights, int input_size) {
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / input_size));

  for (int i = 0; i < weights.getR(); ++i)
    for (int j = 0; j < weights.getC(); ++j)
      weights.set(i, j, dist(global_rng()));
}
