#include "random.h"

std::mt19937 &global_rng() {
  static std::random_device rd;
  static std::mt19937 gen(rd()); // Seed once!
  return gen;
}
