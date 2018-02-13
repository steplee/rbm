#pragma once

#include <cmath>

inline float max(float a,float b) { return a>b?a:b; }
inline float min(float a,float b) { return a<b?a:b; }

inline float sigmoid(float x) {
  return 1/(1+exp(-x));
}

inline void outer_product(float* out, const float *x, const float *y, int sx, int sy) {
  for (int ix=0; ix<sx; ix++) {
    for (int iy=0; iy<sy; iy++) {
      out[ix*sy+iy] = x[ix]*y[iy];
    }
  }
}
