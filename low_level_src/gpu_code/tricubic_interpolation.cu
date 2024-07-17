#ifndef DICOMNODE_TRICUBIC_INTERPOLATION
#define DICOMNODE_TRICUBIC_INTERPOLATION

// Standard library
#include<assert.h>
#include<stdint.h>
#include<iostream>
#include<exception>
#include<string>

// Thrid party
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

// Dicomnode Cuda Files
#include"core/core.cu"

namespace py = pybind11;

// If constant memory size becomes a problem, move this standard memory
// This matrix takes up 16 kB out of 64 kB sooo....
// Constant matrix "stolen" from:
// https://github.com/danielguterding/pytricubic/blob/master/src/tricubic.cpp
__device__ const int C[64][64] = {
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {9, -9, -9, 9, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-6, 6, 6, -6, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-6, 6, 6, -6, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {4, -4, -4, 4, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
      {-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {9, -9, 0, 0, -9, 9, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-6, 6, 0, 0, 6, -6, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, 0, 0, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0},
      {9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0},
      {-27, 27, 27, -27, 27, -27, -27, 27, -18, -9, 18, 9, 18, 9, -18, -9, -18, 18, -9, 9, 18, -18, 9, -9, -18, 18, 18, -18, -9, 9, 9, -9, -12, -6, -6, -3, 12, 6, 6, 3, -12, -6, 12, 6, -6, -3, 6, 3, -12, 12, -6, 6, -6, 6, -3, 3, -8, -4, -4, -2, -4, -2, -2, -1},
      {18, -18, -18, 18, -18, 18, 18, -18, 9, 9, -9, -9, -9, -9, 9, 9, 12, -12, 6, -6, -12, 12, -6, 6, 12, -12, -12, 12, 6, -6, -6, 6, 6, 6, 3, 3, -6, -6, -3, -3, 6, 6, -6, -6, 3, 3, -3, -3, 8, -8, 4, -4, 4, -4, 2, -2, 4, 4, 2, 2, 2, 2, 1, 1},
      {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0},
      {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 9, -9, 9, -9, -9, 9, -9, 9, 12, -12, -12, 12, 6, -6, -6, 6, 6, 3, 6, 3, -6, -3, -6, -3, 8, 4, -8, -4, 4, 2, -4, -2, 6, -6, 6, -6, 3, -3, 3, -3, 4, 2, 4, 2, 2, 1, 2, 1},
      {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -6, 6, -6, 6, 6, -6, 6, -6, -8, 8, 8, -8, -4, 4, 4, -4, -3, -3, -3, -3, 3, 3, 3, 3, -4, -4, 4, 4, -2, -2, 2, 2, -4, 4, -4, 4, -2, 2, -2, 2, -2, -2, -2, -2, -1, -1, -1, -1},
      {2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-6, 6, 0, 0, 6, -6, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {4, -4, 0, 0, -4, 4, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, 0, 0, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
      {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0},
      {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 12, -12, 6, -6, -12, 12, -6, 6, 9, -9, -9, 9, 9, -9, -9, 9, 8, 4, 4, 2, -8, -4, -4, -2, 6, 3, -6, -3, 6, 3, -6, -3, 6, -6, 3, -3, 6, -6, 3, -3, 4, 2, 2, 1, 4, 2, 2, 1},
      {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -8, 8, -4, 4, 8, -8, 4, -4, -6, 6, 6, -6, -6, 6, 6, -6, -4, -4, -2, -2, 4, 4, 2, 2, -3, -3, 3, 3, -3, -3, 3, 3, -4, 4, -2, 2, -4, 4, -2, 2, -2, -2, -1, -1, -2, -2, -1, -1},
      {4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
      {-12, 12, 12, -12, 12, -12, -12, 12, -8, -4, 8, 4, 8, 4, -8, -4, -6, 6, -6, 6, 6, -6, 6, -6, -6, 6, 6, -6, -6, 6, 6, -6, -4, -2, -4, -2, 4, 2, 4, 2, -4, -2, 4, 2, -4, -2, 4, 2, -3, 3, -3, 3, -3, 3, -3, 3, -2, -1, -2, -1, -2, -1, -2, -1},
      {8, -8, -8, 8, -8, 8, 8, -8, 4, 4, -4, -4, -4, -4, 4, 4, 4, -4, 4, -4, -4, 4, -4, 4, 4, -4, -4, 4, 4, -4, -4, 4, 2, 2, 2, 2, -2, -2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 1, 1, 1, 1, 1, 1, 1, 1}
};

template<typename T>
struct TricubicInterpolationCoefficients {
  T data[64];
};

template<typename T>
__device__ T idx(const T* image, const Space<3> dims, const Index<3> i, const T defaultval){
  cuda::std::optional<uint64_t> flat_index = dims.flat_index(i.x(), i.y(), i.z());
  return flat_index.has_value() ? image[flat_index.value()] : defaultval;
}

template<typename T>
__device__ T dfdx(const T* image, const Space<3> dims, const Index<3> i, const T defaultval){
  return 0.5 * (idx(image, dims, {i.x() + 1, i.y(), i.z()}, defaultval)
              - idx(image, dims, {i.x() - 1, i.y(), i.z()}, defaultval));
}

template<typename T>
__device__ T dfdy(const T* image, const Space<3> dims, const Index<3> i, const T defaultval){
  return 0.5 * (idx(image, dims, {i.x(), i.y() + 1, i.z()}, defaultval)
              - idx(image, dims, {i.x(), i.y() - 1, i.z()}, defaultval));
}

template<typename T>
__device__ T dfdz(const T* image, const Space<3> dims, const Index<3> i, const T defaultval){
  return 0.5 * (idx(image, dims, {i.x(), i.y(), i.z() + 1}, defaultval)
              - idx(image, dims, {i.x(), i.y(), i.z() - 1}, defaultval));
}

template<typename T>
__device__ T d2fdxdy(const T* image, const Space<3> dims, const Index<3> i, const T defaultval){
  return 0.25 * (idx(image, dims, {i.x() + 1, i.y() + 1, i.z()}, defaultval)
               + idx(image, dims, {i.x() - 1, i.y() - 1, i.z()}, defaultval)
               - idx(image, dims, {i.x() + 1, i.y() - 1, i.z()}, defaultval)
               - idx(image, dims, {i.x() - 1, i.y() + 1, i.z()}, defaultval));
}

template<typename T>
__device__ T d2fdxdz(const T* image, const Space<3> dims, const Index<3> i, const T defaultval){
  return 0.25 * (idx(image, dims, {i.x() + 1, i.y(), i.z() + 1}, defaultval)
               + idx(image, dims, {i.x() - 1, i.y(), i.z() - 1}, defaultval)
               - idx(image, dims, {i.x() + 1, i.y(), i.z() - 1}, defaultval)
               - idx(image, dims, {i.x() - 1, i.y(), i.z() + 1}, defaultval));
}

template<typename T>
__device__ T d2fdydz(const T* image, const Space<3> dims, const Index<3> i, const T defaultval){
  return 0.25 * (idx(image, dims, {i.x(), i.y() + 1, i.z() + 1}, defaultval)
               + idx(image, dims, {i.x(), i.y() - 1, i.z() - 1}, defaultval)
               - idx(image, dims, {i.x(), i.y() + 1, i.z() - 1}, defaultval)
               - idx(image, dims, {i.x(), i.y() - 1, i.z() + 1}, defaultval));
}

template<typename T>
__device__ T d3fdxdydz(const T* image, const Space<3> dims, const Index<3> i, const T defaultval){
  return 0.125 * (idx(image, dims, {i.x() + 1, i.y() + 1, i.z() + 1}, defaultval)
                + idx(image, dims, {i.x() + 1, i.y() - 1, i.z() - 1}, defaultval)
                + idx(image, dims, {i.x() - 1, i.y() + 1, i.z() - 1}, defaultval)
                + idx(image, dims, {i.x() - 1, i.y() - 1, i.z() + 1}, defaultval)
                - idx(image, dims, {i.x() - 1, i.y() + 1, i.z() + 1}, defaultval)
                - idx(image, dims, {i.x() + 1, i.y() - 1, i.z() + 1}, defaultval)
                - idx(image, dims, {i.x() + 1, i.y() + 1, i.z() - 1}, defaultval)
                - idx(image, dims, {i.x() - 1, i.y() - 1, i.z() - 1}, defaultval));
}


/**
 * @brief
 *
 * @param image
 * @param x_max
 * @param y_max
 * @param z_max

 */
template<typename T>
__global__ void tricubic_interpolation_kernel(const T* image,
                                              const Space<3> imagedim,
                                              const T* affine,
                                              const T* targets,
                                              const size_t num_targets,
                                              T* destination,
                                              const T minimum_value){
  const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(!(tid < num_targets)){
    return;
  }

  const T x_source = targets[(tid * 3) + 0];
  const T y_source = targets[(tid * 3) + 1];
  const T z_source = targets[(tid * 3) + 2];

  const T x_target = x_source * affine[0]
                   + y_source * affine[1]
                   + z_source * affine[2]
                   +            affine[3];

  const T y_target = x_source * affine[4]
                   + y_source * affine[5]
                   + z_source * affine[6]
                   +            affine[7];

  const T z_target = x_source * affine[8]
                   + y_source * affine[9]
                   + z_source * affine[10]
                   +            affine[11];

  const int32_t xlc = (int32_t)floor(x_target); // X lower corner
  const int32_t ylc = (int32_t)floor(y_target); // Y lower corner
  const int32_t zlc = (int32_t)floor(z_target); // Z lower corner

  T values[64] = {
    // values
    idx(image, imagedim, {xlc, ylc, zlc}, minimum_value),
    idx(image, imagedim, {xlc + 1, ylc, zlc}, minimum_value),
    idx(image, imagedim, {xlc, ylc + 1, zlc}, minimum_value),
    idx(image, imagedim, {xlc + 1, ylc + 1, zlc}, minimum_value),
    idx(image, imagedim, {xlc, ylc, zlc + 1}, minimum_value),
    idx(image, imagedim, {xlc + 1, ylc, zlc + 1}, minimum_value),
    idx(image, imagedim, {xlc, ylc + 1, zlc + 1}, minimum_value),
    idx(image, imagedim, {xlc + 1, ylc + 1, zlc + 1}, minimum_value),
    // df / dx
    dfdx(image, imagedim, {xlc, ylc, zlc}, minimum_value),
    dfdx(image, imagedim, {xlc + 1, ylc, zlc}, minimum_value),
    dfdx(image, imagedim, {xlc, ylc + 1, zlc}, minimum_value),
    dfdx(image, imagedim, {xlc + 1, ylc + 1, zlc}, minimum_value),
    dfdx(image, imagedim, {xlc, ylc, zlc + 1}, minimum_value),
    dfdx(image, imagedim, {xlc + 1, ylc, zlc + 1}, minimum_value),
    dfdx(image, imagedim, {xlc, ylc + 1, zlc + 1}, minimum_value),
    dfdx(image, imagedim, {xlc + 1, ylc + 1, zlc + 1}, minimum_value),
    // df / dy
    dfdy(image, imagedim, {xlc, ylc, zlc}, minimum_value),
    dfdy(image, imagedim, {xlc + 1, ylc, zlc}, minimum_value),
    dfdy(image, imagedim, {xlc, ylc + 1, zlc}, minimum_value),
    dfdy(image, imagedim, {xlc + 1, ylc + 1, zlc}, minimum_value),
    dfdy(image, imagedim, {xlc, ylc, zlc + 1}, minimum_value),
    dfdy(image, imagedim, {xlc + 1, ylc, zlc + 1}, minimum_value),
    dfdy(image, imagedim, {xlc, ylc + 1, zlc + 1}, minimum_value),
    dfdy(image, imagedim, {xlc + 1, ylc + 1, zlc + 1}, minimum_value),
    // df / dz
    dfdz(image, imagedim, {xlc, ylc, zlc}, minimum_value),
    dfdz(image, imagedim, {xlc + 1, ylc, zlc}, minimum_value),
    dfdz(image, imagedim, {xlc, ylc + 1, zlc}, minimum_value),
    dfdz(image, imagedim, {xlc + 1, ylc + 1, zlc}, minimum_value),
    dfdz(image, imagedim, {xlc, ylc, zlc + 1}, minimum_value),
    dfdz(image, imagedim, {xlc + 1, ylc, zlc + 1}, minimum_value),
    dfdz(image, imagedim, {xlc, ylc + 1, zlc + 1}, minimum_value),
    dfdz(image, imagedim, {xlc + 1, ylc + 1, zlc + 1}, minimum_value),
    // d2f / dxdy
    d2fdxdy(image, imagedim, {xlc, ylc, zlc}, minimum_value),
    d2fdxdy(image, imagedim, {xlc + 1, ylc, zlc}, minimum_value),
    d2fdxdy(image, imagedim, {xlc, ylc + 1, zlc}, minimum_value),
    d2fdxdy(image, imagedim, {xlc + 1, ylc + 1, zlc}, minimum_value),
    d2fdxdy(image, imagedim, {xlc, ylc, zlc + 1}, minimum_value),
    d2fdxdy(image, imagedim, {xlc + 1, ylc, zlc + 1}, minimum_value),
    d2fdxdy(image, imagedim, {xlc, ylc + 1, zlc + 1}, minimum_value),
    d2fdxdy(image, imagedim, {xlc + 1, ylc + 1, zlc + 1}, minimum_value),
    // d2f / dxdz
    d2fdxdz(image, imagedim, {xlc, ylc, zlc}, minimum_value),
    d2fdxdz(image, imagedim, {xlc + 1, ylc, zlc}, minimum_value),
    d2fdxdz(image, imagedim, {xlc, ylc + 1, zlc}, minimum_value),
    d2fdxdz(image, imagedim, {xlc + 1, ylc + 1, zlc}, minimum_value),
    d2fdxdz(image, imagedim, {xlc, ylc, zlc + 1}, minimum_value),
    d2fdxdz(image, imagedim, {xlc + 1, ylc, zlc + 1}, minimum_value),
    d2fdxdz(image, imagedim, {xlc, ylc + 1, zlc + 1}, minimum_value),
    d2fdxdz(image, imagedim, {xlc + 1, ylc + 1, zlc + 1}, minimum_value),
    // d2f / dydz
    d2fdydz(image, imagedim, {xlc, ylc, zlc}, minimum_value),
    d2fdydz(image, imagedim, {xlc + 1, ylc, zlc}, minimum_value),
    d2fdydz(image, imagedim, {xlc, ylc + 1, zlc}, minimum_value),
    d2fdydz(image, imagedim, {xlc + 1, ylc + 1, zlc}, minimum_value),
    d2fdydz(image, imagedim, {xlc, ylc, zlc + 1}, minimum_value),
    d2fdydz(image, imagedim, {xlc + 1, ylc, zlc + 1}, minimum_value),
    d2fdydz(image, imagedim, {xlc, ylc + 1, zlc + 1}, minimum_value),
    d2fdydz(image, imagedim, {xlc + 1, ylc + 1, zlc + 1}, minimum_value),
    // d3f / dxdydz
    d3fdxdydz(image, imagedim, {xlc, ylc, zlc}, minimum_value),
    d3fdxdydz(image, imagedim, {xlc + 1, ylc, zlc}, minimum_value),
    d3fdxdydz(image, imagedim, {xlc, ylc + 1, zlc}, minimum_value),
    d3fdxdydz(image, imagedim, {xlc + 1, ylc + 1, zlc}, minimum_value),
    d3fdxdydz(image, imagedim, {xlc, ylc, zlc + 1}, minimum_value),
    d3fdxdydz(image, imagedim, {xlc + 1, ylc, zlc + 1}, minimum_value),
    d3fdxdydz(image, imagedim, {xlc, ylc + 1, zlc + 1}, minimum_value),
    d3fdxdydz(image, imagedim, {xlc + 1, ylc + 1, zlc + 1}, minimum_value)
  };
  T coef[64];
  for(int i = 0; i < 64; ++i){
    coef[i] = 0.0;
    for(int j = 0; j < 64; ++j){
      coef[i] += C[i][j] * values[j];
    }
  }

  const T dx = x_target - xlc;
  const T dy = y_target - ylc;
  const T dz = z_target - zlc;

  uint8_t ijkn = 0;
  T dest = 0.0;
  T dzpow = 1.0;

  for(int k = 0; k < 4; ++k) {
    T dypow = 1.0;
    for(int j = 0; j < 4; ++j){
      dest += dzpow * dypow * (coef[ijkn]
                       + dx * (coef[ijkn + 1]
                       + dx * (coef[ijkn + 2]
                       + dx *  coef[ijkn + 3])));
      ijkn += 4;
      dypow *= dy;
    }
    dzpow *= dz;
  }

  destination[tid] = dest;
}

template<typename T>
void tricubic_interpolation(const py::array_t<T, array_flags> data,
                            const py::array_t<double, array_flags> affine,
                            const py::array_t<T, array_flags> targets,
                            const T minimum_value){
  const pybind11::buffer_info data_buffer = data.request(false);
  if (data_buffer.ndim != 3){
    throw std::runtime_error("Data space must be a 3 dimensional volume");
  }

  const pybind11::buffer_info affine_buffer = affine.request(false);
  if (affine_buffer.ndim != 2){
    throw std::runtime_error("Affine is not a 4x4 matrix!");
  }
  if (affine_buffer.shape[0] != 4 || affine_buffer.shape[1] != 4){
    throw std::runtime_error("Affine is not a 4x4 matrix!");
  }

  const pybind11::buffer_info targets_buffer = targets.request(false);
  if (targets_buffer.ndim != 2){
    throw std::runtime_error("Targets must be a (3,n) dimensional array");
  }
  if (targets_buffer.shape[0] != 3){
    throw std::runtime_error("Targets must be a (3,n) dimensional array");
  }

  const uint32_t x = data_buffer.shape[0];
  const uint32_t y = data_buffer.shape[1];
  const uint32_t z = data_buffer.shape[2];
  const uint32_t num_targets = targets_buffer.shape[1];

  const size_t data_size = x * y * z * sizeof(T);
  const size_t affine_size = 16 * sizeof(T);
  const size_t targets_size = num_targets * 3 * sizeof(T);
  const size_t destinations_size = num_targets * sizeof(T);

  const size_t required_memory = data_size + affine_size + targets_size + destinations_size;

  size_t free_memory, total_memory;
  run_cuda([&](){return cudaMemGetInfo(&free_memory, &total_memory);},
           [&](cudaError_t error){
              // No free here because none of the points could have been alocated yet!
              throw std::runtime_error("Could not get free memory of device, something is very wrong");
            });


  if(free_memory < required_memory){
    std::string error_message = "Attempted to allocate: ";
    error_message += std::to_string(required_memory);
    error_message += " However only ";
    error_message += std::to_string(free_memory);
    error_message += " are available!";
    throw std::runtime_error(error_message);
  }

  const uint32_t threads = 128;
  const uint32_t blocks = num_targets % threads == 0 ? num_targets / threads : (num_targets / threads) + 1;

  T* gpu_data = nullptr;
  T* gpu_affine = nullptr;
  T* gpu_targets = nullptr;
  T* gpu_destinations = nullptr;

  auto error_function = [&](cudaError_t error){
    free_device_memory(&gpu_data, &gpu_affine, &gpu_targets, &gpu_destinations);
    throw std::runtime_error("Failed to allocate GPU resources!");
  };

  const ssize_t result_size = ((int32_t)num_targets) * sizeof(T);
  py::array_t<T, array_flags> result{result_size};
  py::buffer_info result_buffer = result.request(true);

  // I hope you like monads
  CudaRunner actions{error_function};
  actions | [&](){return cudaMalloc(&gpu_data, data_size);}
          | [&](){return cudaMalloc(&gpu_affine, affine_size);}
          | [&](){return cudaMalloc(&gpu_targets, targets_size);}
          | [&](){return cudaMalloc(&gpu_destinations, destinations_size);}
          | [&](){return cudaMemcpy(gpu_data,
                                    data_buffer.ptr,
                                    data_size,
                                    cudaMemcpyHostToDevice);}
          | [&](){return cudaMemcpy(gpu_affine,
                                    affine_buffer.ptr,
                                    affine_size,
                                    cudaMemcpyHostToDevice);}
          | [&](){return cudaMemcpy(gpu_targets,
                                    targets_buffer.ptr,
                                    targets_size,
                                    cudaMemcpyHostToDevice);}
          | [&](){
            const Space dims = Space<3>(x,y,z);
            tricubic_interpolation_kernel<<<blocks, threads>>>(gpu_data, dims,
                                                               gpu_affine,
                                                               gpu_targets, num_targets,
                                                               gpu_destinations,
                                                               minimum_value);
            return cudaGetLastError();}
          | [&](){
            return cudaMemcpy(result_buffer.ptr,
                              gpu_destinations,
                              sizeof(T) * num_targets,
                              cudaMemcpyDeviceToHost);
          };


  free_device_memory(&gpu_data, &gpu_affine, &gpu_targets, &gpu_destinations);
}


void apply_tricubic_interpolation_module(py::module& m){
  m.def("tricubic_interpolation", &tricubic_interpolation<float>);
}



#endif