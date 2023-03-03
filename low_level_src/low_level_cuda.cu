#ifndef LOW_LEVEL_CUDA_DICOMNODE_H
#define LOW_LEVEL_CUDA_DICOMNODE_H

#include <pybind11/pybind11.h>

int add(int i, int j){
  return i + j;
}

PYBIND11_MODULE(dicomnode_c_cuda, m){
  m.doc() = "pybind11 example plugin";

  m.def("add", &add, "A function that adds two numbers");
}

#endif