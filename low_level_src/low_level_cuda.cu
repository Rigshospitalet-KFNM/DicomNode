#ifndef LOW_LEVEL_CUDA_DICOMNODE_H
#define LOW_LEVEL_CUDA_DICOMNODE_H

// Standard library
#include<iostream>
#include<stdint.h>
#include<float.h>
#include<vector>
#include<string.h>
#include<typeinfo>
#include<variant>

// Third party
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

// Dicomnode imports
#include"gpu_code/numpy_array.cu"
#include"gpu_code/mirror.cu"

PYBIND11_MODULE(_cuda, m){
  m.doc() = "pybind11 example plugin";
  m.attr("__name__") = "dicomnode.math._cuda";

  // mirror.cu
  apply_mirror_module(m);
}

#endif