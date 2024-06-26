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
#include"gpu_code/cuda_management.cu"
#include"gpu_code/mirror.cu"
#include"gpu_code/tricubic_interpolation.cu"

PYBIND11_MODULE(_cuda, m){
  m.doc() = "Dicomnode cuda library of functions, you shouldn't really need to\
 to import this, as there's wrappers for most of these functions";
  m.attr("__name__") = "dicomnode.math._cuda";

  apply_mirror_module(m);
  apply_tricubic_interpolation_module(m);
  apply_cuda_management_module(m);

}

#endif