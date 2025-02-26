/**
 * @file python_entry_point.cu
 * @author Demiguard (cjen0668@regionh.dk)
 * @brief This file is the entry point from for the python module
 * @version 0.1
 * @date 2024-12-18
 *
 * @copyright Copyright (c) 2024
 *
 */


// Third party
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

// Dicomnode imports
#include"python/python_bounding_box.cuh"
#include"python/python_cuda_management.cuh"
#include"python/python_mirror.cuh"
#include"python/python_interpolation.cuh"
#include"python/python_labeling.cuh"

PYBIND11_MODULE(_cuda, m){
  m.doc() = "Dicomnode cuda library of functions, you shouldn't really need to\
 to import this, as there's wrappers for most of these functions";
  m.attr("__name__") = "dicomnode.math._cuda";

  apply_cuda_management_module(m);
  apply_mirror_module(m);
  apply_bounding_box_module(m);
  apply_interpolation_module(m);
  apply_labeling_module(m);
}
