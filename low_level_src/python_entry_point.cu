// Third party
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

// Dicomnode imports
#include"python/python_bounding_box.cu"
#include"python/python_cuda_management.cu"
#include"python/python_mirror.cu"

PYBIND11_MODULE(_cuda, m){
  m.doc() = "Dicomnode cuda library of functions, you shouldn't really need to\
 to import this, as there's wrappers for most of these functions";
  m.attr("__name__") = "dicomnode.math._cuda";

  apply_mirror_module(m);
  apply_cuda_management_module(m);
  apply_bounding_box_module(m);
}
