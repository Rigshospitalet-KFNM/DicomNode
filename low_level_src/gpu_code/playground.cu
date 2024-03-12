#ifndef DICOMNODE_PLAYGROUND
#define DICOMNODE_PLAYGROUND

// Standard library
#include<iostream>

// Third party imports
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

// Dicomnode imports
#include"numpy_array.cu"

void print_dtype(pybind11::array arr){
  pybind11::dtype arr_dtype = arr.dtype();
  pybind11::detail::str_attr_accessor type_name = arr_dtype.attr("name");

  
}


#endif