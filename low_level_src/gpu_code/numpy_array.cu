#ifndef DICOMNODE_NUMPY_ARRAY
#define DICOMNODE_NUMPY_ARRAY

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

const int array_flags = pybind11::array_t<int>::forcecast | pybind11::array_t<int>::c_style;

#endif