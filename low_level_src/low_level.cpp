#ifndef LOW_LEVEL_DICOMNODE_H
#define LOW_LEVEL_DICOMNODE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

/* https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html */
/*
struct buffer_info {
    void *ptr;
    pybind11::ssize_t itemsize;
    std::string format;
    pybind11::ssize_t ndim;
    std::vector<pybind11::ssize_t> shape;
    std::vector<pybind11::ssize_t> strides;
};
*/

namespace py = pybind11;


py::array_t<double> add_arrays(
    py::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>input1,
    py::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>input2
  ){
  py::buffer_info buf_1 = input1.request();
  py::buffer_info buf_2 = input2.request();

  if (buf_1.ndim != 1 || buf_2.ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");

  if (buf_1.size != buf_2.size)
    throw std::runtime_error("Input shapes must match");

  auto result = pybind11::array_t<double>(buf_1.size);

  py::buffer_info buf_3 = result.request();

  double *ptr1 = static_cast<double *>(buf_1.ptr);
  double *ptr2 = static_cast<double *>(buf_2.ptr);
  double *ptr3 = static_cast<double *>(buf_3.ptr);

  for (size_t idx = 0; idx < buf_1.shape[0]; idx++)
    ptr3[idx] = ptr1[idx] + ptr2[idx];

  return result;
}

PYBIND11_MODULE(_c, m){
  m.doc() = "pybind11 example plugin";
  m.attr("__name__") = "dicomnode.math._c";

  m.def("add_arrays", &add_arrays, py::return_value_policy::move, "A function that adds two numpy arrays");
}

#endif