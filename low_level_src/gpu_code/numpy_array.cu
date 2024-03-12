#ifndef DICOMNODE_NUMPY_ARRAY
#define DICOMNODE_NUMPY_ARRAY

#include<variant>
#include<stdint.h>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

using numpy_types = std::variant<
  double,
  float,
  int8_t,
  int16_t,
  int32_t,
  int64_t,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  >;


using numpy_array = std::variant<
  pybind11::array_t<double>,
  pybind11::array_t<float>,
  pybind11::array_t<int8_t>,
  pybind11::array_t<int16_t>,
  pybind11::array_t<int32_t>,
  pybind11::array_t<int64_t>,
  pybind11::array_t<uint8_t>,
  pybind11::array_t<uint16_t>,
  pybind11::array_t<uint32_t>,
  pybind11::array_t<uint64_t>,
  >;

void to_typed_array(pybind11::array arr){
  pybind11::dtype arr_dtype = arr.dtype();
}

#endif