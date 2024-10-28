#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include<iostream>
#include<string>

#include"../gpu_code/dicom_node_gpu.cu"

using basis_t = pybind11::array_t<float>;
using domain_array = pybind11::array_t<int>;


template<typename T>
pybind11::array_t<T> interpolate_linear_templated(
  const pybind11::array_t<T>& raw_image,

){

}

pybind11::array interpolate_linear(const pybind11::object& image,
                                   const pybind11::object& new_space
  ){
  const pybind11::array& raw_image = image.attr("raw");
  const pybind11::object& original_space = image.attr("space");
  const basis_t& inverted_basis = pybind11::cast<basis_t>(original_space.attr("inverted_basis"));
  const pybind11::buffer_info& inverted_basis_buffer = inverted_basis.request(false);

  if(inverted_basis_buffer.ndim != 2){
    throw std::runtime_error("The Basis is not a 3 by 3 matrix");
  }
  const std::string dtype = pybind11::str(raw_image.attr("dtype"));
  //Switch statement doesn't work because I am comparing strings

  if(dtype == "float32"){
    return interpolate_linear_templated<float>(raw_image);
  } else if (dtype == "int32"){

  }

  const std::string error_message = "Unsupported dtype:" + dtype;
  throw std::runtime_error(error_message);
}

void apply_interpolation_module(pybind11::module& m){
  pybind11::module sub_module = m.def_submodule(
    "interpolation",
    "This module contains functions for resampling and interpolation.\n"
  );

  sub_module.def("linear_interpolate", &interpolate_linear);
}