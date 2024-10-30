#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include<iostream>
#include<string>

#include"../gpu_code/dicom_node_gpu.cu"

using basis_t = pybind11::array_t<float>;
using domain_array = pybind11::array_t<int>;


template<typename T>
pybind11::array_t<T> interpolate_linear_templated(
  const pybind11::object& image,
  const pybind11::object& new_space
){


  Image<3, T> *device_image;

  cudaError_t cuda_error = cudaMalloc(&device_image,sizeof(Image<3, T>));
  load_image(device_image, image);


  cudaFree(device_image);
}

pybind11::array interpolate_linear(const pybind11::object& image,
                                   const pybind11::object& new_space
  ){
  const pybind11::array& raw_image = image.attr("raw");
  const std::string dtype = pybind11::str(raw_image.attr("dtype"));

  //Switch statement doesn't work because I am comparing strings
  if(dtype == "float32"){
    return interpolate_linear_templated<float>(image, new_space);
  } else if (dtype == "float64"){
    return interpolate_linear_templated<double>(image, new_space);
  } else if (dtype == "int8"){
    return interpolate_linear_templated<int8_t>(image, new_space);
  } else if (dtype == "int16"){
    return interpolate_linear_templated<int16_t>(image, new_space);
  } else if (dtype == "int32"){
    return interpolate_linear_templated<int32_t>(image, new_space);
  } else if (dtype == "int64"){
    return interpolate_linear_templated<int64_t>(image, new_space);
  } else if (dtype == "uint8"){
    return interpolate_linear_templated<uint8_t>(image, new_space);
  } else if (dtype == "uint16"){
    return interpolate_linear_templated<uint16_t>(image, new_space);
  } else if (dtype == "uint32"){
    return interpolate_linear_templated<uint32_t>(image, new_space);
  } else if (dtype == "uint64"){
    return interpolate_linear_templated<uint64_t>(image, new_space);
  } else if (dtype == "bool") {
    return interpolate_linear_templated<bool>(image, new_space);
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