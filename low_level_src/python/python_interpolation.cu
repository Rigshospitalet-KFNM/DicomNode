#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include<iostream>
#include<string>
#include<tuple>

#include"../gpu_code/dicom_node_gpu.cuh"
#include"utilities.cuh"

using basis_t = pybind11::array_t<float>;
using domain_array = pybind11::array_t<uint32_t>;

template<typename T>
std::tuple<dicomNodeError_t, python_array<T>> interpolate_linear_templated(
  const pybind11::object& image,
  const pybind11::object& new_space
){
  Space<3> destination_space;
  dicomNodeError_t error = load_space(&destination_space, new_space);
  if (error){
    return {error , python_array<T>(1)};
  }
  const size_t image_size = get_image_size<T>(image);
  const size_t out_image_size = get_image_size<T>(new_space);

  if(!out_image_size){
    return {dicomNodeError_t::INPUT_TYPE_ERROR, python_array<T>(1)};
  }
  const size_t out_image_elements = out_image_size / sizeof(T);

  const size_t shape[3] = {
    destination_space.domain[0],
    destination_space.domain[1],
    destination_space.domain[2]
  };
  const size_t strides[3] = {
    destination_space.domain[1] * destination_space.domain[2] * sizeof(T),
    destination_space.domain[2] * sizeof(T),
    sizeof(T)
  };

  pybind11::array_t<T> out_array(shape, strides);
  pybind11::buffer_info out_buffer = out_array.request(true);

  Texture<T> *device_texture = nullptr;
  T* device_out_image = nullptr;

  auto error_function = [&](dicomNodeError_t _){
    std::cout << "Error Trigger!\n";
    free_texture<T>(&device_texture);
    free_device_memory(&device_out_image);
  };

  DicomNodeRunner runner{error_function};
  runner
    | [&](){ return check_buffer_pointers(std::cref(out_buffer), out_image_elements);}
    | [&](){ return cudaMalloc(&device_texture, sizeof(Texture<T>)); }
    | [&](){
      return load_texture_from_python_image<T>(
        device_texture,
        image
      );
    }
    | [&](){
      return cudaMalloc(&device_out_image, out_image_size);}
    | [&](){ return gpu_interpolation_linear<T>(
      device_texture,
      std::cref(destination_space),
      device_out_image
    );}
    | [&](){
      return cudaMemcpy(out_buffer.ptr, device_out_image, out_image_size, cudaMemcpyDefault);
    }
    | [&](){
      return free_texture<T>(&device_texture);}
    | [&](){
      free_device_memory(&device_out_image);
      return dicomNodeError_t::SUCCESS;
    };

  return {runner.error(), out_array};
}

std::tuple<dicomNodeError_t, pybind11::array> interpolate_linear(const pybind11::object& image,
                                   const pybind11::object& new_space
  ){
  const pybind11::array& raw_image = image.attr("raw");
  const std::string dtype = pybind11::str(raw_image.attr("dtype"));

  //Switch statement doesn't work because I am comparing strings
  if(dtype == "float32"){
    return interpolate_linear_templated<float>(image, new_space);
  } else if (dtype == "uint8") {
    return interpolate_linear_templated<uint8_t>(image, new_space);
  } if (dtype == "uint16") {
    return interpolate_linear_templated<uint16_t>(image, new_space);
  } if (dtype == "uint32") {
    return interpolate_linear_templated<uint32_t>(image, new_space);
  } else if (dtype == "int8") {
    return interpolate_linear_templated<int8_t>(image, new_space);
  } if (dtype == "int16") {
    return interpolate_linear_templated<int16_t>(image, new_space);
  } if (dtype == "int32") {
    return interpolate_linear_templated<int32_t>(image, new_space);
  }

  // The other types are not supported by the hardware :(
  // Well int32 are

  const std::string error_message = "Unsupported dtype:" + dtype;
  throw std::runtime_error(error_message);
}

void apply_interpolation_module(pybind11::module& m){
  pybind11::module sub_module = m.def_submodule(
    "interpolation",
    "This module contains functions for resampling and interpolation.\n"
  );

  sub_module.def("linear", &interpolate_linear);
}