//
// Created by cjen0668 on 4/7/26.
//

#include "python_center_of_gravity.cuh"
#include <pybind11/numpy.h>

#include "../gpu_code/core/core.cuh"
#include "utilities.cuh"
#include "../gpu_code/center_of_gravity.cuh"


template<typename T>
std::tuple<dicomNodeError_t, std::tuple<f32,f32,f32>> tpl_python_center_of_gravity(const pybind11::array_t<T>& python_array) {
  Volume<3, T> volume;

  std::array<f32, 3> cog{-1.0f, -1.0f, -1.0f};

  DicomNodeRunner runner{[&](dicomNodeError_t error) {
    free_volume(&volume);
  }};

  runner | [&]() {
    return load_volume_from_array(&volume, python_array);
  } | [&]() {
    return CENTER_OF_GRAVITY::center_of_gravity<T>(volume, cog);
  } | [&]() {
    return free_volume(&volume);
  };

  return std::make_tuple(runner.error(), std::make_tuple(cog[0], cog[1], cog[2]));
}

std::tuple<dicomNodeError_t, std::tuple<f32,f32,f32>> python_center_of_gravity(const pybind11::array& image) {
  const pybind11::dtype& image_dtype = image.dtype();

  if (image_dtype.is(pybind11::dtype::of<f32>())) {
    return tpl_python_center_of_gravity<f32>(image);
  }

  return std::make_tuple(dicomNodeError_t::INPUT_TYPE_ERROR, std::make_tuple(-1.0f,-1.0f, -1.0f));
}

void apply_center_of_gravity_module(pybind11::module &m) {
  m.def("center_of_gravity", &python_center_of_gravity);
}
