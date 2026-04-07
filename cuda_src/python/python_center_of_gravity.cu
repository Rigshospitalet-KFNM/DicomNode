//
// Created by cjen0668 on 4/7/26.
//

#include "python_center_of_gravity.cuh"

#include "utilities.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include "../gpu_code/center_of_gravity.cuh"
#include "../gpu_code/core/cuda_management.cuh"
#include "../gpu_code/core/error.cuh"


template<typename T>
std::tuple<dicomNodeError_t, std::tuple<f32,f32,f32>> tpl_center_of_gravity(pybind11::object& python_image) {
  Image<3, T> image;

  T sum = 0;

  f32 x_cog = 1.0f, y_cog = -1.0f, z_cog = -1.0f;

  DicomNodeRunner runner{[&](dicomNodeError_t error) {
    free_image(&image);
  }};

  runner | [&]() {
    return load_image(&image, python_image);
  } | [&]() {
    return reduce_no_mem<8, VOLUME_SUM_OP<3, T>, T, Volume<3,T>>(
      image.elements(), &sum, image.volume
    );
  } | [&](){
    return reduce_no_mem<8, CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<T, DIMENSION::X>>(
      image.elements(), &x_cog, image.volume);
  } | [&](){
    return reduce_no_mem<8, CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<T, DIMENSION::Y>>(
      image.elements(), &y_cog, image.volume);
  } | [&](){
    return reduce_no_mem<8, CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<T, DIMENSION::Z>>(
      image.elements(), &z_cog, image.volume);
  } | [&]() {

    return free_image(&image);
  };

  f32 sumf = static_cast<f32>(sum);

  return std::make_tuple(runner.error(), std::make_tuple(x_cog / sumf, y_cog / sumf, z_cog / sumf));
}

std::tuple<dicomNodeError_t, std::tuple<f32,f32,f32>> center_of_gravity(pybind11::object& image) {
  const pybind11::array& raw_image = image.attr("raw");
  const pybind11::dtype& image_dtype = raw_image.dtype();

  if (image_dtype.is(pybind11::dtype::of<f32>())) {
    return tpl_center_of_gravity<f32>(image);
  } else if (image_dtype.is(pybind11::dtype::of<f64>())){
      return tpl_center_of_gravity<f64>(image);
  } else if (image_dtype.is(pybind11::dtype::of<i8>())){
      return tpl_center_of_gravity<i8>(image);
  } else if (image_dtype.is(pybind11::dtype::of<i16>())) {
      return tpl_center_of_gravity<i16>(image);
  } else if (image_dtype.is(pybind11::dtype::of<i32>())) {
      return tpl_center_of_gravity<i32>(image);
  } else if (image_dtype.is(pybind11::dtype::of<i64>())) {
      return tpl_center_of_gravity<i64>(image);
  } else if (image_dtype.is(pybind11::dtype::of<u8>())) {
      return tpl_center_of_gravity<u8>(image);
  } else if (image_dtype.is(pybind11::dtype::of<u16>())) {
      return tpl_center_of_gravity<u16>(image);
  } else if (image_dtype.is(pybind11::dtype::of<u32>())) {
      return tpl_center_of_gravity<u32>(image);
  } else if (image_dtype.is(pybind11::dtype::of<u64>())) {
      return tpl_center_of_gravity<u64>(image);
  }

  return std::make_tuple(dicomNodeError_t::INPUT_TYPE_ERROR, std::make_tuple(-1.0f,-1.0f, -1.0f));

}

void apply_center_of_gravity_module(pybind11::module &m) {
  m.def("center_of_gravity", &center_of_gravity);
}
