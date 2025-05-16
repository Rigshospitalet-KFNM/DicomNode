//
#include<stdint.h>
#include<tuple>
#include<execution>
#include<algorithm>
#include<utility>

#include<pybind11/numpy.h>

#include"core/core.hpp"
#include"utils.hpp"
#include"python_interpolation.hpp"

template<typename T, u8 DIM>
dicomNodeError_t interpolation(
  const T* in_image,
  T* out_image,
  const Space<DIM>& source_space,
  const Space<DIM>& destination_space
) {

  if(in_image == nullptr){
    return INPUT_TYPE_ERROR;
  }
  if(out_image == nullptr){
    return INPUT_TYPE_ERROR;
  }

  const size_t elements = destination_space.elements();
  //T* out_image_end = out_image + elements;

  // So just for future performance geeks. The reason why I don't reuse the
  // memory from out, is that it may not be sufficient to hold all the indexs
  // Say you have a u8 image. Then the maximum index is 255, which isn't a lot
  // for a volume. So I make a vector of Indexes
  std::vector<u64> indexes(elements);
  std::iota(indexes.begin(), indexes.end(), static_cast<u64>(0));

  std::transform(
    std::execution::par,
    indexes.begin(),
    indexes.end(),
    out_image,
    [&](const u64& flat_index){
      const Index<DIM> index = destination_space.extent.from_flat_index(
        flat_index
      );

      const Point<DIM> destination_point = destination_space.at_index(index);
      const Point<DIM> source_indexes = source_space.interpolate_point(destination_point);

      Point<3> bb_lower_corner;
      for(u8 i = 0; const f32& sidx : source_indexes){
        bb_lower_corner[i] = std::floor(sidx);
        i++;
      }

      Point<3> diff = source_indexes - bb_lower_corner;

      f64 out = 0;

      constexpr size_t bounding_points = std::pow(2, DIM);
      for(u64 bit_mask = 0; bit_mask < bounding_points; bit_mask++){
        Index<3> image_index;
        f64 multiplier = 1;

        for(u8 dim = 0; dim < DIM; dim++){
          if((bit_mask >> dim) & 1) {
            multiplier *= diff[dim];
            image_index[dim] = static_cast<i32>(std::ceil(source_indexes[dim]));
          } else {
            multiplier *= (1 - diff[dim]);
            image_index[dim] = static_cast<i32>(bb_lower_corner[dim]);
          }
        }
        auto opt_source_flat_index = source_space.extent.flat_index(image_index);
        if(opt_source_flat_index.has_value()){
          const u64& source_flat_index = *opt_source_flat_index;
          out += multiplier * static_cast<f64>(in_image[source_flat_index]);
        }
      }

      return static_cast<T>(out);
    }
  );

  return SUCCESS;
}


template<typename T>
std::tuple<dicomNodeError_t, pybind11::array_t<T>> templated_interpolate_linear(
  const pybind11::object& image,
  const pybind11::object& new_space
) {
  constexpr u8 DIM = 3;

  pybind11::array_t<T> out_array;

  const pybind11::object& image_space = image.attr("space");

  Space<DIM> source_space;
  Space<DIM> destination_space;

  const pybind11::array_t<T>& image_array = image.attr("raw").cast<const pybind11::array_t<T>>();
  const pybind11::buffer_info& source_buffer = image_array.request(false);

  DicomNodeRunner runner;
  runner
    | [&](){
      return dicomnode::load_space<3>(image_space, source_space);
  } | [&](){
      return dicomnode::load_space<3>(new_space, destination_space);

  } | [&](){
    const T* source_ptr = (T*) source_buffer.ptr;
    if (source_ptr == NULL){
      return dicomNodeError_t::UNABLE_TO_ACQUIRE_BUFFER;
    }
    const std::array<u32, DIM> shape = destination_space.extent.sizes;
    const std::array<size_t, DIM> strides = destination_space.extent.python_strides(sizeof(T));

    out_array = pybind11::array_t<T>(shape, strides);
    pybind11::buffer_info out_buffer = out_array.request(true);

    if(out_buffer.ptr == NULL){
      return dicomNodeError_t::UNABLE_TO_ACQUIRE_BUFFER;
    }

    if(out_buffer.size == SSIZE_ERROR){
      return dicomNodeError_t::NON_POSITIVE_SHAPE;
    }

    T* output_buffer_pointer = static_cast<T*>(out_buffer.ptr);

    return interpolation(
      source_ptr,
      output_buffer_pointer,
      source_space,
      destination_space
    );
  };

  return {runner.error(), out_array};
}




std::tuple<dicomNodeError_t, pybind11::array> interpolate_linear(
    const pybind11::object& image,
    const pybind11::object& new_space
) {
  DicomNodeRunner runner;

  pybind11::array return_array;

  runner
    | [&](){
    return dicomnode::is_instance(image, "dicomnode.math.image", "Image");
  } | [&](){
    return dicomnode::is_instance(new_space, "dicomnode.math.space", "Space");
  };

  if(runner.error()){
    return {runner.error(), return_array};
  }

  const pybind11::array& image_data = image.attr("raw");
  const pybind11::dtype& image_dtype = image_data.dtype();

  if(image_dtype.equal(pybind11::dtype::of<f32>())){
    return templated_interpolate_linear<f32>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<f64>())){
    return templated_interpolate_linear<f64>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<u8>())){
    return templated_interpolate_linear<u8>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<u16>())){
    return templated_interpolate_linear<u16>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<u32>())){
    return templated_interpolate_linear<u32>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<u64>())){
    return templated_interpolate_linear<u64>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<i8>())){
    return templated_interpolate_linear<i8>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<i16>())){
    return templated_interpolate_linear<i16>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<i32>())){
    return templated_interpolate_linear<i32>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<i64>())){
    return templated_interpolate_linear<i64>(image, new_space);
  }

  const std::string dtype = pybind11::str(image_data.dtype());
  const std::string error_message = "Unsupported dtype:" + dtype;
  throw std::runtime_error(error_message);
}


void apply_interpolation_module(pybind11::module m){
  pybind11::module sub_module = m.def_submodule(
    "interpolation",
    "This module contains functions for resampling and interpolation.\n"
  );

  sub_module.def("linear", &interpolate_linear);
}