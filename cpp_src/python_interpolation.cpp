//
#include<stdint.h>
#include<tuple>
#include<execution>
#include<algorithm>

#include<pybind11/numpy.h>

#include"core/core.hpp"
#include"utils.hpp"
#include"python_interpolation.hpp"

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

  const pybind11::array_t<T>& image_array = image.attr("raw").cast<const pybind11::array_t<T>&>();
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
    const size_t shape[DIM] = {
      0,0,0
    };

    const size_t strides[DIM] = {
      0,0,0
    };

    out_array = pybind11::array_t<T>(shape, strides);
    pybind11::buffer_info out_buffer = out_array.request(true);

    if(out_buffer.ptr == NULL){
      return dicomNodeError_t::UNABLE_TO_ACQUIRE_BUFFER;
    }

    if(out_buffer.size == SSIZE_ERROR){
      return dicomNodeError_t::NON_POSITIVE_SHAPE;
    }

    T* output_buffer_pointer = static_cast<T*>(out_buffer.ptr);
    T* output_buffer_end = output_buffer_pointer + out_buffer.size;


    std::iota(output_buffer_pointer, output_buffer_end, static_cast<T>(0));

    std::transform(
      std::execution::par,
      output_buffer_pointer,
      output_buffer_end,
      output_buffer_pointer,
      [&](const T& flat_index){
        const Space<DIM>& thread_source_space = source_space;
        const Space<DIM>& thread_destination_space = destination_space;

        // Ok because of this was initialized with iota
        const uint64_t destination_image_index = static_cast<uint64_t>(flat_index);

        const Index<DIM> index = thread_destination_space.extent.from_flat_index(
          destination_image_index
        );

        const Point<DIM> destination_point = thread_destination_space.at_index(index);
        const Point<DIM> source_indexes = thread_source_space.interpolate_point(destination_point);

        Point<3> bb_lower_corner;
        for(u8 i = 0; const f32& sidx : source_indexes){
          bb_lower_corner[i] = std::floor(sidx);
          i++;
        }

        Point<3> diff = source_indexes - bb_lower_corner;

        T out = 0;

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
              image_index[dim] = static_cast<i32>(bb_lower_corner[i]);
            }
          }
          auto opt_source_flat_index = thread_source_space.extent.flat_index(image_index);
          if(opt_source_flat_index.has_value()){
            const u64& source_flat_index = *opt_source_flat_index;
            out += multiplier * source_ptr[source_flat_index];
          }
        }

        return out;
      }
    );

   return SUCCESS;
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

  if(image_dtype.equal(pybind11::dtype::of<float>())){
    return templated_interpolate_linear<float>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<uint8_t>())){
    return templated_interpolate_linear<uint8_t>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<uint16_t>())){
    return templated_interpolate_linear<uint16_t>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<uint32_t>())){
    return templated_interpolate_linear<uint32_t>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<int8_t>())){
    return templated_interpolate_linear<int8_t>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<int16_t>())){
    return templated_interpolate_linear<int16_t>(image, new_space);
  } else if(image_dtype.equal(pybind11::dtype::of<int32_t>())){
    return templated_interpolate_linear<int32_t>(image, new_space);
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