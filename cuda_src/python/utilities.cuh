#pragma once

#include<functional>
#include<string>
#include<vector>
#include<iostream>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include"../gpu_code/dicom_node_gpu.cuh"

//# REGION DECLARATIONS

constexpr int PYBIND_ARRAY_FLAGS = pybind11::array::c_style | pybind11::array::forcecast;

template<typename T>
using python_array = pybind11::array_t<T, PYBIND_ARRAY_FLAGS>;

std::string get_byte_string (size_t bytes);

// Header declaration, definitions in utilities.cu
/**
 * @brief Checks if a python object is of an expected type. As python is not
 * staticly typed and you can't trust that users don't pass stupid shit.
 *
 * @param python_object
 * @param module_name
 * @param instance_type
 * @return dicomNodeError_t
 */
dicomNodeError_t is_instance(
  const pybind11::object& python_object,
  const char* module_name,
  const char* instance_type
);

dicomNodeError_t load_into_host_space(
  Space<3>* host_space,
  const pybind11::object& space
);

dicomNodeError_t check_buffer_pointers(
  const pybind11::buffer_info& buffer, const size_t elements
);

dicomNodeError_t load_into_device_space(
  Space<3>* device_space,
  const pybind11::object& space
);

template<typename... Ts>
dicomNodeError_t check_buffer_pointers(
  const pybind11::buffer_info& buffer,
  const size_t elements,
  Ts&&...  rest
){
  dicomNodeError_t ret = check_buffer_pointers(buffer, elements);
  if(ret != dicomNodeError_t::SUCCESS){
    return ret;
  }

  ret = check_buffer_pointers(std::forward<Ts>(rest)...);
  return ret;
}

dicomNodeError_t load_space(Space<3>* space, const pybind11::object& python_space);

//# REGION FREEING

/**
 *
 * @tparam T
 * @tparam DIMENSION
 * @param out_volume
 * @return
 */
template<typename T, u8 DIMENSION>
dicomNodeError_t free_host_volume(Volume<DIMENSION, T>* out_volume) {
  const dicomNodeError_t error = encode_cuda_error(cudaFree(out_volume->data));
  out_volume->data = nullptr;

  return error;
}

template<typename T, u8 DIMENSION>
dicomNodeError_t free_device_volume(Volume<DIMENSION, T>* out_volume) {
  Volume<DIMENSION, T> volume;

  DicomNodeRunner runner;

  runner | [&]() {
    return cudaMemcpy(&volume, out_volume, sizeof(Volume<DIMENSION, T>), cudaMemcpyDefault);
  } | [&]() {
    return cudaFree(volume.data);
  } | [&]() {
    return cudaMemset(out_volume, 0, sizeof(Volume<DIMENSION, T>));
  };

  return runner.error();
}
template<typename T, u8 DIMENSION>
dicomNodeError_t free_volume(Volume<DIMENSION, T>* out_volume) {
  return pointer_location_dependent_code<
    free_host_volume<T, DIMENSION>,
    free_device_volume<T, DIMENSION>
    >(out_volume);
}

template<typename T>
dicomNodeError_t free_device_image(Image<3, T>* dev_out_image){
  T* data_pointer = nullptr;
  DicomNodeRunner runner;

  // Note that if this shit fucks up, we
  runner
    | [&](){
      return cudaMemcpy(
        &data_pointer,
        &(dev_out_image->volume.data),
        sizeof(T*),
        cudaMemcpyDefault
      );
  } | [&](){
      return cudaFree(data_pointer);
  };
  return runner.error();
}

template<typename T>
__host__ dicomNodeError_t free_host_image(Image<3, T>* host_image){
  return free_host_volume(&host_image->volume);
}

/**
 * @brief Frees an image, not that this functions assumes that the gpu driver
 * is working
 *
 * @tparam T - The type
 * @param dev_out_image
 */
template<typename T>
dicomNodeError_t free_image(Image<3, T>* out_image){
  DicomNodeRunner runner{[](dicomNodeError_t error){
    std::cout << "Free Image encountered error: " << error_to_human_readable(error) << "\n";
  }};
  cudaPointerAttributes out_image_attr;

  runner | [&](){
    return cudaPointerGetAttributes(&out_image_attr, out_image);
  } | [&](){
    dicomNodeError_t error;

    if(is_host_pointer(out_image_attr)){
      error = free_host_image(out_image);
    } else {
      error = free_device_image(out_image);
    }
    return error;
  };

  return runner.error();
}


//# Region LOADING

template<u8 DIMENSION>
dicomNodeError_t load_extent(
    Extent<3>& out_extent,
    const pybind11::object& python_extent
  ) {

  pybind11::array_t<u32, pybind11::array::c_style | pybind11::array::forcecast> arr =
    pybind11::array_t<u32, pybind11::array::c_style | pybind11::array::forcecast>::ensure(python_extent);

  if (!arr) {
    return dicomNodeError_t::UNABLE_TO_ACQUIRE_BUFFER;
  }
  if (arr.ndim() != 1 || arr.shape(0) != DIMENSION) {
    return dicomNodeError_t::INPUT_SIZE_MISMATCH;
  }

  const u32* data = arr.data();
  for (u8 i = 0; i < DIMENSION; ++i) {
    out_extent.sizes[i] = data[i];
  }

  return SUCCESS;
}

template<typename T, u8 DIMENSION>
dicomNodeError_t load_host_volume(Volume<DIMENSION, T>* out_volume, const pybind11::array_t<T>& python_array) {
  if(out_volume == nullptr) {
    return dicomNodeError_t::ARG_IS_NULL_POINTER;
  }
  Volume<DIMENSION, T>& volume = *out_volume;

  if (volume.data != nullptr) {
    return dicomNodeError_t::ALREADY_ALLOCATED_OBJECT;
  }

  // this might throw
  const pybind11::buffer_info array_buffer = python_array.request();
  if (array_buffer.ptr == nullptr) {
    return dicomNodeError_t::UNABLE_TO_ACQUIRE_BUFFER;
  }

  DicomNodeRunner runner{
    [&]([[maybe_unused]] const dicomNodeError_t& error) noexcept  {
      free_volume(out_volume);
    }
  };
  volume.set_extent(array_buffer.shape);

  runner | [&]() noexcept {
    return cudaMalloc(&volume.data, array_buffer.size * sizeof(T));
  } | [&]() noexcept {
    return cudaMemcpy(volume.data, array_buffer.ptr, array_buffer.size * sizeof(T), cudaMemcpyDefault);
  };

  return runner.error();
}

template<typename T, u8 DIMENSION>
dicomNodeError_t load_device_volume(Volume<DIMENSION, T>* out_volume, const pybind11::array_t<T>& python_array) {
  Volume<DIMENSION, T> volume;
  DicomNodeRunner runner;
  runner | [&]() noexcept {
    return load_host_volume(&volume, python_array);
  } | [&]() noexcept {
    return cudaMemcpy(out_volume, &volume, sizeof(Volume<DIMENSION, T>), cudaMemcpyDefault);
  };

  return runner.error();
}

template<typename T, u8 DIMENSION>
dicomNodeError_t load_volume(Volume<DIMENSION, T>* out_volume, const pybind11::array_t<T>& python_array) {
  return pointer_location_dependent_code<
      load_host_volume<T,DIMENSION>,
      load_device_volume<T,DIMENSION>,
      Volume<DIMENSION, T>
    >(out_volume, python_array);
}


template<typename T>
dicomNodeError_t load_host_image(
    Image<3, T>* host_out_image,
    const pybind11::object& python_image
  ){
  const python_array<T>& raw_image = python_image.attr("raw").cast<python_array<T>>();
  const pybind11::object& space = python_image.attr("space");

  DicomNodeRunner runner{
    [&]([[maybe_unused]] const dicomNodeError_t& error) noexcept {
      std::cout << "Load into host image encountered: " << error_to_human_readable(error) << "\n";
    }
  };

  runner
    | [&]() noexcept {
    return load_into_host_space(&host_out_image->space, space);
  } | [&]() noexcept {
    return load_host_volume<T, 3>(&host_out_image->volume, raw_image);
  };

  return runner.error();
}


template<typename T>
dicomNodeError_t load_device_image(
  Image<3, T>* dev_out_image,
  const pybind11::object& python_image
){
  Image<3, T> host_image;
  DicomNodeRunner runner;

  runner | [&]() noexcept {
    return load_host_image(&host_image, python_image);
  } | [&]() noexcept {
    return cudaMemcpy(dev_out_image, &host_image, sizeof(Image<3, T>), cudaMemcpyDefault);
  };

  return runner.error();
}

/**
 * @brief Converts an Image from it's python representation to an equivalent C++ object
 * @tparam T type of the image.
 * @param image pointer to image where we wish to store the c++ object. Assumed not to be null.
 * Pointer can be Device or host memory
 * @param python_image the python object to be copied from.
 */
template<typename T>
[[nodiscard("Forgot to check error value on image loading")]]
dicomNodeError_t load_image(
  Image<3, T>* image,
  const pybind11::object& python_image
){
  DicomNodeRunner runner{
    [](const dicomNodeError_t& error){
      std::cout << "Load Image encountered error: " << (uint32_t)error << "\n";
    }
  };

  return pointer_location_dependent_code<
    load_host_image<T>,
    load_device_image<T>
  >(runner, image, python_image);
}

/**
 * @brief Returns the amount of bytes spaned by an image object or a space for
 * an image object
 *
 * @tparam T
 * @param python_object
 * @return size_t
 */
template<typename T>
size_t get_image_size(const pybind11::object& python_object){
  const pybind11::module_& space_module = pybind11::module_::import("dicomnode.math.space");
  const pybind11::object& space_class = space_module.attr("Space");

  if(pybind11::isinstance(python_object, space_class)){
    using EXTENT_DATA_TYPE = uint32_t;
    const python_array<EXTENT_DATA_TYPE>& python_extent = python_object.attr("extent").cast<python_array<int>>();
    const pybind11::buffer_info& python_extent_buffer = python_extent.request();
    if (python_extent_buffer.ptr == nullptr) {
      return 0;
    }
    const auto * const data = (EXTENT_DATA_TYPE*)python_extent_buffer.ptr;

    size_t size = sizeof(T);
    for(int i = 0; i < python_extent_buffer.size; i++){
      size *= data[i];
    }

    return size;
  }

  const pybind11::module_& image_module = pybind11::module_::import("dicomnode.math.image");
  const pybind11::object& image_class = image_module.attr("Image");

  if(pybind11::isinstance(python_object, image_class)){
    const python_array<T>& raw_image = python_object.attr("raw").cast<python_array<T>>();
    return raw_image.size() * sizeof(T);
  }

  return 0;
}

template<typename T>
dicomNodeError_t get_image_pointer(
  const pybind11::object& image,
  T** out
){
  DicomNodeRunner runner{
    [](const dicomNodeError_t& error) noexcept {
      std::cout << "Get Image pointer encountered error: " << (uint32_t)error << "\n";
    }
  };
  runner
    | [&]() noexcept {
      return is_instance(image, "dicomnode.math.image", "Image");
    } | [&]() noexcept {
      const python_array<T>& raw_image = image.attr("raw").cast<python_array<T>>();
      const pybind11::buffer_info& buffer = raw_image.request();

      dicomNodeError_t error = check_buffer_pointers(
        std::cref(buffer), buffer.size
      );

      if(!error){
        *out = (T*)buffer.ptr;
      }

      return error;
    };

  return runner.error();
}

template<typename T>
dicomNodeError_t load_texture_from_python_image(
  Texture<3, T>* texture,
  const pybind11::object& python_image
){
  if(texture == nullptr){
    return ARG_IS_NULL_POINTER;
  }

  DicomNodeRunner runner{
    [](const dicomNodeError_t& error){
      std::cout << "load_python_texture encountered error: " << (uint32_t)error << "\n";
    }
  };

  Image<3, T> image;

  runner
    | [&]() noexcept {
      return is_instance(python_image, "dicomnode.math.image", "Image");
    } | [&]() noexcept {
      return load_image(&image, python_image);
    } | [&]() noexcept {
      return load_texture(texture, image);
    };
  return runner.error();
}