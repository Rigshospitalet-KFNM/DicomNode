#pragma once
#include<functional>
#include<string>
#include<cstring>
#include<sstream>
#include<tuple>
#include<vector>
#include<iostream>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include"../gpu_code/dicom_node_gpu.cuh"

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

dicomNodeError_t _load_into_host_space(
  Space<3>* host_space,
  const pybind11::object& space
);

dicomNodeError_t check_buffer_pointers(
  const pybind11::buffer_info& buffer, const size_t elements
);

dicomNodeError_t _load_into_device_space(
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

bool is_host_pointer(const cudaPointerAttributes& attr);

dicomNodeError_t load_space(Space<3>* space, const pybind11::object& python_space);


// Anon namespace for templates
namespace {
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
    } | [&](){
        return cudaFree(dev_out_image);
    };
    return runner.error();
  }

  template<typename T>
  __host__ dicomNodeError_t free_host_image(Image<3, T>* host_out_image){
    const dicomNodeError_t error = encode_cuda_error(
      cudaFree(host_out_image->volume.data)
    );

    host_out_image->volume.data=nullptr;

    return error;
  }

  template<typename T>
  dicomNodeError_t _load_into_host_image(
    Image<3, T>* host_out_image,
    const pybind11::object& python_image
  ){
    const python_array<T>& raw_image = python_image.attr("raw").cast<python_array<T>>();
    const pybind11::buffer_info& image_buffer = raw_image.request();
    const pybind11::object& space = python_image.attr("space");
    const ssize_t data_size = image_buffer.size * sizeof(T);

    DicomNodeRunner runner{
      [&](dicomNodeError_t error){
        std::cout << "Load into host image encountered: " << error_to_human_readable(error) << "\n";

        free_device_memory(&(host_out_image->volume.data));
      }
    };

    runner
      | [&](){
      // This function checks if the data is valid to copy
      if(host_out_image->volume.data != nullptr){
        return dicomNodeError_t::ALREADY_ALLOCATED_OBJECT;
      }

      // This check should prevent the automatic typecasting to
      if(image_buffer.size <= SSIZE_T_ERROR_VALUE){
        return dicomNodeError_t::UNABLE_TO_ACQUIRE_BUFFER;
      }

      return host_out_image->volume.set_extent(image_buffer.shape);
    } | [&](){
      return _load_into_host_space(&host_out_image->space, space);
    } | [&](){
      return cudaMalloc(
        &(host_out_image->volume.data),
        data_size
      );
    } | [&](){
      return cudaMemcpy(
        host_out_image->volume.data,
        image_buffer.ptr,
        data_size,
        cudaMemcpyDefault
      );
    };

    return runner.error();
  }


  template<typename T>
  dicomNodeError_t _load_into_dev_image(
    Image<3, T>* dev_out_image,
    const pybind11::object& python_image
  ){
    size_t image_size = sizeof(T);
    const python_array<T>& raw_image = python_image.attr("raw").cast<python_array<T>>();
    const pybind11::buffer_info& image_buffer = raw_image.request();
    const pybind11::object& space = python_image.attr("space");

    DicomNodeRunner runner{[&](dicomNodeError_t error){
      std::cout << "load into device image encountered error:" << error_to_human_readable(error) << "\n";

      free_device_memory(&dev_out_image->volume.data);}
    };

    runner
      | [&](){
        return _load_into_device_space(&(dev_out_image->space), space);
      } | [&](){
        for (const ssize_t& dim : image_buffer.shape){
          if(dim <= 0){
            return dicomNodeError_t::NON_POSITIVE_SHAPE;
          }
          image_size *= dim;
        }

        return encode_cuda_error(cudaMemcpy(
          &(dev_out_image->volume.m_extent),
          &(dev_out_image->space.extent),
          sizeof(Extent<3>),
          cudaMemcpyDefault
        ));
      } | [&](){
       return cudaMalloc(&(dev_out_image->volume.data), image_size);
      } | [&](){
        return cudaMemcpy(dev_out_image->volume.data, image_buffer.ptr, image_size, cudaMemcpyDefault);
      };

    return runner.error();
  }
}

/**
 * @brief This function extracts the underlying cudaError_t that was triggered.
 * Note that this function returns an correct value if it wasn't triggered by a
 * cudaError_t
 * @param error a dicomNodeError_t that was triggered by an none sucess
 * cudaError_t
 * @return cudaError_t
 */
template<typename T>
[[nodiscard("Forgot to check error value on image loading")]]
dicomNodeError_t load_image(
  Image<3, T>* image,
  const pybind11::object& python_image
){
  cudaPointerAttributes attr;
  DicomNodeRunner runner{
    [](const dicomNodeError_t& error){
      std::cout << "Load Image encountered error: " << (uint32_t)error << "\n";
    }
  };

  runner
    | [&](){
    return is_instance(python_image, "dicomnode.math.image", "Image");
  } | [&](){
    return cudaPointerGetAttributes(&attr, image);
  } | [&](){
    if(is_host_pointer(attr)){
      return _load_into_host_image(image, python_image);
    } else {
      return _load_into_dev_image(image, python_image);
    }
  };

  return runner.error();
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
    const python_array<int>& python_extent = python_object.attr("extent").cast<python_array<int>>();
    const pybind11::buffer_info& python_extent_buffer = python_extent.request();
    if (python_extent_buffer.ptr == nullptr) {
      return 0;
    }
    int* data = (int*)python_extent_buffer.ptr;

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
    [](const dicomNodeError_t& error){
      std::cout << "Get Image pointer encountered error: " << (uint32_t)error << "\n";
    }
  };
  runner
    | [&](){
      return is_instance(image, "dicomnode.math.image", "Image");
    } | [&](){
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
  DicomNodeRunner runner{
    [](const dicomNodeError_t& error){
      std::cout << "load_python_texture encountered error: " << (uint32_t)error << "\n";
    }
  };

  T* data = nullptr;
  Space<3> space;

  runner
    | [&](){
      return is_instance(python_image, "dicomnode.math.image", "Image");
    } | [&](){
      const pybind11::object& python_space = python_image.attr("space");
      return load_space(&space, python_space);
    } | [&](){
      return get_image_pointer(python_image, &data);
    } | [&](){
      return load_texture(texture, data, space);
    };
  return runner.error();
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
