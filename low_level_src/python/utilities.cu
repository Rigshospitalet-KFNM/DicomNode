#include"utilities.cuh"

std::string get_byte_string (size_t bytes){
  std::stringstream ss;
  if(bytes < 1024){
    ss << bytes << " B";
    return ss.str();
  }
  bytes >>= 10;
  if(bytes < 1024){
    ss << bytes << " kB";
    return ss.str();
  }
  bytes >>= 10;
  if(bytes < 1024){
    ss << bytes << " MB";
    return ss.str();
  }

  bytes >>= 10;
  ss << bytes << " GB";
  return ss.str();
}

template<typename T>
dicomNodeError_t load_image(
  Image<3, T>* dev_out_image,
  const pybind11::object& python_image
){
  const pybind11::module_& image_module = pybind11::module_::import("dicomnode.math.image");
  const pybind11::object& image_class = image_module.attr('Image');

  if(!pybind11::isinstance(python_image, image_class)){
    return dicomNodeError_t::INPUT_TYPE_ERROR;
  }

  const pybind11::array_t<T>& raw_image = image.attr("raw");
  const pybind11::buffer_info& image_buffer = raw_image.request(false);
  const pybind11::object& space = image.attr("space");
  const pybind11::array_t<float>& starting_point = space.attr("starting_point");

  Image<3, T> host_image;

  size_t image_size = sizeof(T);
  for (int i = 0; const ssize_t dim : image_buffer.shape){
    if(dim <= 0){
      return dicomNodeError_t::NON_POSITIVE_SHAPE;
    }
    host_image.domain[i] = dim;
    image_size *= dim;
    i++;
  }

  DicomNodeRunner runner{[&](){free_device_memory(host_image.data);}};

  runner
    | [&](){ return cudaMalloc(&host_image.data, image_size); }
    | [&](){
      return cudaMemcpy(dev_out_image, &host_image, sizeof(Image<3, T>));
    };


  return runner.error();
}

template<typename T>
void free_image(Image<3, T>* dev_out_image){
  T* data_pointer = nullptr;
  cudaMemcpy(data_pointer, &(dev_out_image->data), sizeof(T*), cudaMemcpyDefault);

  // Note that
  cudaFree(data_pointer);
  cudaFree(dev_out_image);
}
