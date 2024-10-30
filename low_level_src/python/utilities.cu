#include"utilities.cuh"

cudaError_t extract_cuda_error(dicomNodeError_t error){
  return (cudaError_t)(error & 0x7FFFFFFF);
}

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



}
