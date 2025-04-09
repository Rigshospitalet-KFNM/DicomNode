#include"../gpu_code/dicom_node_gpu.cuh"

#include"utilities.cuh"

#include"python_labeling.cuh"


namespace {
  template<typename T>
  std::tuple<dicomNodeError_t, python_array<T>> templated_slice_based_ccl(
    pybind11::object& python_image
  ){
    //const python_array<T> raw_image = python_image.attr("raw");
    Image<3, T> image;
    python_array<T> return_array = {};

    DicomNodeRunner runner{
      [&](dicomNodeError_t error){ free_image(&image); }
    };
    runner | [&](){
      return load_image(&image, python_image);
    } | [&](){
      return SUCCESS;
    }
    | [&](){
      return free_image(&image);
    };

    return {runner.error(), std::move(return_array)};
  }


  std::tuple<dicomNodeError_t, pybind11::array> slice_based_ccl(
    pybind11::object& image
  ){
  if(!is_instance(image, "dicomnode.math.image", "Image")){
    const std::string error_message("Error: Sliced Based component labeling takes a dicomnode.math.image.Image object as argument");
    throw std::runtime_error(error_message);
  }

  const pybind11::array& raw_image = image.attr("raw");
  const std::string dtype = pybind11::str(raw_image.attr("dtype"));

  //Switch statement doesn't work because I am comparing strings
  if(dtype == "float32"){
    return templated_slice_based_ccl<float>(image);
  } else if (dtype == "uint8") {
    return templated_slice_based_ccl<uint8_t>(image);
  } if (dtype == "uint16") {
    return templated_slice_based_ccl<uint16_t>(image);
  } if (dtype == "uint32") {
    return templated_slice_based_ccl<uint32_t>(image);
  } else if (dtype == "int8") {
    return templated_slice_based_ccl<int8_t>(image);
  } if (dtype == "int16") {
    return templated_slice_based_ccl<int16_t>(image);
  } if (dtype == "int32") {
    return templated_slice_based_ccl<int32_t>(image);
  }

  const std::string error_message = "Unsupported dtype:" + dtype;
  throw std::runtime_error(error_message);
}
}; // End of Anon namespace


void apply_labeling_module(pybind11::module& m){
  pybind11::module sub_module = m.def_submodule(
    "labeling",
    "GPU module for performing connected component labeling"
  );

  sub_module.def(
    "slice_based",
    &slice_based_ccl,
    "Performs Slice based Connected Component labeling on the GPU for each\
 slice of the time\n\n\
  Args:\n\
      (dicomnode.math.image.Image) - An image with an underlying type of:\n\
 * float\n * uint8_t\n * uint16_t\n * uint32_t\n, * int8_t\n * int16_t\n\
 * int32_t\n"
  );
}