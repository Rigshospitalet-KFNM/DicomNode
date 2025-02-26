#include"../gpu_code/dicom_node_gpu.cuh"

#include"utilities.cuh"

#include"python_labeling.cuh"




namespace {
  template<typename T>
  std::tuple<dicomNodeError_t, pybind11::array_t<T>> templated_slice_based_ccl(
    pybind11::object& python_image
  ){
    //const python_array<T> raw_image = python_image.attr("raw");
    Image<3, T> image;

    DicomNodeRunner runner;
    runner | [&](){
      return SUCCESS;
      //return load_image(&image, python_image);
    } | [&](){
      //return free_image(&image);
      return SUCCESS;
    };

    return {runner.error(), {}};
  }


  std::tuple<dicomNodeError_t, pybind11::array> slice_based_ccl(
    pybind11::object& image
  ){
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

  sub_module.def("slice_based", &slice_based_ccl);
}