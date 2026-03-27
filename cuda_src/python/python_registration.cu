#include"python_registration.cuh"

#include<iostream>
#include<tuple>

#include"../gpu_code/dicom_node_gpu.cuh"
#include"utilities.cuh"



std::tuple<dicomNodeError_t, python_array<f32>> registration(
  pybind11::object& p_destination_image,
  pybind11::object& p_source_image
){

  python_array<f32> out;

  Image<3, f32> source_image;
  Image<3, f32> destination_image;

  DicomNodeRunner runner([&](dicomNodeError_t error){
    std::cout << "While performing a registration the function encountered error code:" << (u32)error << "\n" ;

    free_image(source_image);
    free_image(destination_image);
  });

  runner
      | [&](){
      return is_instance(p_destination_image, "dicomnode.math.image", "Image");
    } | [&](){
      return is_instance(p_source_image, "dicomnode.math.image", "Image");
    } | [&](){
      return load_image<f32>(source_image, source_image);
    } | [&]() {
      return load_image<f32>(destination_image, destination_image);
    } | [&](){
      return REGISTRATION::register_to<f32>(source_image, destination_image);
    } | [&](){
      return free_image(source_image);
    } | [&](){
      return free_image(destination_image);
    };

  return {runner.error(), out};
}


void apply_registration_module(pybind11::module& m){
  const char* help_message = R"(
    This function registers a destination image to a source image
  )";

  pybind11::module registration_submodule = m.def_submodule(
    "registration"
  );

  registration_submodule.def("register", registration, help_message);
}