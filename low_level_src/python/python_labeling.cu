#include"../gpu_code/dicom_node_gpu.cuh"

#include"utilities.cuh"

#include"python_labeling.cuh"


namespace {
  template<typename T>
  std::tuple<dicomNodeError_t, python_array<uint32_t>> templated_slice_based_ccl(
    pybind11::object& python_image
  ){
    //const python_array<T> raw_image = python_image.attr("raw");
    Image<3, T> image;
    uint32_t* device_out_labels = nullptr;
    python_array<uint32_t> return_array = {};

    DicomNodeRunner runner{
      [&](dicomNodeError_t error){
        std::cout << "Sliced based CCL encountered: " << error_to_human_readable(error) << "\n"
                   << "device_out_labels: " << device_out_labels << "\n"
                   << "Image volume Data: " << image.volume.data << "\n"
                   << "label size " << sizeof(T) * image.elements() << "\n";

        free_image(&image);
        free_device_memory(&device_out_labels);
      }
    };
    runner | [&](){
      return load_image(&image, python_image);
    } | [&](){
      const size_t label_size = sizeof(T) * image.elements();
      return cudaMalloc(&device_out_labels, label_size);
    } | [&](){
      return slicedConnectedComponentLabeling<T>(
        device_out_labels, image.volume
      );
    } | [&](){
      const size_t label_size = sizeof(T) * image.elements();
      return_array = python_array<uint32_t>(
        {image.num_slices(), image.num_rows(), image.num_cols()},
        {image.num_rows() * image.num_cols() * sizeof(T) ,image.num_cols() * sizeof(T), sizeof(T)}
      );

      pybind11::buffer_info return_buffer = return_array.request(true);
      return cudaMemcpy(return_buffer.ptr, device_out_labels, label_size, cudaMemcpyDefault);
    } | [&](){
      free_device_memory(&device_out_labels);
      return free_image(&image);
    };

    return {runner.error(), std::move(return_array)};
  }


  std::tuple<dicomNodeError_t, pybind11::array> slice_based_ccl(
    pybind11::object& image
  ){
  if(is_instance(image, "dicomnode.math.image", "Image") != dicomNodeError_t::SUCCESS){
    const std::string error_message("Error: Sliced Based component labeling tak"
      "es a dicomnode.math.image.Image object as argument\n");

    throw std::runtime_error(error_message);
  }

  const pybind11::array& raw_image = image.attr("raw");
  const pybind11::dtype image_dtype = raw_image.dtype();

  //Switch statement doesn't work because I am comparing strings
  if(image_dtype.equal(pybind11::dtype::of<float>())){
    return templated_slice_based_ccl<float>(image);
  } else if (image_dtype.equal(pybind11::dtype::of<uint8_t>())) {
    return templated_slice_based_ccl<uint8_t>(image);
  } if (image_dtype.equal(pybind11::dtype::of<uint16_t>())) {
    return templated_slice_based_ccl<uint16_t>(image);
  } if (image_dtype.equal(pybind11::dtype::of<uint32_t>())) {
    return templated_slice_based_ccl<uint32_t>(image);
  } else if (image_dtype.equal(pybind11::dtype::of<int8_t>())) {
    return templated_slice_based_ccl<int8_t>(image);
  } if (image_dtype.equal(pybind11::dtype::of<int16_t>())) {
    return templated_slice_based_ccl<int16_t>(image);
  } if (image_dtype.equal(pybind11::dtype::of<int32_t>())) {
    return templated_slice_based_ccl<int32_t>(image);
  }

  const std::string data_type_name = pybind11::str(raw_image.attr("dtype"));
  const std::string error_message = "Unsupported dtype:" + data_type_name;
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