#include<gtest/gtest.h>

#include"../gpu_code/dicom_node_gpu.cuh"

#include<cuda/std/array>

constexpr u32 FULL_MASK = 0xFFFFFFFF;

namespace UTILITIES_TEST {


  struct ShuffleStruct1 {
    f32 f1{};
    f32 f2{};
    f32 f3{};
  };

  struct __align__(16) ShuffleStruct2 {
    f32 f1{};
    f32 f2{};
    f32 f3{};
  };

  using ShuffleArray1 = cuda::std::array<ShuffleStruct1, 1024>;
  using ShuffleArray2 = cuda::std::array<ShuffleStruct2, 1024>;

  __global__ void init_shuffle_struct_1(ShuffleArray1* t){
    f32 init_start_value = static_cast<f32>(threadIdx.x) * 3.0f;

    ShuffleStruct1& obj = (*t)[threadIdx.x];

    obj.f1 = init_start_value + 1.0f;
    obj.f2 = init_start_value + 2.0f;
    obj.f3 = init_start_value + 3.0f;
  }

  __global__ void init_shuffle_struct_1_2(ShuffleStruct1* t){
    f32 init_start_value = static_cast<f32>(threadIdx.x) * 3.0f;

    ShuffleStruct1& obj = t[threadIdx.x];

    obj.f1 = init_start_value + 1.0f;
    obj.f2 = init_start_value + 2.0f;
    obj.f3 = init_start_value + 3.0f;
  }

  __global__ void init_shuffle_struct_2(ShuffleArray2* t){
    f32 init_start_value = static_cast<f32>(threadIdx.x) * 3.0f;

    ShuffleStruct2& obj = (*t)[threadIdx.x];

    obj.f1 = init_start_value + 1.0f;
    obj.f2 = init_start_value + 2.0f;
    obj.f3 = init_start_value + 3.0f;
  }

  __global__ void shuffle_2(ShuffleStruct1* t_in, ShuffleStruct1* t_out){
    ShuffleStruct1& in = t_in[threadIdx.x];
    ShuffleStruct1& out = t_out[threadIdx.x];

    out.f1 = __shfl_up_sync(FULL_MASK, in.f1, 1);
    out.f2 = __shfl_up_sync(FULL_MASK, in.f2, 1);
    out.f3 = __shfl_up_sync(FULL_MASK, in.f3, 1);
  }



};



TEST(SHUFFLE_TESTS, IT_SHOULD_WORK){
  using ShuffleArray = UTILITIES_TEST::ShuffleArray1;

  UTILITIES_TEST::ShuffleStruct1* dev_in = nullptr;
  UTILITIES_TEST::ShuffleStruct1* dev_out = nullptr;

  ShuffleArray host_out;

  dicomNodeError_t error = SUCCESS;

  DicomNodeRunner runner([&](dicomNodeError_t err){
    error = err;
  });

  runner
    | [&](){ return cudaMalloc(&dev_in, sizeof(UTILITIES_TEST::ShuffleStruct1) * 1024); }
    | [&](){ return cudaMalloc(&dev_out, sizeof(UTILITIES_TEST::ShuffleStruct1) * 1024); }
    | [&](){
      UTILITIES_TEST::init_shuffle_struct_1_2<<<1, 1024>>>(dev_in);
      return cudaGetLastError();
    } | [&](){
      UTILITIES_TEST::shuffle_2<<<1,1024>>>(dev_in, dev_out);
      return cudaGetLastError();
    } | [&](){ return cudaMemcpy(host_out.data(), dev_out, sizeof(UTILITIES_TEST::ShuffleStruct1) * 1024, cudaMemcpyDefault );}
    | [&](){
      free_device_memory(&dev_in, &dev_out);
      return cudaDeviceSynchronize();
    };

  if(runner.error()){
    cudaError_t cuda_error = extract_cuda_error(runner.error());

    const char* error_name = cudaGetErrorName(cuda_error);
    const char* error_desc = cudaGetErrorString(cuda_error);

    std::cout << error_name << " " << error_desc << "\n";
  }


  EXPECT_EQ(runner.error(), dicomNodeError_t::SUCCESS);

  for(u16 i = 0; i < 16; i++){
    UTILITIES_TEST::ShuffleStruct1& ref = host_out[i];

    std::cout << "i: " << i << "\n";
    std::cout << "f1: " << ref.f1 << "\n";
    std::cout << "f2: " << ref.f2 << "\n";
    std::cout << "f3: " << ref.f3 << "\n";
  }
}