#pragma once

#include <gtest/gtest.h>

#include "test_data.cuh"
#include"../gpu_code/dicom_node_gpu.cuh"

namespace REDUCE {
  constexpr std::array<f32, 3 * 3 * 3> image_1_data = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f,

    10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f,
    16.0f, 17.0f, 18.0f,

    19.0f, 20.0f, 21.0f,
    22.0f, 23.0f, 24.0f,
    25.0f, 26.0f, 27.0f,
  };

  constexpr std::array<f32, 3 * 3 * 3> image_2_data = {
    2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f,
    8.0f, 9.0f, 10.0f,

    11.0f, 12.0f, 13.0f,
    14.0f, 15.0f, 16.0f,
    17.0f, 18.0f, 19.0f,

    20.0f, 21.0f, 22.0f,
    23.0f, 24.0f, 25.0f,
    26.0f, 27.0f, 28.0f,
  };

constexpr Extent<3> extent{3,3,3};


TEST(REDUCE, IMAGE_REDUCTION) {
  Image<3,f32> image_1{
    TEST_DATA::make_spaces(extent),
    Volume<3,f32>{
      .data= nullptr,
      .m_extent = extent
    }
    };

  Image<3,f32> image_2{
    TEST_DATA::make_spaces(extent),
    Volume<3,f32>{
      .data=nullptr,
      .m_extent = extent
    }
  };

  cudaMalloc(&image_1.data(), image_1.size());
  cudaMalloc(&image_2.data(), image_2.size());
  cudaMemcpy(image_1.data(), image_1_data.data(), image_1.size(), cudaMemcpyDefault);
  cudaMemcpy(image_2.data(), image_2_data.data(), image_2.size(), cudaMemcpyDefault);

  Image<3,f32>* device_image_1 = nullptr;
  Image<3,f32>* device_image_2 = nullptr;

  f32 difference = 0.0f;


  cudaMalloc(&device_image_1, sizeof(Image<3,f32>));
  cudaMalloc(&device_image_2, sizeof(Image<3,f32>));

  cudaMemcpy(device_image_1, &image_1, sizeof(Image<3,f32>), cudaMemcpyDefault);
  cudaMemcpy(device_image_2, &image_2, sizeof(Image<3,f32>), cudaMemcpyDefault);

  reduce_no_mem<1, REGISTRATION::IMAGE_DIFFERENCE<f32>, f32>(
    image_1.elements(), &difference, device_image_1, device_image_2
  );

  EXPECT_FLOAT_EQ(27.0f, difference);

  cudaFree(image_1.data());
  cudaFree(image_2.data());
  cudaFree(device_image_1);
  cudaFree(device_image_2);

}
template<typename T>
__global__ void image_registration_maps_to_kernel(T* output, Image<3, T>* image_1, Image<3, T>* image_2) {
  if (threadIdx.x < image_1->elements()) {
    output[threadIdx.x] = REGISTRATION::IMAGE_DIFFERENCE<f32>::map_to(threadIdx.x, image_1, image_2);
  }
}

TEST(REDUCE, COPY_INTO_SHARED_WORKS) {
  Image<3,f32> image_1{
    TEST_DATA::make_spaces(extent),
    Volume<3,f32>{
      .data= nullptr,
      .m_extent = extent
    }
  };

  Image<3,f32> image_2{
    TEST_DATA::make_spaces(extent),
    Volume<3,f32>{
      .data=nullptr,
      .m_extent = extent
    }
  };


  cudaMalloc(&image_1.data(), image_1.size());
  cudaMalloc(&image_2.data(), image_2.size());
  cudaMemcpy(image_1.data(), image_1_data.data(), image_1.size(), cudaMemcpyDefault);
  cudaMemcpy(image_2.data(), image_2_data.data(), image_2.size(), cudaMemcpyDefault);

  Image<3,f32>* device_image_1 = nullptr;
  Image<3,f32>* device_image_2 = nullptr;

  f32* device_output = nullptr;
  cudaMalloc(&device_output, sizeof(f32) * image_1.elements());

  cudaMalloc(&device_image_1, sizeof(Image<3,f32>));
  cudaMalloc(&device_image_2, sizeof(Image<3,f32>));

  CUDA_CHECK(cudaMemcpy(device_image_1, &image_1, sizeof(Image<3,f32>), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(device_image_2, &image_2, sizeof(Image<3,f32>), cudaMemcpyDefault));

  image_registration_maps_to_kernel<<<1, 1024>>>(device_output, device_image_1, device_image_2);
  CUDA_CHECK(cudaGetLastError());

  std::array<f32, extent.elements()> output;
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, sizeof(f32) * image_1.elements(), cudaMemcpyDefault));

  for (const f32& o : output) {
    EXPECT_FLOAT_EQ(o, 1.0f);
  }

  cudaFree(device_output);
  cudaFree(image_1.data());
  cudaFree(image_2.data());
  cudaFree(device_image_1);
  cudaFree(device_image_2);

}


}
