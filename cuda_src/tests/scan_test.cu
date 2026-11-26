#include <gtest/gtest.h>
#include <array>
#include <memory>
#include <random>

#include "test_data.cuh"
#include"../gpu_code/dicom_node_gpu.cuh"


namespace SCAN {
  template<typename T>
  struct ADD {
    static __host__ __device__ T identity() noexcept {
      return 0;
    }

    static __host__ __device__ bool equals(const T lhs, const T rhs) {
      return lhs == rhs;
    }

    static __host__ __device__ T map_to(const T in, [[maybe_unused]] u64 flat_index) {
      return in;
    }

    static __host__ __device__ T apply(const T a, const T b) {
      return a + b;
    }

    static __host__ __device__ T remove_volatile(volatile T& x) {
      T xx = x;
      return xx;
    }
  };

  TEST(SCAN, STANDARD_SCAN_U32) {
    using TestType = u32;
    constexpr auto INPUT_RAND_MAX = 1 << 16;
    constexpr auto DATA_SIZE = 1 << 16;

    std::random_device dev{};
    std::mt19937 rng{dev()};
    std::uniform_int_distribution<TestType> distribution(0, INPUT_RAND_MAX);

    const auto host_input_data = std::make_unique<std::array<TestType, DATA_SIZE>>();

    for (auto& i : *host_input_data) {
      i = distribution(rng);
    }

    const auto host_expected = std::make_unique<std::array<TestType, DATA_SIZE>>();

    TestType accum = 0;
    for (u64 idx = 0; idx < DATA_SIZE; idx++) {
      accum += (*host_input_data)[idx];
      (*host_expected)[idx] = accum;
    }

    const auto host_output = std::make_unique<std::array<TestType, DATA_SIZE>>();

    dicomNodeError_t error = scan<8, ADD<TestType>, TestType, TestType> (host_input_data->data(), DATA_SIZE, host_output->data());
    ASSERT_EQ(error, dicomNodeError_t::SUCCESS);

    for (u64 idx = 0; idx < DATA_SIZE; idx++) {
      ASSERT_EQ((*host_output)[idx], (*host_expected)[idx]);
    }
  }

  TEST(SCAN, STANDARD_SCAN_F64) {
    // This test doesn't really work for f32 because addition fucks up.
    using TestType = f64;
    constexpr auto INPUT_RAND_MAX = 1 << 4;
    constexpr auto DATA_SIZE = 1 << 16;

    std::random_device dev{};
    std::mt19937 rng{dev()};
    std::uniform_real_distribution<TestType> distribution(0, INPUT_RAND_MAX);

    const auto host_input_data = std::make_unique<std::array<TestType, DATA_SIZE>>();

    for (auto& i : *host_input_data) {
      i = distribution(rng);
    }

    const auto host_expected = std::make_unique<std::array<TestType, DATA_SIZE>>();

    TestType accum = 0;
    for (u64 idx = 0; idx < DATA_SIZE; idx++) {
      accum += (*host_input_data)[idx];
      (*host_expected)[idx] = accum;
    }

    const auto host_output = std::make_unique<std::array<TestType, DATA_SIZE>>();

    const dicomNodeError_t error = scan<4, ADD<TestType>, TestType, TestType> (host_input_data->data(), DATA_SIZE, host_output->data());
    ASSERT_EQ(error, dicomNodeError_t::SUCCESS);

    for (u64 idx = 0; idx < DATA_SIZE; idx++) {
      ASSERT_FLOAT_EQ((*host_output)[idx],(*host_expected)[idx]);
    }
  }
}

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
    TEST_DATA::make_spaces(),
    Volume<3,f32>{
      .data= nullptr,
      .m_extent = extent
    }
    };

  Image<3,f32> image_2{
    TEST_DATA::make_spaces(),
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

  reduce_no_mem<1, REGISTRATION::VolumeDifference<f32>, f32>(
    image_1.elements(), &difference, &(device_image_1->volume), &(device_image_2->volume)
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
    output[threadIdx.x] = REGISTRATION::VolumeDifference<f32>::map_to(threadIdx.x, &(image_1->volume), &(image_2->volume));
  }
}

TEST(REDUCE, COPY_INTO_SHARED_WORKS) {
  Image<3,f32> image_1{
    TEST_DATA::make_spaces(),
    Volume<3,f32>{
      .data= nullptr,
      .m_extent = extent
    }
  };

  Image<3,f32> image_2{
    TEST_DATA::make_spaces(),
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
