#include<iostream>

#include<gtest/gtest.h>

#include"../gpu_code/dicom_node_gpu.cuh"
#include"../gpu_code/linear_interpolation.cuh"

#include"../python/utilities.cuh"

#include"test_helper.cuh"
#include"test_data.cuh"

namespace TEST_INTERPOLATION {
  namespace DEFAULT_TEST_OBJECTS {
    constexpr static u32 x = 10;
    constexpr static u32 y = 10;
    constexpr static u32 z = 10;

    constexpr static Extent<3> source_extent = Extent<3>{z,y,x};

    constexpr std::array<f32, x * y * z> host_image_data() noexcept {
      std::array<f32, x * y * z> a;

      for(int i = 0; i < x * y * z; i++){
        const u32 lx = i % x;
        const u32 ly = (i / x) % y;
        const u32 lz = i / (x * y);

        a[i] = static_cast<f32>(std::max(std::max(lx, ly), lz) + 1);
      }
      return a;
    };

    constexpr static Space source_space {
      .starting_point = Point<3>{
        0.0f, 0.0f, 0.0f
      },

      .basis = SquareMatrix<3>{
        .points={
          1.0f, 0.0f, 0.0f,
          0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 1.0f
        }
      },
      .inverted_basis = SquareMatrix<3>{
        .points={
          1.0f, 0.0f, 0.0f,
          0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 1.0f
        },
      },
    };

    constexpr static Space destination_space {
      .starting_point = Point<3>{
        1.0f, 1.0f, 1.0f
      },

      .basis = SquareMatrix<3>{
        .points={
          1.0f, 0.0f, 0.0f,
          0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 1.0f
        }
      },
      .inverted_basis = SquareMatrix<3>{
        .points={
          1.0f, 0.0f, 0.0f,
          0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 1.0f
        },
      },
    };
}

namespace {
  __global__ void interpolation_tests(const Texture<3, f32>* texture, Image<3, f32>* out_image, float3* coords){
    const Texture<3, float>& refTexture = *texture;
    const uint32_t x = texture->extent().x();
    const uint32_t y = texture->extent().y();

    const float lx = threadIdx.x % x;
    const float ly = (threadIdx.x / x) % y;
    const float lz = threadIdx.x / (x * y);

    Point<3> point{lx, ly, lz};

    out_image->volume.data[threadIdx.x] = refTexture(point);
    coords[threadIdx.x].x = lx;
    coords[threadIdx.x].y = ly;
    coords[threadIdx.x].z = lz;

  }
}



TEST(INTERPOLATION, INTERPOLATE_AT_POINT){
  constexpr size_t z = 3;
  constexpr size_t y = 4;
  constexpr size_t x = 3;
  constexpr size_t threads = x * y * z;

  constexpr f32 data[z * y * x] = {
    10.0f, 30.0f, 40.0f,
    20.0f, 50.0f, 70.0f,
    310.0f, 130.0f, 240.0f,
    320.0f, 150.0f, 270.0f,

    110.0f, 130.0f, 140.0f,
    120.0f, 150.0f, 170.0f,
    10.0f, 30.0f, 40.0f,
    20.0f, 50.0f, 70.0f,

    -10.0f, 160.0f, -40.0f,
    -20.0f, 150.0f, -720.0f,
    -5.0f, 350.0f, 40.0f,
    -50.0f, -150.0f, -720.0f,
  };

  Space<3> local_space{
    .starting_point = Point<3>{
      0.0f,0.0f,0.0f
    },
    .basis = SquareMatrix<3>{
      .points{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
    .inverted_basis = SquareMatrix<3>{
      .points{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    }
  };

  Image<3,f32> image {
    .space = local_space,
    .volume = {
      .m_extent = Extent<3>{z,y,x}
    }
  };

  cudaError_t cuda_error = cudaMalloc(&image.volume.data, threads * sizeof(f32));
  EXPECT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMemcpy(image.volume.data, data, image.size(), cudaMemcpyDefault);
  EXPECT_EQ(cuda_error, cudaSuccess);

  Texture<3, f32>* device_texture = nullptr;
  Image<3, f32>* device_image = nullptr;
  float3* coords = nullptr;

  cuda_error = cudaMalloc(&device_texture, sizeof(Texture<3, float>));
  EXPECT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMalloc(&device_image, sizeof(Image<3, f32>));
  EXPECT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMemcpy(device_image, &image, sizeof(Image<3, f32>), cudaMemcpyDefault);
  EXPECT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMalloc(&coords, sizeof(float3) * threads);
  EXPECT_EQ(cuda_error, cudaSuccess);

  dicomNodeError_t dn_error = load_texture<f32>(device_texture, image);
  EXPECT_EQ(dn_error, dicomNodeError_t::SUCCESS);

  interpolation_tests<<<1,threads>>>(device_texture, device_image, coords);
  cuda_error = cudaGetLastError();
  EXPECT_EQ(cuda_error, cudaSuccess);

  float out_calced[image.elements()];
  cuda_error = cudaMemcpy(out_calced, image.volume.data, image.size(), cudaMemcpyDefault);
  EXPECT_EQ(cuda_error, cudaSuccess);

  float3 out_coords[threads];
  cuda_error = cudaMemcpy(&out_coords, coords, sizeof(float3) * threads, cudaMemcpyDefault);
  EXPECT_EQ(cuda_error, cudaSuccess);

  for(size_t idx = 0; idx < threads; idx++){
    const size_t lx = idx % x;
    const size_t ly = (idx / x) % y;
    const size_t lz = idx / (x * y);

    float3 c = out_coords[idx];

    EXPECT_EQ(c.x, lx);
    EXPECT_EQ(c.y, ly);
    EXPECT_EQ(c.z, lz);
    EXPECT_EQ(out_calced[idx], data[idx]);
  }

  free_texture<float>(&device_texture);
  cudaFree(image.volume.data);
  cudaFree(device_image);
  cudaFree(coords);
}

#ifdef PERFORMANCE

TEST(INTERPOLATION, INTERPOLATE_REAL_BIG){
  constexpr size_t z = 354;
  constexpr size_t y = 512;
  constexpr size_t x = 512;
  constexpr size_t data_elements = x * y * z;
  constexpr size_t data_size = data_elements * sizeof(float);

  float* data_host = new float[data_elements];

  for(int64_t i = 0; i < data_elements; i++){
    data_host[i] = -1.0f/(float)((i + 1));
  }

  Space<3> local_space;

  local_space.basis = SquareMatrix<3>{
    .points={
      1.5234375f, 0.0f, 0.0f,
      0.0f, 1.5234375f, 0.0f,
      0.0f, 0.0f, 3.0f
    }
  };

  local_space.inverted_basis = SquareMatrix<3>{
    .points={
      0.6564103f, 0.0f, 0.0f,
      0.0f, 0.6564103f, 0.0f,
      0.0f, 0.0f, 0.33333333f
    }
  };

  local_space.starting_point = Point<3>{
    -389.23828125f, -573.23828125f, -1133.0f
  };

  local_space.extent = Extent<3>{z,y,x};

  float *out = nullptr;
  cudaError_t cuda_error = cudaMalloc(&out, data_size);
  EXPECT_EQ(cuda_error, cudaSuccess);

  Texture<float>* texture=nullptr;
  Space<3>* out_space = nullptr;
  float3* coords = nullptr;

  cuda_error = cudaMalloc(&texture, sizeof(Texture));
  EXPECT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMalloc(&out_space, sizeof(Space<3>));
  EXPECT_EQ(cuda_error, cudaSuccess);


  dicomNodeError_t dicomnode_error = load_texture<float>(texture, data_host, local_space);
  EXPECT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);

  dicomnode_error = gpu_interpolation_linear<float>(
    texture, local_space, out
  );
  EXPECT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);

  float* interpolated = new float[data_elements];

  cuda_error = cudaMemcpy(interpolated, out, data_size, cudaMemcpyDefault);
  EXPECT_EQ(cuda_error, cudaSuccess);

  for(int64_t i = 0; i < data_elements; i++){
    EXPECT_FLOAT_EQ(interpolated[i], data_host[i]);
  }

  delete[] interpolated;
  delete[] data_host;

  free_texture<float>(&texture);
  cudaFree(out);
  cudaFree(texture);
  cudaFree(out_space);
  cudaFree(coords);
}

#endif

__global__ void manual_interpolation(cudaTextureObject_t tex, float* out, float3* coords){
  const float lx = (static_cast<float>(threadIdx.x) + 0.5f);
  const float ly = (static_cast<float>(threadIdx.y) + 0.5f);
  const float lz = (static_cast<float>(threadIdx.z) + 0.5f);

  const size_t out_index = threadIdx.z * blockDim.x * blockDim.z
    + threadIdx.y * blockDim.x
    + threadIdx.x;

  out[out_index] = tex3D<float>(tex, lx + 0.5f, ly + 0.5f, lz + 0.5f);

  coords[out_index].x = lx;
  coords[out_index].y = ly;
  coords[out_index].z = lz;
}

TEST(INTERPOLATION, Manual_interpolation){
  constexpr size_t z = 4;
  constexpr size_t y = 4;
  constexpr size_t x = 4;

  constexpr size_t data_elements = z * y * x;

  constexpr size_t out_z = 3;
  constexpr size_t out_y = 3;
  constexpr size_t out_x = 3;


  constexpr size_t threads = out_x * out_y * out_z;

  float data[data_elements];

  for(int i = 0; i < data_elements; i++){
    data[i] = static_cast<float>(i + 1);
  }

  cudaChannelFormatDesc chdsc = cudaCreateChannelDesc<float>();
  cudaExtent extent = make_cudaExtent(x,y,z);

  cudaArray_t cuArray;

  cudaError_t error = cudaMalloc3DArray(&cuArray, &chdsc, extent, cudaArrayDefault);
  EXPECT_EQ(error, cudaSuccess);

  cudaMemcpy3DParms params = {0};

  params.srcPtr = make_cudaPitchedPtr((void*)data,
                                      x * sizeof(float),
                                      x,
                                      y);
  params.dstArray = cuArray;
  params.extent = extent;
  params.kind = cudaMemcpyHostToDevice;

  error = cudaMemcpy3D(&params);
  EXPECT_EQ(error, cudaSuccess);

  cudaResourceDesc resDesc = {};
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  cudaTextureDesc texDesc = {};
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.normalizedCoords = 0;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.readMode = cudaReadModeElementType;

  // Create texture object
  cudaTextureObject_t texObj;

  error = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  EXPECT_EQ(error, cudaSuccess);

  float *out = nullptr;
  cudaError_t cuda_error = cudaMalloc(&out, threads * sizeof(float));
  EXPECT_EQ(cuda_error, cudaSuccess);

  float3 *coords = nullptr;
  error = cudaMalloc(&coords, sizeof(float3) * threads);
  EXPECT_EQ(error, cudaSuccess);

  manual_interpolation<<<1, {out_x, out_y, out_z}>>>(texObj, out, coords);

  float out_calced[threads];
  cuda_error = cudaMemcpy(out_calced, out, threads * sizeof(float), cudaMemcpyDefault);
  EXPECT_EQ(cuda_error, cudaSuccess);

  float3 out_coords[threads];
  cuda_error = cudaMemcpy(out_coords, coords, threads * sizeof(float3), cudaMemcpyDefault);
  EXPECT_EQ(cuda_error, cudaSuccess);

  cudaFreeArray(cuArray);
  cudaFree(coords);
  cudaFree(out);
  cudaDestroyTextureObject(texObj);
}

TEST(INTERPOLATION, INTERPOLATE_UINT8){
  constexpr u32 x = 10;
  constexpr u32 y = 10;
  constexpr u32 z = 10;

  constexpr Extent<3> source_extent{z,y,x};

  constexpr u32 ux = x - 1;
  constexpr u32 uy = y - 1;
  constexpr u32 uz = z - 1;

  constexpr Extent<3> target_extent = { uz, uy, ux};

  std::array<u8, source_extent.elements()> host_data;

  for(int i = 0; i < x * y * z; i++){
    const u32 lx = i % x;
    const u32 ly = (i / x) % y;
    const u32 lz = i / (x * y);

    host_data[i] = std::max(std::max(lx, ly), lz) + 1;
  }

  constexpr Space<3> source_space {
    .starting_point = Point<3>{
      0.0f, 0.0f, 0.0f
    },

    .basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
    .inverted_basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      },
    },
  };

  Volume<3, u8> source_volume = {
    .m_extent = source_extent
  };

  cudaMalloc(&source_volume.data, source_volume.size());
  cudaMemcpy(source_volume.data, host_data.data(), source_volume.size(), cudaMemcpyDefault);

  Image<3,u8> source_image = {
    .space = source_space,
    .volume = source_volume
  };

  constexpr Space<3> target_space {
    .starting_point = Point<3>{
      1.0f, 1.0f, 1.0f
    },
    .basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
   .inverted_basis = SquareMatrix<3>{
     .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      },
    },
  };


  Image<3,u8> target_image = {
    .space = target_space,
    .volume = {
      .m_extent = target_extent
    }
  };

  cudaError_t error;
  CUDA_CHECK(cudaMalloc(&target_image.data(), target_image.size()));

  Texture<3, u8>* texture = nullptr;
  error = cudaMalloc(&texture, sizeof(Texture<3, uint8_t>));

  ASSERT_EQ(error, cudaSuccess);

  dicomNodeError_t dicomnode_error = load_texture<uint8_t>(
    texture,
    source_image
  );

  if(dicomnode_error){
    error = extract_cuda_error(dicomnode_error);
    const char* error_name = cudaGetErrorName(error);
    printf("Encountered %s while creating the texture\n", error_name);
  }
  ASSERT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);

  dicomnode_error = gpu_interpolation_linear<u8>(
    texture, target_image, target_image.data()
  );
  ASSERT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);

  std::array<u8, target_extent.elements()>  host_result;

  error = cudaMemcpy(host_result.data(), target_image.data(), target_image.size(), cudaMemcpyDefault);
  ASSERT_EQ(error, cudaSuccess);

  for(uint32_t i = 0; i < target_image.elements(); i++ ){
    const uint32_t lux = i % ux;
    const uint32_t luy = (i / ux) % uy;
    const uint32_t luz = i / (ux * uy);

    const uint32_t lx = lux + 1;
    const uint32_t ly = luy + 1;
    const uint32_t lz = luz + 1;

    EXPECT_EQ(
      host_result[i], host_data[ lz * y * x + ly * x + lx]
    );
  }

  free_texture(&texture);
  cudaFree(source_image.data());
  cudaFree(target_image.data());
}

TEST(INTERPOLATION, INTERPOLATE_IMAGE){
  using DataType = f32;

  constexpr u32 x = 10;
  constexpr u32 y = 10;
  constexpr u32 z = 10;

  constexpr Extent<3> source_extent{z,y,x};

  constexpr u32 ux = 9;
  constexpr u32 uy = 9;
  constexpr u32 uz = 9;

  constexpr Extent<3> target_extent{uz,uy,ux};


  std::array<DataType, source_extent.elements()> host_data;

  for(int i = 0; i < x * y * z; i++){
    const u32 lx = i % x;
    const u32 ly = (i / x) % y;
    const u32 lz = i / (x * y);

    host_data[i] = static_cast<DataType>(std::max(std::max(lx, ly), lz) + 1);
  }

  constexpr Space<3> source_space {
    .starting_point = Point<3>{
      0.0f, 0.0f, 0.0f
    },

    .basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
    .inverted_basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      },
    },
  };



  Image<3, DataType> source_image {
    source_space,
    Volume<3, DataType>{
      .m_extent = source_extent
    }
  };

  cudaMalloc(&source_image.data(), source_image.size());
  cudaMemcpy(source_image.data(), host_data.data(), source_image.size(), cudaMemcpyDefault);

  constexpr Space<3> target_space {
    .starting_point = Point<3>{
      1.0f, 1.0f, 1.0f
    },
    .basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
    .inverted_basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      },
    },
  };

  Image<3, DataType> target_image{
    target_space,
    Volume<3, DataType> {
      .m_extent = target_extent
    }
  };

  cudaMalloc(&target_image.data(), target_image.size());

  dicomNodeError_t dicomnode_error = gpu_interpolation_linear<DataType>(
    source_image, target_image, target_image.data()
  );

  ASSERT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);

  std::array<DataType, target_extent.elements()> host_result;
  cudaMemcpy(host_result.data(), target_image.data(), target_image.size(), cudaMemcpyDefault);


  for(uint32_t i = 0; i < target_image.elements(); i++ ){
    const uint32_t lux = i % ux;
    const uint32_t luy = (i / ux) % uy;
    const uint32_t luz = i / (ux * uy);

    const uint32_t lx = lux + 1;
    const uint32_t ly = luy + 1;
    const uint32_t lz = luz + 1;

    EXPECT_EQ(
      host_result[i], host_data[ lz * y * x + ly * x + lx]
    );
  }

  cudaFree(source_image.data());
  cudaFree(target_image.data());
}

TEST(INTERPOLATION, THREAD_BLOCK_INDEXING) {
  using DataType = f32;
  auto host_data = DEFAULT_TEST_OBJECTS::host_image_data();
    // This is allowed to be a variable because it's used twice, and it's very
    // important that the target space is the same for both images
  constexpr Space<3> target_space = DEFAULT_TEST_OBJECTS::destination_space;

  Image<3, DataType> host_image {
    .space = DEFAULT_TEST_OBJECTS::source_space,
    .volume = Volume<3, DataType>{
      .m_extent = DEFAULT_TEST_OBJECTS::source_extent
    }
  };

  CUDA_CHECK(cudaMalloc(&host_image.data(), host_image.size()));
  CUDA_CHECK(cudaMemcpy(host_image.data(), host_data.data(), host_image.size(), cudaMemcpyDefault));


  constexpr Extent<3> out_extent{8,8,8};

  Image<3, DataType> target_image_normal{
    target_space,
    Volume<3, DataType>{
      .m_extent = out_extent
    }
  };

  cudaMalloc(&target_image_normal.data(), target_image_normal.size());

  Image<3, DataType> target_image_shared{
    target_space,
    Volume<3, DataType>{
      .m_extent = out_extent
    }
  };

  cudaMalloc(&target_image_shared.data(), target_image_shared.size());

  std::array<DataType, out_extent.elements()> host_result;
  std::array<DataType, out_extent.elements()> host_result_shared;

  dicomNodeError_t dicomnode_error = gpu_interpolation_linear<DataType>(
    host_image, target_image_normal, target_image_normal.data()
  );
  EXPECT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);

  CUDA_CHECK(cudaMemcpy(
    host_result.data(),
    target_image_normal.data(),
    target_image_normal.size(),
    cudaMemcpyDefault
    ));

  dicomnode_error = gpu_interpolation_linear_t<DataType, INTERPOLATION::kernel_interpolation_linear_blocked<DataType>>(
    host_image, target_image_shared, target_image_shared.data()
  );
  EXPECT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);

  CUDA_CHECK(cudaMemcpy(host_result_shared.data(), target_image_shared.data(), target_image_shared.size(), cudaMemcpyDefault));

  for (u32 i = 0; i < target_image_shared.elements(); i++) {
    EXPECT_FLOAT_EQ(host_result[i], host_result_shared[i]);
    if (host_result[i] != host_result_shared[i]) {
      std::cout << "Error at: " << i << "\n";
      break;
    }
  }

  // Freeing
  cudaFree(host_image.data());
  cudaFree(target_image_normal.data());
  cudaFree(target_image_shared.data());

}

TEST(INTERPOLATION, SHARED_MEMORY_INTERPOLATION) {
  using DataType = f32;

  auto host_data = DEFAULT_TEST_OBJECTS::host_image_data();

  Image<3, DataType> host_image{
    DEFAULT_TEST_OBJECTS::source_space,
    Volume<3, DataType>{
      .m_extent = DEFAULT_TEST_OBJECTS::source_extent,
      .default_value = -1
    }
  };

  constexpr Extent<3> target_extent{5,5,5};

  Image<3, DataType> target_image{
    DEFAULT_TEST_OBJECTS::destination_space,
    Volume<3, DataType>{
      .m_extent = target_extent
    }
  } ;

  CUDA_CHECK(cudaMalloc(&host_image.data(), host_image.size()));
  CUDA_CHECK(cudaMemcpy(host_image.data(), host_data.data(), host_image.size(), cudaMemcpyDefault));
  CUDA_CHECK(cudaMalloc(&target_image.data(), target_image.size()));


  std::array<DataType, target_extent.elements()> host_result;
  std::array<DataType, target_extent.elements()> host_result_shared;

  dicomNodeError_t dicom_node_error = gpu_interpolation_linear<DataType>(
    host_image, target_image, target_image.data()
  );

  EXPECT_EQ(dicom_node_error, dicomNodeError_t::SUCCESS);

  CUDA_CHECK(cudaMemcpy(host_result.data(), target_image.data(), target_image.size(), cudaMemcpyDefault));
  dicom_node_error = gpu_interpolation_linear_t<DataType, INTERPOLATION::kernel_interpolation_linear_shared<DataType>>(
    host_image, target_image, target_image.data()
  );
  EXPECT_EQ(dicom_node_error, dicomNodeError_t::SUCCESS);
  CUDA_CHECK(cudaMemcpy(host_result_shared.data(), target_image.data(), target_image.size(), cudaMemcpyDefault));

  for (u32 i = 0; i < target_extent.elements(); i++) {
     EXPECT_FLOAT_EQ(host_result[i], host_result_shared[i]);
    if (host_result[i] != host_result_shared[i]) {
      std::cout << "Error at: " << i << "\n";
      break;
    }
  }

  // Freeing
  cudaFree(host_image.data());
  cudaFree(target_image.data());
}

TEST(INTERPOLATION, INTERPOLATION_CHEATING) {
  using DataType = f32;
  constexpr auto host_data = DEFAULT_TEST_OBJECTS::host_image_data();

  Image<3, DataType> host_image{
    .space=DEFAULT_TEST_OBJECTS::source_space,
    .volume{
      .m_extent = DEFAULT_TEST_OBJECTS::source_extent,
      .default_value = -1.0
    }
  };
  cudaMalloc(&host_image.data(), host_image.size());
  cudaMemcpy(host_image.data(), host_data.data(), host_image.size(), cudaMemcpyDefault);

  constexpr Extent<3> out_extent{5,5,5};
  constexpr u64 out_elements = out_extent.elements();

  Image<3, DataType> target_image{
    .space=DEFAULT_TEST_OBJECTS::destination_space,
    .volume=Volume<3, DataType>{
      .m_extent = out_extent,
      .default_value = -1.0
    }
  };

  cudaMalloc(&target_image.data(), target_image.size());


  auto *result = new DataType[out_elements];
  auto *result_shared = new DataType[out_elements];

  //std::array<DataType, 5 * 5 * 5> result;
  //std::array<DataType, 5 * 5 * 5> result_shared;


  dicomNodeError_t derror = gpu_interpolation_linear<DataType>(
    host_image, target_image, target_image.data()
  );

  EXPECT_EQ(derror, dicomNodeError_t::SUCCESS);

  CUDA_CHECK(cudaMemcpy(
    result,
    target_image.data(),
    target_image.size(),
    cudaMemcpyDefault
  ));

  derror = gpu_interpolation_linear_shared_cheating<DataType>(
    host_image, target_image, target_image.data()
  );
  EXPECT_EQ(derror, dicomNodeError_t::SUCCESS);
  CUDA_CHECK(cudaMemcpy(
    result_shared, target_image.data(), target_image.size(), cudaMemcpyDefault));

  for (u32 i = 0; i < target_image.elements(); i++) {
     EXPECT_FLOAT_EQ(result[i], result_shared[i]);
    if (result[i] != result_shared[i]) {
      std::cout << "Error at: " << i << "\n";
      break;
    }
  }

  // Freeing
  cudaFree(host_image.data());
  cudaFree(target_image.data());

  delete[] result;
  delete[] result_shared;
}



TEST(INTERPOLATION, DIFFERENT_SPACES) {
  using DataType = f32;
  constexpr Extent<3> extent = {3,3,3};

  auto data = [&]() {
    std::array<DataType,extent.elements()> host_data;

    for (u32 i = 0; i < extent.elements(); i++) {
      host_data[i] = static_cast<DataType>(i + 1);
    }

    return host_data;
  }();

  Space<3> starting_space = TEST_DATA::make_spaces();

  Image<3, DataType> host_image{
    starting_space,
    Volume<3, DataType>{
      .data = nullptr,
      .m_extent = extent,
      .default_value = 0.0
    }
  };

  cudaMalloc(&host_image.data(), host_image.size());
  cudaMemcpy(host_image.data(), data.data(), host_image.size(), cudaMemcpyDefault);

  Space<3> ending_space{
    .starting_point = {2.0,2.0,2.0},
    .basis = {
      0.2f,-1.0f,0.0f,
      -1.0f, 0.6f, 4.0f,
      0.4f,0.0f,1.0f
    },
    .inverted_basis = {
      -0.24193548f,-0.40322581f,1.61290323f,
      -1.0483871f, -0.08064516f, 0.32258065,
      0.09677419f,0.16129032f,0.35483871
    },
  };

  constexpr Extent<3> out_extent{3,3,3};

  Image<3, DataType> target_image{
    .space=ending_space,
    .volume=Volume<3, DataType>{
      .m_extent = out_extent,
      .default_value = -0.0
    }
  };

  cudaMalloc(&target_image.data(), target_image.size());

  dicomNodeError_t error = gpu_interpolation_linear_shared_cheating(
    host_image,
    target_image,
    target_image.data()
  );

  EXPECT_EQ(error, dicomNodeError_t::SUCCESS);

  std::array<DataType,extent.elements()> host_out;
  cudaMemcpy(host_out.data(), target_image.data(), target_image.size(), cudaMemcpyDefault);

  error = gpu_interpolation_linear_cheating(
    host_image,
    target_image,
    target_image.data()
  );
  EXPECT_EQ(error, dicomNodeError_t::SUCCESS);

  std::array<DataType,extent.elements()> host_out_shared;
  cudaMemcpy(
    host_out_shared.data(),
    target_image.data(),
    target_image.size(),
    cudaMemcpyDefault
  );

  for (u32 i = 0; i < extent.elements(); i++) {
    EXPECT_FLOAT_EQ(host_out[i], host_out_shared[i]);
  }

}

}
