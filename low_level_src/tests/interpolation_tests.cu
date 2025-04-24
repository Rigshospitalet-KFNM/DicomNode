#pragma once

#include<iostream>

#include<gtest/gtest.h>


#include"../gpu_code/dicom_node_gpu.cuh"

__global__ void interpolation_tests(const Texture<3, float>* texture, float* out, Space<3>* out_space, float3* coords){
  const Texture<3, float>& refTexture = *texture;
  const uint32_t x = texture->space.extent[2];
  const uint32_t y = texture->space.extent[1];

  const float lx = threadIdx.x % x;
  const float ly = (threadIdx.x / x) % y;
  const float lz = threadIdx.x / (x * y);

  Point<3> point{lx, ly, lz};

  out[threadIdx.x] = refTexture(point);
  coords[threadIdx.x].x = lx;
  coords[threadIdx.x].y = ly;
  coords[threadIdx.x].z = lz;

  if(threadIdx.x == 0){
    *out_space = texture->space;
  }

}

TEST(INTERPOLATION, INTERPOLATE_AT_POINT){
  constexpr size_t z = 3;
  constexpr size_t y = 4;
  constexpr size_t x = 3;
  constexpr size_t threads = x * y * z;

  float data[ z * y * x] = {
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

  Space<3> local_space;

  local_space.basis = SquareMatrix<3>{
    .points={
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f
    }
  };

  local_space.inverted_basis = SquareMatrix<3>{
    .points={
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f
    }
  };

  local_space.starting_point = Point<3>{
    0.0f,0.0f,0.0f
  };

  local_space.extent = Extent<3>{z,y,x};

  float *out = nullptr;
  cudaError_t cuda_error = cudaMalloc(&out, threads * sizeof(float));
  EXPECT_EQ(cuda_error, cudaSuccess);

  Texture<3, float>* texture=nullptr;
  Space<3>* out_space = nullptr;
  float3* coords = nullptr;

  cuda_error = cudaMalloc(&texture, sizeof(Texture<3, float>));
  EXPECT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMalloc(&out_space, sizeof(Space<3>));
  EXPECT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMalloc(&coords, sizeof(float3) * threads);
  EXPECT_EQ(cuda_error, cudaSuccess);

  dicomNodeError_t dn_error = load_texture<float>(texture, data, local_space);
  EXPECT_EQ(dn_error, dicomNodeError_t::SUCCESS);

  interpolation_tests<<<1,threads>>>(texture, out, out_space, coords);
  cuda_error = cudaGetLastError();
  EXPECT_EQ(cuda_error, cudaSuccess);

  float out_calced[threads];
  cuda_error = cudaMemcpy(out_calced, out, threads * sizeof(float), cudaMemcpyDefault);
  EXPECT_EQ(cuda_error, cudaSuccess);

  Space<3> out_host_space;

  cuda_error = cudaMemcpy(&out_host_space, out_space, sizeof(Space<3>), cudaMemcpyDefault);
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

  free_texture<float>(&texture);
  cudaFree(out);
  cudaFree(texture);
  cudaFree(out_space);
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

constexpr size_t z = 4;
constexpr size_t y = 4;
constexpr size_t x = 4;

constexpr size_t data_elements = z * y * x;

constexpr size_t out_z = 3;
constexpr size_t out_y = 3;
constexpr size_t out_x = 3;

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
  constexpr uint32_t x = 10;
  constexpr uint32_t y = 10;
  constexpr uint32_t z = 10;

  constexpr uint32_t ux = x - 1;
  constexpr uint32_t uy = x - 1;
  constexpr uint32_t uz = x - 1;


  uint8_t* host_data = new uint8_t[ x * y * z];

  for(int i = 0; i < x * y * z; i++){
    const uint32_t lx = i % x;
    const uint32_t ly = (i / x) % y;
    const uint32_t lz = i / (x * y);

    host_data[i] = std::max(std::max(lx, ly), lz) + 1;
  }

  const Space<3> source_space {
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
    .extent = Extent<3>{z,y,x}
  };

  const Space<3> target_space {
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
    .extent = Extent<3>{uz, uy, ux}
  };

  constexpr size_t out_elements = uz * uy * ux;
  constexpr size_t out_size = out_elements * sizeof(uint8_t);

  cudaError_t error;

  Texture<3, uint8_t>* texture = nullptr;
  error = cudaMalloc(&texture, sizeof(Texture<3, uint8_t>));

  ASSERT_EQ(error, cudaSuccess);
  uint8_t* device_interpolated_image = nullptr;
  error = cudaMalloc(&device_interpolated_image, out_size);
  ASSERT_EQ(error, cudaSuccess);

  dicomNodeError_t dicomnode_error = load_texture<uint8_t>(
    texture,
    host_data, source_space
  );
  if(dicomnode_error){
    error = extract_cuda_error(dicomnode_error);
    const char* error_name = cudaGetErrorName(error);
    printf("Encountered %s while creating the texture\n", error_name);
  }
  ASSERT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);

  dicomnode_error = gpu_interpolation_linear<uint8_t>(
    texture, target_space, device_interpolated_image
  );
  ASSERT_EQ(dicomnode_error, dicomNodeError_t::SUCCESS);


  uint8_t* host_result = new uint8_t[out_elements];

  error = cudaMemcpy(host_result, device_interpolated_image, out_size, cudaMemcpyDefault);
  ASSERT_EQ(error, cudaSuccess);

  for(uint32_t i = 0; i < out_elements; i++ ){
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
  cudaFree(device_interpolated_image);

  delete[] host_data;
  delete[] host_result;
}