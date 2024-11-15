#pragma once

#include<iostream>

#include<gtest/gtest.h>


#include"../gpu_code/dicom_node_gpu.cuh"

__global__ void interpolation_tests(const Texture* texture, float* out, Space<3>* out_space, float3* coords){
  const Texture& refTexture = *texture;
  const uint32_t x = texture->space.domain[2];
  const uint32_t y = texture->space.domain[1];

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

  local_space.domain = Domain<3>{z,y,x};

  float *out = nullptr;
  cudaError_t cuda_error = cudaMalloc(&out, threads * sizeof(float));
  ASSERT_EQ(cuda_error, cudaSuccess);

  Texture* texture=nullptr;
  Space<3>* out_space = nullptr;
  float3* coords = nullptr;

  cuda_error = cudaMalloc(&texture, sizeof(Texture));
  ASSERT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMalloc(&out_space, sizeof(Space<3>));
  ASSERT_EQ(cuda_error, cudaSuccess);

  cuda_error = cudaMalloc(&coords, sizeof(float3) * threads);
  ASSERT_EQ(cuda_error, cudaSuccess);

  dicomNodeError_t dn_error = load_texture<float>(texture, data, local_space);
  ASSERT_EQ(dn_error, dicomNodeError_t::SUCCESS);

  interpolation_tests<<<1,threads>>>(texture, out, out_space, coords);
  cuda_error = cudaGetLastError();
  ASSERT_EQ(cuda_error, cudaSuccess);

  float out_calced[threads];
  cuda_error = cudaMemcpy(out_calced, out, threads * sizeof(float), cudaMemcpyDefault);
  ASSERT_EQ(cuda_error, cudaSuccess);

  Space<3> out_host_space;

  cuda_error = cudaMemcpy(&out_host_space, out_space, sizeof(Space<3>), cudaMemcpyDefault);
  ASSERT_EQ(cuda_error, cudaSuccess);

  float3 out_coords[threads];
  cuda_error = cudaMemcpy(&out_coords, coords, sizeof(float3) * threads, cudaMemcpyDefault);
  ASSERT_EQ(cuda_error, cudaSuccess);

  for(size_t idx = 0; idx < threads; idx++){
    const size_t lx = idx % x;
    const size_t ly = (idx / x) % y;
    const size_t lz = idx / (x * y);

    float3 c = out_coords[idx];

    ASSERT_EQ(c.x, lx);
    ASSERT_EQ(c.y, ly);
    ASSERT_EQ(c.z, lz);
    ASSERT_EQ(out_calced[idx], data[idx]);
  }

  free_texture(&texture);
  cudaFree(out);
  cudaFree(texture);
  cudaFree(out_space);
}

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
  constexpr size_t threads = out_x * out_y * out_z;

  float data[data_elements];

  for(int i = 0; i < data_elements; i++){
    data[i] = static_cast<float>(i + 1);
  }

  cudaChannelFormatDesc chdsc = cudaCreateChannelDesc<float>();
  cudaExtent extent = make_cudaExtent(x,y,z);

  cudaArray_t cuArray;

  cudaError_t error = cudaMalloc3DArray(&cuArray, &chdsc, extent, cudaArrayDefault);
  ASSERT_EQ(error, cudaSuccess);

  cudaMemcpy3DParms params = {0};

  params.srcPtr = make_cudaPitchedPtr((void*)data,
                                      x * sizeof(float),
                                      x,
                                      y);
  params.dstArray = cuArray;
  params.extent = extent;
  params.kind = cudaMemcpyHostToDevice;

  error = cudaMemcpy3D(&params);
  ASSERT_EQ(error, cudaSuccess);

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
  ASSERT_EQ(error, cudaSuccess);

  float *out = nullptr;
  cudaError_t cuda_error = cudaMalloc(&out, threads * sizeof(float));
  ASSERT_EQ(cuda_error, cudaSuccess);

  float3 *coords = nullptr;
  error = cudaMalloc(&coords, sizeof(float3) * threads);
  ASSERT_EQ(error, cudaSuccess);

  manual_interpolation<<<1, {out_x, out_y, out_z}>>>(texObj, out, coords);

  float out_calced[threads];
  cuda_error = cudaMemcpy(out_calced, out, threads * sizeof(float), cudaMemcpyDefault);
  ASSERT_EQ(cuda_error, cudaSuccess);

  float3 out_coords[threads];
  cuda_error = cudaMemcpy(out_coords, coords, threads * sizeof(float3), cudaMemcpyDefault);
  ASSERT_EQ(cuda_error, cudaSuccess);

  /*
  for(size_t idx = 0; idx < threads; idx++){
    const size_t lx = idx % out_x;
    const size_t ly = (idx / out_x) % out_y;
    const size_t lz = idx / (out_x * out_y);

    float3 l = out_coords[idx];
    float o =  out_calced[idx];

    std::cout << "Thread (" << lx << ", " << ly << ", " << lz << ") value: "
      << o << " at coord: (" << l.x << ", " << l.y << ", " << l.z << ")"<< "\n";
  }

  */

  cudaFree(coords);
  cudaFree(out);
  cudaDestroyTextureObject(texObj);
}