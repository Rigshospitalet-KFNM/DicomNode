#pragma once

#include"lin_alg.cuh"

class Texture {
  public:
    cudaTextureObject_t texture;
    Space<3> space;

  __device__ float operator()(Point<3> point) const {
    return tex3D<float>(texture, point[0] + 0.5f, point[1] + 0.5f, point[2] + 0.5f);
  }

  __device__ float operator()(float3 point) const {
    return tex3D<float>(texture, point.x + 0.5f, point.y + 0.5f, point.z + 0.5f);
  }
};

template<typename T>
dicomNodeError_t load_texture(
  Texture* texture,
  const T* data,
  const Space<3>& space
){
  cudaTextureDesc textureDescription;
  memset(&textureDescription, 0, sizeof(textureDescription));
  cudaResourceDesc resourceDescription;
  memset(&resourceDescription, 0, sizeof(resourceDescription));
  const cudaChannelFormatDesc channelFormatDecription = cudaCreateChannelDesc<float>();
  cudaTextureObject_t host_texture;

  const cudaExtent extent = make_cudaExtent(
    space.domain[2],
    space.domain[1],
    space.domain[0]
  );

  DicomNodeRunner runner;

  runner
    | [&](){
      return cudaMemcpy(&(texture->space), &space, sizeof(Space<3>), cudaMemcpyDefault);
  } | [&](){
      resourceDescription.resType = cudaResourceTypeArray;
      return cudaMalloc3DArray(
        &(resourceDescription.res.array.array),
        &channelFormatDecription,
        extent,
        cudaArrayDefault
      );
  } | [&](){
    cudaMemcpy3DParms params = { 0 };
    params.dstArray = resourceDescription.res.array.array;
    params.srcPtr = make_cudaPitchedPtr(
      (void*)data,
      space.domain[2] * sizeof(T),
      space.domain[2],
      space.domain[1]
    );
    params.extent = extent;
    params.kind = cudaMemcpyDefault;
    return cudaMemcpy3D(&params);

  } | [&](){
      textureDescription.normalizedCoords = 0;
      textureDescription.filterMode = cudaFilterModeLinear;
      textureDescription.addressMode[0] = cudaAddressModeClamp;
      textureDescription.addressMode[1] = cudaAddressModeClamp;
      textureDescription.addressMode[2] = cudaAddressModeClamp;
      textureDescription.readMode = cudaReadModeElementType;

      return cudaCreateTextureObject(
        &host_texture,
        &resourceDescription,
        &textureDescription,
        NULL
      );
  } | [&](){
    return cudaMemcpy(
      &(texture->texture),
      &host_texture,
      sizeof(cudaTextureObject_t),
      cudaMemcpyDefault
    );
  };

  return runner.error();
}

static void free_texture(Texture** texture){
  cudaPointerAttributes attr;
  Texture* ptr = *texture;
  cudaPointerGetAttributes(&attr, ptr);

  if(attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged){
    cudaTextureObject_t host_texture;
    cudaMemcpy(&host_texture, &ptr->texture, sizeof(cudaTextureObject_t), cudaMemcpyDefault);
    cudaDestroyTextureObject(host_texture);
    cudaFree(ptr);
    *texture = nullptr;
  } else {
    cudaDestroyTextureObject((*texture)->texture);
  }

}
