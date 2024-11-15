#pragma once

#include"lin_alg.cuh"

class Texture {
  public:
    cudaTextureObject_t texture;
    Space<3> space;

  __device__ float operator()(const Point<3>& point) const {
    Point<3> interpolated_coordinate = space.interpolate_point(point);

    //printf("thread: %u: %f,%f,%f\n", threadIdx.x, interpolated_coordinate[0], interpolated_coordinate[1], interpolated_coordinate[2]);

    return tex3D<float>(texture,
      interpolated_coordinate[0] + 0.5f,
      interpolated_coordinate[1] + 0.5f,
      interpolated_coordinate[2] + 0.5f
    );
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
  if(!(*texture)){
    return;
  }

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
