#pragma once

#include"lin_alg.cuh"

class Texture {
  public:
    cudaTextureObject_t texture;
    Space<3> space;

  __device__ float operator()(const Point<3>& point) const {
    const Point<3> interpolated_coordinate = space.interpolate_point(point);

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

static cudaError_t free_texture(Texture** texture){
  if(!(*texture)){
    return cudaSuccess;
  }

  cudaPointerAttributes attr;
  Texture* ptr = *texture;
  cudaError_t error = cudaPointerGetAttributes(&attr, ptr);

  if(error){
    const char* error_name = cudaGetErrorName(error);
    printf("Encountered %s not get pointer info of %p\n", error_name, ptr);
    return error;
  }

  if(attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged){
    Texture host_texture;
    error = cudaMemcpy(&host_texture, ptr, sizeof(Texture), cudaMemcpyDefault);
    if(error){
      const char* error_name = cudaGetErrorName(error);
      printf("ptr: %p\n", ptr);
      printf("ptr->texture: %p\n", &(ptr->texture));

      printf("Encountered %s while copying %p to %p\n", error_name, ptr, &host_texture);
      return error;
    }

    error = cudaDestroyTextureObject(host_texture.texture);
    if(error){
      const char* error_name = cudaGetErrorName(error);
      printf("Encountered %s while destroying the texture object\n", error_name);
      return error;
    }

    error = cudaFree(ptr);
    if(error){
      const char* error_name = cudaGetErrorName(error);
      printf("Encountered %s while freeing %p\n", error_name, ptr);
      return error;
    }
    *texture = nullptr;
  } else {
    error = cudaDestroyTextureObject((*texture)->texture);
    if(error){
      const char* error_name = cudaGetErrorName(error);
      printf("Encountered %s while destroying the texture object\n", error_name);
    }
  }

  return error;
}
