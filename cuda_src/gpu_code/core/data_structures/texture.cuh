#pragma once

#include"../declarations.cuh"
#include"../concepts.cuh"
#include"space.cuh"
#include"../cuda_management.cuh"

template<uint8_t DIMENSION, typename T>
class Texture {
  static_assert(DIMENSION == 3, "Texture is only support for 3 dimensional volumes");

  public:
    cudaTextureObject_t texture;
    Space<DIMENSION> space;

  __device__ T operator()(const Point<3>& point) const {
    const Point<DIMENSION> interpolated_coordinate = space.interpolate_point(point);

    return tex3D<T>(texture,
      interpolated_coordinate[0] + 0.5f,
      interpolated_coordinate[1] + 0.5f,
      interpolated_coordinate[2] + 0.5f
    );
  }

  const Extent<DIMENSION>& extent() const {
    return space.extent();
  }

  constexpr size_t elements() const {
    return space.elements();
  }
};

static_assert(CImage<Texture<3, float>, 3>, "Texture is not a volume?");

template<typename T>
dicomNodeError_t load_texture(
  Texture<3, T>* texture,
  const T* data,
  const Space<3>& space
){
  cudaTextureDesc textureDescription;
  memset(&textureDescription, 0, sizeof(textureDescription));
  cudaResourceDesc resourceDescription;
  memset(&resourceDescription, 0, sizeof(resourceDescription));
  const cudaChannelFormatDesc channelFormatDescription = cudaCreateChannelDesc<T>();
  cudaTextureObject_t host_texture;

  const cudaExtent extent = make_cudaExtent(
      space.extent[2],
      space.extent[1],
      space.extent[0]
  );

  DicomNodeRunner runner;

  runner
    | [&](){
      return cudaMemcpy(&(texture->space), &space, sizeof(Space<3>), cudaMemcpyDefault);
  } | [&](){
      resourceDescription.resType = cudaResourceTypeArray;
      return cudaMalloc3DArray(
        &(resourceDescription.res.array.array),
        &channelFormatDescription,
        extent,
        cudaArrayDefault
      );
  } | [&](){
    cudaMemcpy3DParms params = { 0 };
    params.dstArray = resourceDescription.res.array.array;
    params.srcPtr = make_cudaPitchedPtr(
      (void*)data,
      space.extent[2] * sizeof(T),
      space.extent[2],
      space.extent[1]
    );
    params.extent = extent;
    params.kind = cudaMemcpyDefault;
    return cudaMemcpy3D(&params);

  } | [&]() {
      constexpr cudaTextureFilterMode filtermode = std::is_same_v<T, float>
        ? cudaFilterModeLinear
        : cudaFilterModePoint;
      textureDescription.normalizedCoords = 0;
      textureDescription.filterMode = filtermode;
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

template<typename T>
dicomNodeError_t free_host_texture(Texture<3, T>& texture){
  cudaResourceDesc res;

  DicomNodeRunner runner{[](dicomNodeError_t error){
    std::cout << "free_host_texture encoutered an error:" << error_to_human_readable(error) << "\n";
  }};

  runner | [&](){
    return cudaGetTextureObjectResourceDesc(&res, texture.texture);
  } | [&](){
    if(res.resType == cudaResourceTypeArray){
      return cudaFreeArray(res.res.array.array);
    } else {
      printf("Texture was not allocated with array?\n");
      return cudaError_t::cudaSuccess;
    }
  } | [&](){
    return cudaDestroyTextureObject(texture.texture);
  };

  return runner.error();
}

template<typename T>
dicomNodeError_t free_texture(Texture<3, T>** texture){
  if(!texture) {
    return dicomNodeError_t::SUCCESS;
  }
  if(!(*texture)){
    return dicomNodeError_t::SUCCESS;
  }

  cudaPointerAttributes attr;
  Texture<3, T>* ptr = *texture;

  DicomNodeRunner runner([&](dicomNodeError_t error){
    std::cout << "free_host_texture encoutered an error:" << error_to_human_readable(error) << "\n";
  });

  runner
    | [&](){
      return cudaPointerGetAttributes(&attr, ptr);
    } | [&](){
      if(is_host_pointer(attr)){
        return free_host_texture(*ptr);
      } else {
        Texture<3, T> host_texture;
        DicomNodeRunner subrunner;
        subrunner | [&](){
          return cudaMemcpy(&host_texture, ptr, sizeof(Texture<3, T>), cudaMemcpyDefault);
        } | [&](){
          return free_host_texture(host_texture);
        } | [&](){
          return cudaFree(ptr);
        };

        return subrunner.error();
      }
    };

  *texture = nullptr;

  return runner.error();
}
