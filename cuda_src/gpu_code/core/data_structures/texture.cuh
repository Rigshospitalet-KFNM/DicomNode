#pragma once

#include"../declarations.cuh"
#include"../concepts.cuh"
#include"space.cuh"
#include"image.cuh"
#include"../cuda_management.cuh"

template<u8 DIMENSION, typename T>
class Texture final {
  static_assert(DIMENSION == 3, "Texture is only support for 3 dimensional volumes");

  public:
    cudaTextureObject_t texture;
    Space<DIMENSION> space;
    Extent<DIMENSION> m_extent;

  __device__ T operator()(const Point<3>& point) const {
    const Point<DIMENSION> interpolated_coordinate = space.interpolate_point(point);

    return tex3D<T>(texture,
      interpolated_coordinate[0] + 0.5f,
      interpolated_coordinate[1] + 0.5f,
      interpolated_coordinate[2] + 0.5f
    );
  }

  [[nodiscard]] __host__ __device__ const Extent<DIMENSION>& extent() const noexcept {
    return m_extent;
  }

  [[nodiscard]] __host__ __device__ constexpr size_t elements() const noexcept {
    return m_extent.elements();
  }
};

static_assert(CImage<Texture<3, float>, 3>, "Texture is not a volume?");

template<typename T>
dicomNodeError_t load_texture(
  Texture<3, T>* texture,
  const Image<3, T>& image
){

  const Space<3>& space = image.space;
  cudaMemcpy(&texture->m_extent, &image.extent(), sizeof(Extent<3>), cudaMemcpyDefault);

  cudaTextureDesc textureDescription{};
  cudaResourceDesc resourceDescription{};

  memset(&textureDescription, 0, sizeof(textureDescription));
  memset(&resourceDescription, 0, sizeof(resourceDescription));

  const cudaChannelFormatDesc channelFormatDescription = cudaCreateChannelDesc<T>();
  cudaTextureObject_t host_texture;

  const cudaExtent extent = make_cudaExtent(
      image.extent().x(),
      image.extent().y(),
      image.extent().z()
  );

  DicomNodeRunner runner;

  runner
    | [&]() noexcept {
      return cudaMemcpy(&(texture->space), &space, sizeof(Space<3>), cudaMemcpyDefault);
  } | [&]() noexcept{
      resourceDescription.resType = cudaResourceTypeArray;
      return cudaMalloc3DArray(
        &(resourceDescription.res.array.array),
        &channelFormatDescription,
        extent,
        cudaArrayDefault
      );
  } | [&]() noexcept {
    cudaMemcpy3DParms params = { 0 };
    params.dstArray = resourceDescription.res.array.array;
    params.srcPtr = make_cudaPitchedPtr(
      (void*)image.volume.data,
      image.extent().x() * sizeof(T),
      image.extent().x(),
      image.extent().y()
    );
    params.extent = extent;
    params.kind = cudaMemcpyDefault;
    return cudaMemcpy3D(&params);

  } | [&]() noexcept {
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
  } | [&]() noexcept {
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
  cudaResourceDesc res{};

  DicomNodeRunner runner{[](dicomNodeError_t error){
    std::cout << "free_host_texture encoutered an error:" << error_to_human_readable(error) << "\n";
  }};

  runner | [&]()noexcept {
    return cudaGetTextureObjectResourceDesc(&res, texture.texture);
  } | [&]()noexcept {
    if(res.resType == cudaResourceTypeArray){
      return cudaFreeArray(res.res.array.array);
    } else {
      printf("Texture was not allocated with array?\n");
      assert(false);
      return cudaError_t::cudaSuccess;
    }
  } | [&]() noexcept {
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

  cudaPointerAttributes attr{};
  Texture<3, T>* ptr = *texture;

  DicomNodeRunner runner([&](dicomNodeError_t error){
    std::cout << "free_host_texture encoutered an error:" << error_to_human_readable(error) << "\n";
  });

  runner
    | [&]() noexcept {
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
