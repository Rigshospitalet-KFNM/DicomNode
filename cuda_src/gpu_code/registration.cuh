#pragma once

#include<cuda/cmath>
#include<cuda/type_traits>

#include "center_of_gravity.cuh"
#include"core/core.cuh"
#include"linear_interpolation.cuh"
#include"map_reduce.cuh"

namespace REGISTRATION {
  template<typename T>
  struct VolumeDifference {
    static __device__ __host__ T map_to(u64 global_index, const Volume<3, T>* volume_1, const Volume<3, T>* volume_2) noexcept {
      // ASSERT volumes ARE OF THE SAME SIZE, and each thread shouldn't assert that
      if (volume_1->elements() <= global_index){
        return identity();
      }
      T t;
      // This avoids underflow with unsigned numbers
      if constexpr (cuda::std::is_unsigned_v<T>){
        // This is a downcast of u64 to u32, but like u64 image values are fucking stupid
        t = __usad(volume_1->data[global_index], volume_2->data[global_index], 0u);
      } else {
        t = cuda::std::abs(volume_1->data[global_index] - volume_2->data[global_index]);
      }
      return t * t;
    }

    static __device__ __host__ T apply(const T& a, const T& b) noexcept {
      return a + b;
    }

    static __device__ __host__ bool equals(const T& a, const T& b) noexcept {
      return a == b;
    }

    static __device__ __host__ T identity() noexcept {
      return 0;
    }

    static __device__ __host__ T remove_volatile(volatile T& v) noexcept {
      T vv = v;
      return vv;
    }
  };

template<typename T>
dicomNodeError_t register_to(
  Image<3, T>& host_source_image,
  Image<3, T>& host_target_image
) {
    Image<3,T> intermediate_image(
      host_target_image.space,
      Volume<3, T>{
        .data = nullptr,
        .m_extent = host_target_image.extent(),
        .default_value = 0
      }
    );

    T difference = 0;

    Image<3, T>* device_source_image = nullptr;
    Image<3, T>* device_target_image = nullptr;
    Image<3, T>* device_intermediate_image = nullptr;

    constexpr static dim3 THREAD_BLOCK = THREAD_BLOCK_3D;
    const dim3 envelope = get_envelope_grid<THREAD_BLOCK>(intermediate_image.extent());

    Point<3> source_center_of_gravity;
    Point<3> destination_center_of_gravity;



    DicomNodeRunner runner{[&](dicomNodeError_t error) {
      free_device_memory(
        &intermediate_image.data(),
        &device_source_image,
        &device_target_image,
        &device_intermediate_image
      );
    }};
    // Initialzation
    runner | [&]() {
      return cudaMalloc(&(intermediate_image.data()), intermediate_image.size());
    } | [&]() {
      return cudaMalloc(&(device_source_image), sizeof(Image<3, T>));
    } | [&]() {
      return cudaMalloc(&(device_target_image), sizeof(Image<3, T>));
    } | [&]() {
      return cudaMalloc(&(device_intermediate_image), sizeof(Image<3, T>));
    } | [&]() {
      return cudaMemcpy(device_source_image, &host_source_image, sizeof(Image<3, T>), cudaMemcpyDefault);
    } | [&]() {
      return cudaMemcpy(device_target_image, &host_target_image, sizeof(Image<3, T>), cudaMemcpyDefault);
    } | [&]() {
      return CENTER_OF_GRAVITY::center_of_gravity(host_source_image.volume, source_center_of_gravity);
    } | [&]() {
      return CENTER_OF_GRAVITY::center_of_gravity(host_source_image.volume, destination_center_of_gravity);
    } | [&]() {
       intermediate_image.space.starting_point += source_center_of_gravity - destination_center_of_gravity;

      return cudaMemcpy(device_intermediate_image, &intermediate_image, sizeof(Image<3, T>), cudaMemcpyDefault);
    };


    runner | [&]() {
      INTERPOLATION::kernel_interpolation_linear_shared<<<envelope, THREAD_BLOCK>>>(
        device_source_image,
        &(device_intermediate_image->space),
        intermediate_image.data()
      );

      return cudaGetLastError();
    } | [&]() {
      return reduce_no_mem<8, VolumeDifference<T>, T>(
        intermediate_image.elements(),
        &difference,
        &(device_intermediate_image->volume),
        &(device_target_image->volume)
      );
    } | [&]() {
      free_device_memory(
        &intermediate_image.data(),
        &device_source_image,
        &device_target_image,
        &device_intermediate_image
      );
      return dicomNodeError_t::SUCCESS;
    };

      std::cout << "Difference: " << difference << "\n";

      return dicomNodeError_t::SUCCESS;
    }



}
