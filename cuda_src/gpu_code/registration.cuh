#pragma once

#include<cuda/cmath>
#include<cuda/std/array>
#include<cuda/std/limits>

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
struct OptimizerParam {
  f32 scale = 1.0f;
  Point<3> translations = {0.0f,0.0f,0.0f};
  cuda::std::array<f32, 3> rotations = {0.0f,0.0f,0.0f};
  T cost = cuda::std::numeric_limits<T>::max();

    OptimizerParam& compare(OptimizerParam& other) noexcept {
      return cost < other.cost ? *this : other;
    };
};

template<typename T>
struct Optimizer {
  OptimizerParam<T> params;
  OptimizerParam<T> max_step_size {
    .scale = 0.025,
    .translations = Point<3>{4.0f,4.0f,4.0f},
    .rotations = {0.05f,0.05f,0.05f},
  };

  Optimizer(auto cost_function) {
    params.cost = cost_function(params);
  }


  T update_scale(auto cost_function) {
    OptimizerParam positive_param = params;
    OptimizerParam negative_param = params;

    // Check scale param
    positive_param.scale += max_step_size.scale;
    negative_param.scale -= max_step_size.scale;

    positive_param.cost = cost_function(positive_param);
    negative_param.cost = cost_function(negative_param);

    params = params.compare(positive_param);
    params = params.compare(negative_param);

    return params.cost;
  }

  T update_translation(auto cost_function) {
    OptimizerParam positive_param = params;
    OptimizerParam negative_param = params;

    for (u8 dim = 0; dim < 3; dim++) {
      positive_param.translations[dim] += max_step_size.translations[dim];
      negative_param.translations[dim] -= max_step_size.translations[dim];

      positive_param.cost = cost_function(positive_param);
      negative_param.cost = cost_function(negative_param);

      params = params.compare(positive_param);
      params = params.compare(negative_param);

      if (dim < 2) {
        positive_param = params;
        negative_param = params;
      }
    }

    return params.cost;
  }

  T update_rotations(auto cost_function) {
    OptimizerParam positive_param = params;
    OptimizerParam negative_param = params;

    for (u8 dim = 0; dim < 3; dim++) {
      positive_param.rotations[dim] += max_step_size.rotations[dim];
      negative_param.rotations[dim] -= max_step_size.rotations[dim];

      positive_param.cost = cost_function(positive_param);
      negative_param.cost = cost_function(negative_param);

      params = params.compare(positive_param);
      params = params.compare(negative_param);

      if (dim < 2) {
        positive_param = params;
        negative_param = params;
      }
    }

    return params.cost;
  }

};

template<typename T>
dicomNodeError_t modify_space(
  Space<3> host_space, // VERY INTENTIONAL NOT A &
  OptimizerParam<T>& modification,
  Space<3>* device_pointer
) {

  SquareMatrix<3> scale {
    modification.scale, 0.0f,0.0f,
    0.0f, modification.scale, 0.0f,
    0.0f, 0.0f, modification.scale
  };

  f32 cos_x = cos(modification.rotations[0]);
  f32 cos_y = cos(modification.rotations[1]);
  f32 cos_z = cos(modification.rotations[2]);
  f32 sin_x = sin(modification.rotations[0]);
  f32 sin_y = sin(modification.rotations[1]);
  f32 sin_z = sin(modification.rotations[2]);

  SquareMatrix<3> rotation_x = {
    1.0,0.0,0.0,
    0.0, cos_x, sin_x,
    0.0, -sin_x, cos_x
  };

  SquareMatrix<3> rotation_y = {
    cos_y, 0.0f, sin_y,
    0.0, 1.0, 0.0,
    -sin_y, 0.0, cos_y
  };

  SquareMatrix<3> rotation_z = {
    cos_z, sin_y, 0.0f,
    -sin_z, cos_z, 0.0f,
    0.0f,0.0f,1.0f
  };

  host_space.basis *= scale * rotation_x * rotation_y * rotation_z;
  host_space.inverted_basis = host_space.basis.inverse();
  host_space.starting_point += modification.translations;

  return encode_cuda_error(cudaMemcpy(device_pointer, &host_space, sizeof(Space<3>), cudaMemcpyDefault));
}


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



    Image<3, T>* device_source_image = nullptr;
    Image<3, T>* device_target_image = nullptr;
    Image<3, T>* device_intermediate_image = nullptr;

    constexpr static dim3 THREAD_BLOCK = THREAD_BLOCK_3D;
    const dim3 envelope = get_envelope_grid<THREAD_BLOCK>(intermediate_image.extent());

    Point<3> source_center_of_gravity;
    Point<3> destination_center_of_gravity;

    auto free_used_memory = [&]() {
      free_device_memory(
        &intermediate_image.data(),
        &device_source_image,
        &device_target_image,
        &device_intermediate_image
      );
    };

    DicomNodeRunner runner{[&](dicomNodeError_t error) {
      free_used_memory();
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

    auto cost_function = [&](OptimizerParam<T>& param) {
      T difference = cuda::std::numeric_limits<T>::max();

      runner | [&]() {
        return modify_space<T>(intermediate_image.space, param, &(device_intermediate_image->space));
      } | [&]() {
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
      };
      return difference;
    };

    Optimizer<T> optimizer(cost_function);

    std::cout << "Start error: " << optimizer.params.cost << "\n";

    optimizer.update_translation(cost_function);
    optimizer.update_rotations(cost_function);
    optimizer.update_scale(cost_function);

    std::cout << "End error: " << optimizer.params.cost << "\n";

    runner | [&]() {
      free_used_memory();
      return SUCCESS;
    };

    return runner.error();
  }
}
