#pragma once

#include<cuda/cmath>
#include<cuda/std/array>
#include<cuda/std/limits>

#include "center_of_gravity.cuh"
#include"core/core.cuh"
#include"linear_interpolation.cuh"
#include"map_reduce.cuh"


namespace REGISTRATION {
  struct ImageTranslation final {
    // Scale is from the Center
    f32 scale = 1.0f;
    Point<3> translations = {0.0f,0.0f,0.0f};
    // Rotation are from the center of the Image in radians
    cuda::std::array<f32, 3> rotations = {0.0f,0.0f,0.0f};
  };


template<typename T>
struct VolumeDifference final {
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
auto centerImage(DicomNodeRunner& runner, Image<3, T>& host_image) {
  Point<3> center_of_gravity;
  runner | [&]{
    return CENTER_OF_GRAVITY::center_of_gravity(host_image.volume, center_of_gravity);
  };
}

template<typename T>
__host__ dicomNodeError_t volume_difference_device(
u64 elements, Volume<3, T>* volume_1, Volume<3, T>* volume_2, f32& error) noexcept {
  return reduce_no_mem<8, VolumeDifference<T>, T>(
    elements,
    &error,
    volume_1,
    volume_2
  );
}

constexpr Space<3> translate_space(
  Space<3> space, // VERY INTENTIONAL NOT A &
  const ImageTranslation& translation
) {

  const SquareMatrix<3> scale {
    translation.scale, 0.0f, 0.0f,
    0.0f, translation.scale, 0.0f,
    0.0f, 0.0f, translation.scale
  };

  const f32 cos_x = cos(translation.rotations[0]);
  const f32 cos_y = cos(translation.rotations[1]);
  const f32 cos_z = cos(translation.rotations[2]);
  const f32 sin_x = sin(translation.rotations[0]);
  const f32 sin_y = sin(translation.rotations[1]);
  const f32 sin_z = sin(translation.rotations[2]);

  const SquareMatrix<3> rotation_x = {
    1.0,0.0,0.0,
    0.0, cos_x, sin_x,
    0.0, -sin_x, cos_x
  };

  const SquareMatrix<3> rotation_y = {
    cos_y, 0.0f, sin_y,
    0.0, 1.0, 0.0,
    -sin_y, 0.0, cos_y
  };

  const SquareMatrix<3> rotation_z = {
    cos_z, sin_y, 0.0f,
    -sin_z, cos_z, 0.0f,
    0.0f,0.0f,1.0f
  };

  space.basis *= scale * rotation_x * rotation_y * rotation_z;
  space.inverted_basis = space.basis.inverse();
  space.starting_point += translation.translations;

  return space;
}

/** Interpolates the volume used for volume comparison in the registration.
 *
 * @tparam T The type of Image
 * @param runner DicomRunner used to allocate the source image, destination Image and out volume
 * @param host_source_image The volume where data is gathered from
 * @param translation A translation of the source image
 * @param host_destination_space The Space that the translated image should be interpolated to
 * @param out_volume The volume that is interpolated into
 * @return
 */
template<typename T>
__host__ dicomNodeError_t interpolate_for_registration(
  DicomNodeRunner& runner,
  Image<3, T> host_source_image, // SHALLOW COPY!
  const ImageTranslation& translation,
  const Space<3>& host_destination_space,
  Volume<3, T>& out_volume
  ) {
  host_source_image.space = translate_space(host_source_image.space, translation);

  runner | [&]{
    return gpu_interpolation_linear_shared_cheating(
      host_source_image,
      host_destination_space,
      out_volume.data
    );
  };

  return runner.error();
}

  /** Interpolates the volume used for volume comparison in the registration.
 *
 * @tparam T The type of Image
 * @param runner DicomRunner used to allocate the source image, destination Image and out volume
 * @param host_source_image The volume where data is gathered from
 * @param translation A translation of the source image
 * @param host_destination_space The Space that the translated image should be interpolated to
 * @param out_volume The volume that is interpolated into
 * @return
 */
template<typename T>
__host__ dicomNodeError_t interpolate_for_registration(
    Image<3, T> host_source_image, // SHALLOW COPY!
    const ImageTranslation& translation,
    Space<3> host_destination_space, // I modify it, so I need a copy.
    Volume<3, T>& out_volume
    ) {

  //const auto offset_source = host_source_image.space.center_space();
  host_source_image.space = translate_space(host_source_image.space, translation);

  //const auto offset_destination = host_destination_space.center_space();

  dicomNodeError_t error = gpu_interpolation_linear_shared_cheating(
    host_source_image,
    host_destination_space,
    out_volume.data
  );

  return error;
}

template<typename T>
struct OptimizerParam {
  ImageTranslation affine;
  T cost = cuda::std::numeric_limits<T>::max();

    OptimizerParam& compare(OptimizerParam& other) noexcept {
      return cost < other.cost ? *this : other;
    };
};


/**
 *
 * @tparam T type of the Cost
 */
template<typename T>
struct Optimizer {
  static constexpr u32 max_iterations = 25;
  static constexpr u8 max_step_size_reductions = 4;

  OptimizerParam<T> best_params;
  ImageTranslation step_size {
      .scale = 0.025,
      .translations = Point<3>{8.0f,8.0f,8.0f},
      .rotations = {0.05f,0.05f,0.05f},
  };

  u32 iterations = 0;
  u8 step_size_reductions = 0;



  Optimizer(auto cost_function) {
    best_params.cost = cost_function(best_params);
  }

  T update_translation(auto cost_function) {
    OptimizerParam positive_param = best_params;
    OptimizerParam negative_param = best_params;

    for (u8 dim = 0; dim < 3; dim++) {
      positive_param.affine.translations[dim] += step_size.translations[dim];
      negative_param.affine.translations[dim] -= step_size.translations[dim];

      positive_param.cost = cost_function(positive_param);
      negative_param.cost = cost_function(negative_param);

      best_params = best_params.compare(positive_param);
      best_params = best_params.compare(negative_param);

      if (dim < 2) {
        positive_param = best_params;
        negative_param = best_params;
      }
    }

    return best_params.cost;
  }

  void reduce_step_sizes() {
    step_size.translations = step_size.translations / 2.0f;
    step_size_reductions++;
  }

  T optimize(auto cost_function) {
    const f32 current_cost = best_params.cost;

    const f32 translation_cost = update_translation(cost_function);

    std::printf("Current cost: %f, translation_cost %f\n" , current_cost, translation_cost);

    if ( translation_cost < current_cost) {
      return best_params.cost;
    }


    reduce_step_sizes();

    return best_params.cost;
  }

  /**
   * Determines if the optimization loop should continue
   *
   * @example while(optimizer.should_continue()){
   *  ... // Optimization loop
   * }
   * @return if you should continue the while loop
   */
  [[nodiscard]] bool should_continue() noexcept {
    return cost() > 0.0f && step_size_reductions < max_step_size_reductions && ++iterations < max_iterations;
  }

  [[nodiscard]] T cost() const noexcept {
    return best_params.cost;
  }
};

template<typename T>
dicomNodeError_t cost_function (
  DicomNodeRunner& runner,
  const Image<3, T>& host_source_image,
  const Image<3, T>& host_target_image,
  Volume<3, T>& temp_volume,
  Volume<3, T>* device_source_volume,
  Volume<3, T>* device_temp_volume,
  const OptimizerParam<T>& params,
  f32& out_cost
  ){

  runner | [&]() {
    return REGISTRATION::interpolate_for_registration<T>(
      host_target_image,
      params.affine,
      host_source_image.space,
      temp_volume
    );
  } | [&](){return REGISTRATION::volume_difference_device<T>(
    host_source_image.elements(),
    device_source_volume,
    device_temp_volume,
    out_cost
    );};

  return runner.error();
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
    // Initialization
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
      return CENTER_OF_GRAVITY::center_of_gravity(host_target_image.volume, destination_center_of_gravity);
    } | [&]() {
      // TODO: apply the basis here (i'm pretty, that it's correct, since they might be different)
      const Point<3> offset = destination_center_of_gravity - source_center_of_gravity;
      intermediate_image.space.starting_point += offset;

      return cudaMemcpy(device_intermediate_image, &intermediate_image, sizeof(Image<3, T>), cudaMemcpyDefault);
    };

    runner | [&]() {
      free_used_memory();
      return SUCCESS;
    };

    return runner.error();
  }
}
