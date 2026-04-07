#pragma once

#include"core/core.cuh"
#include"linear_interpolation.cuh"
#include"map_reduce.cuh"

namespace REGISTRATION {
    template<typename T>
    struct IMAGE_DIFFERENCE {
        static __device__ __host__ T map_to(u64 global_index, const Image<3, T>* image_1, const Image<3, T>* image_2) {
            if (min(image_1->elements(), image_2->elements()) <= global_index){
                return identity();
            }

            T t = image_1->data()[global_index] - image_2->data()[global_index]; // This is a problem for unsigned
            return t < 0 ? -t : t; // I can't find good docs on abs...
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
        Image<3, T>& host_registration_target_image
    ) {
        const bool source_is_smallest = host_source_image.space.elements() < host_registration_target_image.space.elements();
        Image<3,T>& smallest_image =  source_is_smallest ? host_source_image : host_registration_target_image;
        Space<3>& smallest_space = smallest_image.space;

        Image<3,T> intermediate_image(
            smallest_space, Volume<3, T>{
                .data = nullptr,
                .m_extent = smallest_space.extent,
                .default_value = 0
            }
        );

        T difference = 0;

        Image<3, T>* device_source_image = nullptr;
        Image<3, T>* device_destination_image = nullptr;
        Image<3, T>* device_intermediate_image = nullptr;
        // This pointer is a reference, so it shouldn't be freed
        Image<3, T>* largest_image_ptr = nullptr;
        Image<3, T>* smallest_image_ptr = nullptr;


        DicomNodeRunner runner{[&](dicomNodeError_t error) {
            free_device_memory(
                &intermediate_image.data(),
                &device_source_image,
                &device_destination_image,
                &device_intermediate_image
            );
        }};

        runner | [&]() {
            return cudaMalloc(&(intermediate_image.data()), smallest_image.size());
        } | [&]() {
            return cudaMalloc(&(device_source_image), sizeof(Image<3, T>));
        } | [&]() {
            return cudaMalloc(&(device_destination_image), sizeof(Image<3, T>));
        } | [&]() {
            return cudaMalloc(&(device_intermediate_image), sizeof(Image<3, T>));
        } | [&]() {
            return cudaMemcpy(device_source_image, &host_source_image, sizeof(Image<3, T>), cudaMemcpyDefault);
        } | [&]() {
            return cudaMemcpy(device_destination_image, &host_registration_target_image, sizeof(Image<3, T>), cudaMemcpyDefault);
        } | [&]() {
            return cudaMemcpy(device_intermediate_image, &intermediate_image, sizeof(Image<3, T>), cudaMemcpyDefault);
        } | [&]() {
            largest_image_ptr  = source_is_smallest ? device_destination_image : device_source_image;

            constexpr static dim3 THREAD_BLOCK = THREAD_BLOCK_3D;
            const dim3 envelope = get_envelope_grid<THREAD_BLOCK>(intermediate_image.extent());

            INTERPOLATION::kernel_interpolation_linear_shared<<<envelope, THREAD_BLOCK>>>(
                largest_image_ptr,
                &(device_intermediate_image->space),
                intermediate_image.data()
            );

            return cudaGetLastError();
        } | [&]() {
            smallest_image_ptr = !source_is_smallest ? device_destination_image : device_source_image;
            return reduce_no_mem<8, IMAGE_DIFFERENCE<T>, T>(
                smallest_image.elements(),
                &difference,
                device_intermediate_image,
                smallest_image_ptr
            );
        } | [&]() {
            free_device_memory(
                &intermediate_image.data(),
                &device_source_image,
                &device_destination_image,
                &device_intermediate_image
            );
            return dicomNodeError_t::SUCCESS;
        };

        std::cout << "Difference: " << difference << "\n";

        return dicomNodeError_t::SUCCESS;
    }



}
