#pragma once

#include"core/core.cuh"

#include"map_reduce.cuh"

namespace REGISTRATION {
    inline __device__ f32 distance_between() {
        return 0.0f;
    }



    inline dicomNodeError_t register_to(
        Image<3, f32>* source_image,
        Image<3, f32>* destination_image,
        f32* gpu_intermediate
    ) {

        return dicomNodeError_t::SUCCESS;
    }



}
