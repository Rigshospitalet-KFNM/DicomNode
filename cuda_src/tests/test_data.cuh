#pragma once

#include "../gpu_code/core/core.cuh"

namespace TEST_DATA {
  constexpr Space<3> make_spaces() {
    return {
      .starting_point = Point<3>{1.0f,1.0f, 1.0f},
      .basis = {
            1.0f,0.0f,0.0f,
            0.0f,1.0f,0.0f,
            0.0f,0.0f,1.0f
          },
      .inverted_basis = {
        1.0f,0.0f,0.0f,
        0.0f,1.0f,0.0f,
        0.0f,0.0f,1.0f
        },
    };
  }
}

