#include "stdint.h"

#include <iostream>

#include "constants.cu"
#include "classes/image.cu"

#define ROWS 512
#define COLS 512

typedef std::uint8_t image_type;

// Passing function pointers to Kernels
// https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/
template<typename T>
__device__ T half_circle_draw_function(int x, int y, int rows, int cols){
  const int center_x = cols / 2;
  const int center_y = rows / 2;

  const float radius = float(min(center_x, center_y)) / 2;
  const float distance_to_center = sqrtf(powf(float(x - center_x), 2) + powf(float(y - center_y), 2));

  return center_y >= y && distance_to_center < radius;
}

// Yes this is Assignment is needed
template<typename T>
__device__ playground::kernel::draw_function_t<T> p_draw_function = half_circle_draw_function<T>;

int main(int argc, char* argv[]){
  std::cout << "Start!\n";
  playground::Image<image_type> image(ROWS, COLS);

  // A pointer to the function, is needed on the host side,
  //since the host cannot manipulate device side (function) pointers.
  playground::kernel::draw_function_t<image_type> draw_function_host;
  CUDA_RT_CALL(
    cudaMemcpyFromSymbol(
      &draw_function_host,
      p_draw_function<image_type>,
      sizeof(playground::kernel::draw_function_t<image_type>)
    )
  );
  image.draw_image(draw_function_host);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  image.print("TestFile.bmp");
  CUDA_RT_CALL(cudaDeviceSynchronize());

  image.find_contours();

  std::cout << "Done\n";

  return 0;
}