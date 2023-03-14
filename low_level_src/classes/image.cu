#ifndef PLAYGROUND_CLASSES_IMAGE_H
#define PLAYGROUND_CLASSES_IMAGE_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <fstream>
#include <iostream>
#include <string>

#include "../constants.cu"

namespace playground {
  namespace kernel {
    enum Direction {
      LEFT,
      TOP_LEFT,
      TOP,
      TOP_RIGHT,
      RIGHT,
      BOTTOM_RIGHT,
      BOTTOM,
      BOTTOM_LEFT
    };

    template<typename T>
    using draw_function_t = T (*) (int, int, int, int);

    template<typename T>
    __device__ draw_function_t<T> draw_function;

    struct Triad {
      int current;
      int previous;
      int next;
    };

    template<typename T>
    struct DeviceImage {
      int rows;
      int cols;
      T* image;

      __device__ inline bool in_image(const Index index) const {
        return 0 <= index.x && index.x < cols && 0 <= index.y && index.y < rows;
      }

      __device__ inline bool in_image(const int y, const int x) const {
        return 0 <= x && x < cols && 0 <= y && y < rows;
      }

      __device__ T& operator[](const Index index) const {
        assert(in_image(index));

        return image[index.y * cols + index.x];
      }
    };

    template<typename T>
    class Rectangle {
      T* image;
      const int rectangle_size = BLOCKSIZE + border_size * 2;

      public:
        const static int border_size = 1;
        __device__ Rectangle(const DeviceImage<T> device_image, T shared_memory_pointer[]){
          image = shared_memory_pointer;
          const int x_offset = blockDim.x * blockIdx.x;
          const int y_offset = blockDim.y * blockIdx.y;

          assert(BLOCKSIZE == blockDim.x);

          for(int i = -border_size; i < BLOCKSIZE + border_size; i++){
            Index loop_index = {y_offset + i, x_offset + (int)threadIdx.x - border_size};
            image[(i + border_size) * rectangle_size + threadIdx.x] = (device_image.in_image(loop_index)) ? device_image[loop_index] : 0;
          }


          const int y_minor_offset = threadIdx.x / (2 * border_size) - border_size;
          const int x_minor_offset = threadIdx.x % (2 * border_size);

          const int shared_x_offset = x_minor_offset + BLOCKSIZE;
          const int edge_x_offset = x_offset + x_minor_offset + BLOCKSIZE;

          Index edge_offset_1 = {y_offset + y_minor_offset, edge_x_offset };
          Index edge_offset_2 = {y_offset + y_minor_offset + (int)blockDim.x / 2, edge_x_offset};

          image[(y_minor_offset + border_size) * rectangle_size + shared_x_offset] = (device_image.in_image(edge_offset_1)) ? device_image[edge_offset_1] : 0;
          image[(y_minor_offset + border_size + blockDim.x / 2) * rectangle_size + shared_x_offset] = (device_image.in_image(edge_offset_2)) ? device_image[edge_offset_2] : 0;

          if(threadIdx.x < (2 * border_size) * (2 * border_size)){
            Index final_index = {y_minor_offset + border_size + (int)blockDim.x, edge_x_offset };
            image[(y_minor_offset + border_size + blockDim.x) * rectangle_size + shared_x_offset] = device_image.in_image(final_index) ? device_image[final_index] : 0;
          }

          __syncthreads(); // pointless since there's only 1 threadgroup per block. However this might change if you move to 64 blocks
        };

        __device__ ~Rectangle(){};

        __device__ inline bool in_rectangle(const Index index) const {
          return -border_size <= index.y && index.y < BLOCKSIZE + border_size
              && -border_size <= index.x && index.x < BLOCKSIZE + border_size;
        }

        __device__ T& operator[](const Index index) const {
          assert(in_rectangle(index));
          return image[(index.y + border_size) * rectangle_size + index.x + border_size];
        };
    }; // End Rectangle

    class contours {};

    template<typename T>
    __device__ Triad find_triad(
        const Rectangle<T> image,
        int j,
        int i,
        Direction starting_direction=Direction::LEFT
      ){
      return {0,0,0};
    }

    /**
     * @brief Draws a picture on device_image using a GPU.
     *
     * @note One thread per pixel should be called.
     * @tparam T type of the image data
     * @param device_image image data
     * @param draw_function function to determine the draw
     * @return __global__
     */
    template<typename T>
    __global__ void draw_image_kernel(DeviceImage<T> device_image, draw_function_t<T> draw_function){
      const int x = blockDim.x * blockIdx.x + threadIdx.x;
      const int y = blockDim.y * blockIdx.y + threadIdx.y;

      if (x < device_image.cols && y < device_image.rows){
          device_image[{y,x}] = (device_image.in_image({y,x})) ?
            draw_function(x,y, device_image.rows, device_image.cols) : 0;
        }
      }


    template<typename T>
    __global__ void preprocess(
        DeviceImage<T> device_image,
        DeviceImage<uint8_t> border_image
      ){
      const int x = blockDim.x * blockIdx.x + threadIdx.x;
      const int y = blockDim.y * blockIdx.y + threadIdx.y;

      if (x < device_image.cols && y < device_image.rows){
        uint8_t border = 0;

        if (device_image[{y,x}]){
          if(x - 1 >= 0){
            border = border || (device_image[{y, x - 1}] == 0);
          }
          if(x + 1 < device_image.cols){
            border = border || (device_image[{y, x + 1}] == 0);
          }
          if(y - 1 >= 0){
            border = border || (device_image[{y - 1, x}] == 0);
          }
          if(y + 1 < device_image.rows){
            border = border || (device_image[{y + 1, x}] == 0);
          }
        }
        border_image[{y,x}] = border;
      }
    }

    template<typename T>
    __global__ void track_borders_in_rectangle(DeviceImage<T> device_image, DeviceImage<uint8_t> border){
      __shared__ T shared_memory_image[(BLOCKSIZE + Rectangle<T>::border_size * 2) * (BLOCKSIZE + Rectangle<T>::border_size * 2)];
      __shared__ T shared_memory_border[(BLOCKSIZE + Rectangle<T>::border_size * 2) * (BLOCKSIZE + Rectangle<T>::border_size * 2)];

      Rectangle<T> rectangle(device_image, shared_memory_image);
      Rectangle<T> border_rectangle(border, shared_memory_border);

      if(threadIdx.x == 0){
        for(int j = 0; j < BLOCKSIZE; j++){
          for(int i = 0; i < BLOCKSIZE; i++){
            if(border_rectangle[{j,i}]){
              
            }
          }
        }
      }
    }
  } // End namespace kernels


  template<typename T>
  class Image {
    kernel::DeviceImage<T>* device_image;
    dim3 block_drawing = dim3(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 block_contours = dim3(BLOCKSIZE,1,1);
    dim3 grid;

    public:
      Image(int rows_in, int cols_in){
        CUDA_RT_CALL(cudaMallocManaged(&device_image, sizeof(kernel::DeviceImage<T>)));
        CUDA_RT_CALL(cudaMallocManaged(&device_image->image, rows_in*cols_in*sizeof(T)));
        CUDA_RT_CALL(cudaDeviceSynchronize());
        device_image->rows=rows_in;
        device_image->cols=cols_in;

        const int grid_x = (device_image->cols % BLOCKSIZE) ? device_image->cols / BLOCKSIZE + 1 : device_image->cols / BLOCKSIZE;
        const int grid_y = (device_image->rows % BLOCKSIZE) ? device_image->rows / BLOCKSIZE + 1 : device_image->rows / BLOCKSIZE;
        grid = dim3(grid_x, grid_y, 1);
      }

      ~Image(){
        cudaFree(device_image->image);
        cudaFree(device_image);
      }

      void print(const std::string filepath){
        std::ofstream file(filepath);

        file << "P3\n";
        file << device_image->cols << " " << device_image->rows << "\n255\n";

        for(int idx = 0; idx < device_image->rows * device_image->cols; idx++){
          if(device_image->image[idx]){
            file << "255 255 255\n";
          } else {
            file << "0 0 0\n";
          }
        }
        file.close();
      }

      void draw_image(kernel::draw_function_t<T> draw_function){
        kernel::draw_image_kernel<T><<<grid, block_drawing>>>(*device_image, draw_function);
        CUDA_RT_CALL(cudaDeviceSynchronize());
      }

      void find_contours(){
        kernel::DeviceImage<uint8_t>* border;
        CUDA_RT_CALL(cudaMallocManaged(&border, sizeof(kernel::DeviceImage<uint8_t>)));
        CUDA_RT_CALL(cudaMallocManaged(&border->image, device_image->cols * device_image->rows * sizeof(uint8_t)));
        border->cols = device_image->cols;
        border->rows = device_image->rows;

        std::cout << "Preprocessing!\n";
        kernel::preprocess<T><<<grid, block_drawing>>>(*device_image, *border);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        std::cout << "Tracking!\n";
        kernel::track_borders_in_rectangle<T><<<grid, block_contours>>>(*device_image, *border);
        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaFree(border->image));
        CUDA_RT_CALL(cudaFree(border));
      }
  };

  template<typename T>
  void print(kernel::DeviceImage<T> device_image, const std::string filepath){
    std::ofstream file(filepath);

    file << "P3\n";
    file << device_image.cols << " " << device_image.rows << "\n255\n";

    for(int idx = 0; idx < device_image.rows * device_image.cols; idx++){
      if(device_image.image[idx]){
        file << "255 255 255\n";
      } else {
        file << "0 0 0\n";
      }
      }
    file.close();
  }
}

#endif
