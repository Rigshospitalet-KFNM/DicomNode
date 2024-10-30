#include <gtest/gtest.h>
#include"../gpu_code/dicom_node_gpu.cu"

#include<iostream>

template<uint8_t DIMENSION>
__global__ void swapRowKernel(
  volatile SquareMatrix<DIMENSION> *matrix,
  const uint8_t r1, const uint8_t r2
){
  swapRow<DIMENSION>(matrix, r1, r2);
}

TEST(LIN_ALG, ROW_SLAP){
  float matrix[] = {
     1.0f,  2.0f,  3.0f, 4.0f,  5.0f,
     6.0f,  7.0f,  8.0f, 9.0f, 10.0f,
    11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
    21.0f, 22.0f, 23.0f, 24.0f, 25.0f
  };

  SquareMatrix<5>* gpuMatrix;
  cudaError_t status = cudaMalloc(&gpuMatrix, sizeof(SquareMatrix<5>));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 25,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);
  swapRowKernel<5><<<1,1024>>>(gpuMatrix, 1, 3);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 25,
    cudaMemcpyDeviceToHost
  );

  ASSERT_FLOAT_EQ(matrix[0],  1.0f);
  ASSERT_FLOAT_EQ(matrix[5],  16.0f);
  ASSERT_FLOAT_EQ(matrix[10], 11.0f);
  ASSERT_FLOAT_EQ(matrix[15], 6.0f);
  ASSERT_FLOAT_EQ(matrix[20], 21.0f);

  cudaFree(gpuMatrix);
}

template<uint8_t DIMENSION>
__global__ void forwardEleminationKernel(
  volatile SquareMatrix<DIMENSION> *matrix,
  volatile Point<DIMENSION> *vector,
  dicomNodeError_t* error
){
  *error = ForwardElemination<DIMENSION>(matrix, vector);
}


TEST(LIN_ALG, FORWARD_ELEMINATION){
  float matrix[] = {
    1.0f, 1.0f, 3.0f,
    3.0f, 5.0f, 10.0f,
    5.0f, 7.0f, 17.0f,
  };

  float point[] = {
    4.0f, 0.0f, 23.0f
  };

  SquareMatrix<3>* gpuMatrix;
  Point<3>* gpuPoint;
  dicomNodeError_t* error;
  cudaError_t status = cudaMalloc(&gpuMatrix, sizeof(SquareMatrix<3>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuPoint, sizeof(Point<3>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(gpuPoint->points,
                      point,
                      sizeof(float) * 3,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);

  forwardEleminationKernel<3><<<1,9>>>(gpuMatrix, gpuPoint, error);
  status = cudaGetLastError();
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );

  status = cudaMemcpy(
    point,
    gpuPoint->points,
    sizeof(float) * 3,
    cudaMemcpyDeviceToHost
  );

  ASSERT_FLOAT_EQ(point[0], 4.0f);
  ASSERT_FLOAT_EQ(point[1], -12.0f);
  ASSERT_FLOAT_EQ(point[2], 15.0f);

  ASSERT_FLOAT_EQ(matrix[0], 1.0f);
  ASSERT_FLOAT_EQ(matrix[1], 1.0f);
  ASSERT_FLOAT_EQ(matrix[2], 3.0f);
  ASSERT_FLOAT_EQ(matrix[3], 0.0f);
  ASSERT_FLOAT_EQ(matrix[4], 2.0f);
  ASSERT_FLOAT_EQ(matrix[5], 1.0f);
  ASSERT_FLOAT_EQ(matrix[6], 0.0f);
  ASSERT_FLOAT_EQ(matrix[7], 0.0f);
  ASSERT_FLOAT_EQ(matrix[8], 1.0f);
}


TEST(LIN_ALG, FORWARD_ELEMINATION_2x2){
  float matrix[] = {
    2.0f, 3.0f,
    4.0f, 5.0f
  };

  float point[] = {
    6.0f, 7.0f
  };

  SquareMatrix<2>* gpuMatrix;
  Point<2>* gpuPoint;
  dicomNodeError_t* error;
  cudaError_t status = cudaMalloc(&gpuMatrix, sizeof(SquareMatrix<2>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuPoint, sizeof(Point<2>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 4,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(gpuPoint->points,
                      point,
                      sizeof(float) * 2,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);

  forwardEleminationKernel<2><<<1,4>>>(gpuMatrix, gpuPoint, error);
  status = cudaGetLastError();
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 4,
    cudaMemcpyDeviceToHost
  );

  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    point,
    gpuPoint->points,
    sizeof(float) * 2,
    cudaMemcpyDeviceToHost
  );

  ASSERT_FLOAT_EQ(point[0], 6.0f);
  ASSERT_FLOAT_EQ(point[1], -5.0f);

  ASSERT_FLOAT_EQ(matrix[0], 2.0f);
  ASSERT_FLOAT_EQ(matrix[1], 3.0f);
  ASSERT_FLOAT_EQ(matrix[2], 0.0f);
  ASSERT_FLOAT_EQ(matrix[3], -1.0f);
}


TEST(LIN_ALG, FORWARD_ELEMINATION_SWAP){
  float matrix[] = {
    0.0f,  1.0f, 2.0f,
    1.0f,  3.0f, 5.0f,
    -1.0f, 2.0f, -12.0f
  };

  float point[] = {
    1.0f, 2.0f, 3.0f
  };
  dicomNodeError_t* error;
  SquareMatrix<3>* gpuMatrix;
  Point<3>* gpuPoint;
  cudaError_t status = cudaMalloc(&gpuMatrix, sizeof(SquareMatrix<3>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuPoint, sizeof(Point<3>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(gpuPoint->points,
                      point,
                      sizeof(float) * 3,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);

  forwardEleminationKernel<3><<<1,9>>>(gpuMatrix, gpuPoint, error);
  status = cudaGetLastError();
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(
    point,
    gpuPoint->points,
    sizeof(float) * 3,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  ASSERT_EQ(hostError, dicomNodeError_t::SUCCESS);
}

template<uint8_t DIMENSION>
__global__ void full_reduction_kernel(
  volatile SquareMatrix<DIMENSION>* matrix,
  volatile Point<DIMENSION>* point,
  dicomNodeError_t *error
){
  *error = GaussJordanElemination<DIMENSION>(matrix, point);
}


TEST(LIN_ALG, FULL_REDUCTION){
  float matrix[] = {
    1.0f, 1.0f, 3.0f,
    3.0f, 5.0f, 10.0f,
    5.0f, 7.0f, 17.0f,
  };

  float point[] = {
    4.0f, 0.0f, 23.0f
  };

    SquareMatrix<3>* gpuMatrix;
  Point<3>* gpuPoint;
  dicomNodeError_t* error;
  cudaError_t status = cudaMalloc(&gpuMatrix, sizeof(SquareMatrix<3>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuPoint, sizeof(Point<3>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(gpuPoint->points,
                      point,
                      sizeof(float) * 3,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);

  full_reduction_kernel<3><<<1,1024>>>(gpuMatrix, gpuPoint, error);
  status = cudaGetLastError();
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(
    point,
    gpuPoint->points,
    sizeof(float) * 3,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  ASSERT_EQ(hostError, dicomNodeError_t::SUCCESS);

  ASSERT_FLOAT_EQ(matrix[0], 1.0f);
  ASSERT_FLOAT_EQ(matrix[1], 0.0f);
  ASSERT_FLOAT_EQ(matrix[2], 0.0f);
  ASSERT_FLOAT_EQ(matrix[3], 0.0f);
  ASSERT_FLOAT_EQ(matrix[4], 1.0f);
  ASSERT_FLOAT_EQ(matrix[5], 0.0f);
  ASSERT_FLOAT_EQ(matrix[6], 0.0f);
  ASSERT_FLOAT_EQ(matrix[7], 0.0f);
  ASSERT_FLOAT_EQ(matrix[8], 1.0f);

  ASSERT_FLOAT_EQ(point[0], -27.5f);
  ASSERT_FLOAT_EQ(point[1], -13.5f);
  ASSERT_FLOAT_EQ(point[2],  15.0f);
}

template<uint8_t DIMENSIONS>
__global__ void FORWARD_INVERSION_KERNEL(
  volatile SquareMatrix<DIMENSIONS>* input,
  volatile SquareMatrix<DIMENSIONS>* output,
  volatile dicomNodeError_t *error
){
  *error = _invertMatrixForward<DIMENSIONS>(*input, *output);
}

TEST(LIN_ALG, FORWARD_INVERSION){
  float matrix[] = {
    2.0f, 3.0f, 1.0f,
    1.0f, 1.0f, 2.0f,
    2.0f, 3.0f, 4.0f
  };

  SquareMatrix<3>* gpuMatrix;
  SquareMatrix<3>* gpuMatrixOutput;


  dicomNodeError_t* error;
  cudaError_t status = cudaMalloc(&gpuMatrix, sizeof(SquareMatrix<3>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuMatrixOutput, sizeof(SquareMatrix<3>));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);

  FORWARD_INVERSION_KERNEL<3><<<1,1024>>>(gpuMatrix, gpuMatrixOutput, error);
  status = cudaGetLastError();
  ASSERT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);
  ASSERT_EQ(hostError, dicomNodeError_t::SUCCESS);

  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  ASSERT_FLOAT_EQ(matrix[0], 2.0f);
  ASSERT_FLOAT_EQ(matrix[1], 3.0f);
  ASSERT_FLOAT_EQ(matrix[2], 1.0f);
  ASSERT_FLOAT_EQ(matrix[3], 0.0f);
  ASSERT_FLOAT_EQ(matrix[4], -0.5f);
  ASSERT_FLOAT_EQ(matrix[5], 1.5f);
  ASSERT_FLOAT_EQ(matrix[6], 0.0f);
  ASSERT_FLOAT_EQ(matrix[7], 0.0f);
  ASSERT_FLOAT_EQ(matrix[8], 3.0f);

  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrixOutput->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  ASSERT_FLOAT_EQ(matrix[0],  1.0f);
  ASSERT_FLOAT_EQ(matrix[1],  0.0f);
  ASSERT_FLOAT_EQ(matrix[2],  0.0f);
  ASSERT_FLOAT_EQ(matrix[3], -0.5f);
  ASSERT_FLOAT_EQ(matrix[4],  1.0f);
  ASSERT_FLOAT_EQ(matrix[5],  0.0f);
  ASSERT_FLOAT_EQ(matrix[6], -1.0f);
  ASSERT_FLOAT_EQ(matrix[7],  0.0f);
  ASSERT_FLOAT_EQ(matrix[8],  1.0f);
}

template<uint8_t DIMENSION>
__global__ void matrix_inversion(
  volatile SquareMatrix<DIMENSION>* matrix,
  volatile SquareMatrix<DIMENSION>* output,
  dicomNodeError_t *error
){
  *error = invertMatrix<DIMENSION>(*matrix, *output);
}

TEST(LIN_ALG, INVERSION_2x2){
  constexpr uint8_t DIM = 2;
  constexpr uint8_t DIM_SQ = DIM * DIM;
  float matrix[] = {
    3.0f, 4.0f, 0.0f, 2.0f
  };

  SquareMatrix<DIM>* gpuMatrix;
  SquareMatrix<DIM>* gpuMatrixOutput;


  dicomNodeError_t* error;
  cudaError_t status = cudaMalloc(&gpuMatrix, sizeof(SquareMatrix<DIM>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuMatrixOutput, sizeof(SquareMatrix<DIM>));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * DIM_SQ,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);

  matrix_inversion<DIM><<<1,1024>>>(gpuMatrix, gpuMatrixOutput, error);
  status = cudaGetLastError();
  ASSERT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);
  ASSERT_EQ(hostError, dicomNodeError_t::SUCCESS);

  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * DIM_SQ,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  for(int i = 0; i < DIM; i++){
    for(int j = 0; j < DIM; j++){
      std::cout << matrix[DIM * i + j];
      if(j != DIM - 1){
        std::cout << ", ";
      }
    }
    std::cout << "\n";
  }

  /*
  ASSERT_FLOAT_EQ(matrix[0], 1.0f);
  ASSERT_FLOAT_EQ(matrix[1], 0.0f);
  ASSERT_FLOAT_EQ(matrix[2], 0.0f);
  ASSERT_FLOAT_EQ(matrix[3], 0.0f);
  ASSERT_FLOAT_EQ(matrix[4], 1.0f);
  ASSERT_FLOAT_EQ(matrix[5], 0.0f);
  ASSERT_FLOAT_EQ(matrix[6], 0.0f);
  ASSERT_FLOAT_EQ(matrix[7], 0.0f);
  ASSERT_FLOAT_EQ(matrix[8], 1.0f);
  */

  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrixOutput->points,
    sizeof(float) * DIM_SQ,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  for(int i = 0; i < DIM; i++){
    for(int j = 0; j < DIM; j++){
      std::cout << matrix[DIM * i + j];
      if(j != DIM - 1){
        std::cout << ", ";
      }
    }
    std::cout << "\n";
  }
/*
  ASSERT_FLOAT_EQ(matrix[0],  1.0f);
  ASSERT_FLOAT_EQ(matrix[1],  0.0f);
  ASSERT_FLOAT_EQ(matrix[2],  0.0f);
  ASSERT_FLOAT_EQ(matrix[3], -0.5f);
  ASSERT_FLOAT_EQ(matrix[4],  1.0f);
  ASSERT_FLOAT_EQ(matrix[5],  0.0f);
  ASSERT_FLOAT_EQ(matrix[6], -1.0f);
  ASSERT_FLOAT_EQ(matrix[7],  0.0f);
  ASSERT_FLOAT_EQ(matrix[8],  1.0f);

*/
}

TEST(LIN_ALG, INVERSION_3x3){
  float matrix[] = {
    2.0f, 3.0f, 1.0f,
    1.0f, 1.0f, 2.0f,
    2.0f, 3.0f, 4.0f
  };

  SquareMatrix<3>* gpuMatrix;
  SquareMatrix<3>* gpuMatrixOutput;


  dicomNodeError_t* error;
  cudaError_t status = cudaMalloc(&gpuMatrix, sizeof(SquareMatrix<3>));
  ASSERT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuMatrixOutput, sizeof(SquareMatrix<3>));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  ASSERT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  ASSERT_EQ(status, cudaSuccess);

  matrix_inversion<3><<<1,1024>>>(gpuMatrix, gpuMatrixOutput, error);
  status = cudaGetLastError();
  ASSERT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);
  ASSERT_EQ(hostError, dicomNodeError_t::SUCCESS);

  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      std::cout << matrix[3 * i + j];
      if(j != 2){
        std::cout << ", ";
      }
    }
    std::cout << "\n";
  }

  /*
  ASSERT_FLOAT_EQ(matrix[0], 1.0f);
  ASSERT_FLOAT_EQ(matrix[1], 0.0f);
  ASSERT_FLOAT_EQ(matrix[2], 0.0f);
  ASSERT_FLOAT_EQ(matrix[3], 0.0f);
  ASSERT_FLOAT_EQ(matrix[4], 1.0f);
  ASSERT_FLOAT_EQ(matrix[5], 0.0f);
  ASSERT_FLOAT_EQ(matrix[6], 0.0f);
  ASSERT_FLOAT_EQ(matrix[7], 0.0f);
  ASSERT_FLOAT_EQ(matrix[8], 1.0f);
  */

  ASSERT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrixOutput->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  ASSERT_EQ(status, cudaSuccess);

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      std::cout << matrix[3 * i + j];
      if(j != 2){
        std::cout << ", ";
      }
    }
    std::cout << "\n";
  }
/*
  ASSERT_FLOAT_EQ(matrix[0],  1.0f);
  ASSERT_FLOAT_EQ(matrix[1],  0.0f);
  ASSERT_FLOAT_EQ(matrix[2],  0.0f);
  ASSERT_FLOAT_EQ(matrix[3], -0.5f);
  ASSERT_FLOAT_EQ(matrix[4],  1.0f);
  ASSERT_FLOAT_EQ(matrix[5],  0.0f);
  ASSERT_FLOAT_EQ(matrix[6], -1.0f);
  ASSERT_FLOAT_EQ(matrix[7],  0.0f);
  ASSERT_FLOAT_EQ(matrix[8],  1.0f);

*/
}