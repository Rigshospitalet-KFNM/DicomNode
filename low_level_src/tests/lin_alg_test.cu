#include <gtest/gtest.h>
#include"../gpu_code/dicom_node_gpu.cuh"

#include<iostream>

constexpr float EPSILON = 0.0005f;

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
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 25,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);
  swapRowKernel<5><<<1,1024>>>(gpuMatrix, 1, 3);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 25,
    cudaMemcpyDeviceToHost
  );

  EXPECT_FLOAT_EQ(matrix[0],  1.0f);
  EXPECT_FLOAT_EQ(matrix[5],  16.0f);
  EXPECT_FLOAT_EQ(matrix[10], 11.0f);
  EXPECT_FLOAT_EQ(matrix[15], 6.0f);
  EXPECT_FLOAT_EQ(matrix[20], 21.0f);

  cudaFree(gpuMatrix);
}

template<uint8_t DIMENSION>
__global__ void forwardEliminationKernel(
  volatile SquareMatrix<DIMENSION> *matrix,
  volatile Point<DIMENSION> *vector,
  dicomNodeError_t* error
){
  *error = ForwardElimination<DIMENSION>(matrix, vector);
}


TEST(LIN_ALG, FORWARD_ELIMINATION){
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
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuPoint, sizeof(Point<3>));
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(gpuPoint->points,
                      point,
                      sizeof(float) * 3,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);

  forwardEliminationKernel<3><<<1,9>>>(gpuMatrix, gpuPoint, error);
  status = cudaGetLastError();
  EXPECT_EQ(status, cudaSuccess);
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

  EXPECT_FLOAT_EQ(point[0], 4.0f);
  EXPECT_FLOAT_EQ(point[1], -12.0f);
  EXPECT_FLOAT_EQ(point[2], 15.0f);

  EXPECT_FLOAT_EQ(matrix[0], 1.0f);
  EXPECT_FLOAT_EQ(matrix[1], 1.0f);
  EXPECT_FLOAT_EQ(matrix[2], 3.0f);
  EXPECT_FLOAT_EQ(matrix[3], 0.0f);
  EXPECT_FLOAT_EQ(matrix[4], 2.0f);
  EXPECT_FLOAT_EQ(matrix[5], 1.0f);
  EXPECT_FLOAT_EQ(matrix[6], 0.0f);
  EXPECT_FLOAT_EQ(matrix[7], 0.0f);
  EXPECT_FLOAT_EQ(matrix[8], 1.0f);

  cudaFree(gpuMatrix);
  cudaFree(gpuPoint);
  cudaFree(error);
}


TEST(LIN_ALG, FORWARD_ELIMINATION_2x2){
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
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuPoint, sizeof(Point<2>));
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 4,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(gpuPoint->points,
                      point,
                      sizeof(float) * 2,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);

  forwardEliminationKernel<2><<<1,4>>>(gpuMatrix, gpuPoint, error);
  status = cudaGetLastError();
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 4,
    cudaMemcpyDeviceToHost
  );

  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    point,
    gpuPoint->points,
    sizeof(float) * 2,
    cudaMemcpyDeviceToHost
  );

  EXPECT_FLOAT_EQ(point[0], 6.0f);
  EXPECT_FLOAT_EQ(point[1], -5.0f);

  EXPECT_FLOAT_EQ(matrix[0], 2.0f);
  EXPECT_FLOAT_EQ(matrix[1], 3.0f);
  EXPECT_FLOAT_EQ(matrix[2], 0.0f);
  EXPECT_FLOAT_EQ(matrix[3], -1.0f);

  cudaFree(gpuMatrix);
  cudaFree(gpuPoint);
  cudaFree(error);
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
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuPoint, sizeof(Point<3>));
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(gpuPoint->points,
                      point,
                      sizeof(float) * 3,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);

  forwardEliminationKernel<3><<<1,9>>>(gpuMatrix, gpuPoint, error);
  status = cudaGetLastError();
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(
    point,
    gpuPoint->points,
    sizeof(float) * 3,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  EXPECT_EQ(hostError, dicomNodeError_t::SUCCESS);

  cudaFree(gpuMatrix);
  cudaFree(gpuPoint);
  cudaFree(error);
}

template<uint8_t DIMENSION>
__global__ void full_reduction_kernel(
  volatile SquareMatrix<DIMENSION>* matrix,
  volatile Point<DIMENSION>* point,
  dicomNodeError_t *error
){
  *error = GaussJordanElimination<DIMENSION>(matrix, point);
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
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuPoint, sizeof(Point<3>));
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(gpuPoint->points,
                      point,
                      sizeof(float) * 3,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);

  full_reduction_kernel<3><<<1,1024>>>(gpuMatrix, gpuPoint, error);
  status = cudaGetLastError();
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(
    point,
    gpuPoint->points,
    sizeof(float) * 3,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  EXPECT_EQ(hostError, dicomNodeError_t::SUCCESS);

  EXPECT_FLOAT_EQ(matrix[0], 1.0f);
  EXPECT_FLOAT_EQ(matrix[1], 0.0f);
  EXPECT_FLOAT_EQ(matrix[2], 0.0f);
  EXPECT_FLOAT_EQ(matrix[3], 0.0f);
  EXPECT_FLOAT_EQ(matrix[4], 1.0f);
  EXPECT_FLOAT_EQ(matrix[5], 0.0f);
  EXPECT_FLOAT_EQ(matrix[6], 0.0f);
  EXPECT_FLOAT_EQ(matrix[7], 0.0f);
  EXPECT_FLOAT_EQ(matrix[8], 1.0f);

  EXPECT_FLOAT_EQ(point[0], -27.5f);
  EXPECT_FLOAT_EQ(point[1], -13.5f);
  EXPECT_FLOAT_EQ(point[2],  15.0f);

  cudaFree(error);
  cudaFree(gpuMatrix);
  cudaFree(gpuPoint);
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
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuMatrixOutput, sizeof(SquareMatrix<3>));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);

  FORWARD_INVERSION_KERNEL<3><<<1,1024>>>(gpuMatrix, gpuMatrixOutput, error);
  status = cudaGetLastError();
  EXPECT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);
  EXPECT_EQ(hostError, dicomNodeError_t::SUCCESS);

  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  EXPECT_FLOAT_EQ(matrix[0], 2.0f);
  EXPECT_FLOAT_EQ(matrix[1], 3.0f);
  EXPECT_FLOAT_EQ(matrix[2], 1.0f);
  EXPECT_FLOAT_EQ(matrix[3], 0.0f);
  EXPECT_FLOAT_EQ(matrix[4], -0.5f);
  EXPECT_FLOAT_EQ(matrix[5], 1.5f);
  EXPECT_FLOAT_EQ(matrix[6], 0.0f);
  EXPECT_FLOAT_EQ(matrix[7], 0.0f);
  EXPECT_FLOAT_EQ(matrix[8], 3.0f);

  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrixOutput->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  EXPECT_FLOAT_EQ(matrix[0],  1.0f);
  EXPECT_FLOAT_EQ(matrix[1],  0.0f);
  EXPECT_FLOAT_EQ(matrix[2],  0.0f);
  EXPECT_FLOAT_EQ(matrix[3], -0.5f);
  EXPECT_FLOAT_EQ(matrix[4],  1.0f);
  EXPECT_FLOAT_EQ(matrix[5],  0.0f);
  EXPECT_FLOAT_EQ(matrix[6], -1.0f);
  EXPECT_FLOAT_EQ(matrix[7],  0.0f);
  EXPECT_FLOAT_EQ(matrix[8],  1.0f);

  cudaFree(error);
  cudaFree(gpuMatrix);
  cudaFree(gpuMatrixOutput);
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
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuMatrixOutput, sizeof(SquareMatrix<DIM>));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * DIM_SQ,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);

  matrix_inversion<DIM><<<1,1024>>>(gpuMatrix, gpuMatrixOutput, error);
  status = cudaGetLastError();
  EXPECT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);
  EXPECT_EQ(hostError, dicomNodeError_t::SUCCESS);

  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * DIM_SQ,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  EXPECT_FLOAT_EQ(matrix[0], 1.0f);
  EXPECT_FLOAT_EQ(matrix[1], 0.0f);
  EXPECT_FLOAT_EQ(matrix[2], 0.0f);
  EXPECT_FLOAT_EQ(matrix[3], 1.0f);

  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrixOutput->points,
    sizeof(float) * DIM_SQ,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  EXPECT_NEAR(matrix[0], 0.333333f, EPSILON);
  EXPECT_NEAR(matrix[1], -0.666667f, EPSILON);
  EXPECT_NEAR(matrix[2], 0.0f, EPSILON);
  EXPECT_NEAR(matrix[3], 0.5f, EPSILON);

  cudaFree(error);
  cudaFree(gpuMatrix);
  cudaFree(gpuMatrixOutput);
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
  EXPECT_EQ(status, cudaSuccess);
  status = cudaMalloc(&gpuMatrixOutput, sizeof(SquareMatrix<3>));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMalloc(&error, sizeof(dicomNodeError_t));
  EXPECT_EQ(status, cudaSuccess);

  status = cudaMemcpy(gpuMatrix->points,
                      matrix,
                      sizeof(float) * 9,
                      cudaMemcpyHostToDevice);
  EXPECT_EQ(status, cudaSuccess);

  matrix_inversion<3><<<1,1024>>>(gpuMatrix, gpuMatrixOutput, error);
  status = cudaGetLastError();
  EXPECT_EQ(status, cudaSuccess);

  dicomNodeError_t hostError;
  status = cudaMemcpy(
    &hostError,
    error,
    sizeof(dicomNodeError_t),
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);
  EXPECT_EQ(hostError, dicomNodeError_t::SUCCESS);

  status = cudaMemcpy(
    matrix,
    gpuMatrix->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  EXPECT_FLOAT_EQ(matrix[0], 1.0f);
  EXPECT_FLOAT_EQ(matrix[1], 0.0f);
  EXPECT_FLOAT_EQ(matrix[2], 0.0f);
  EXPECT_FLOAT_EQ(matrix[3], 0.0f);
  EXPECT_FLOAT_EQ(matrix[4], 1.0f);
  EXPECT_FLOAT_EQ(matrix[5], 0.0f);
  EXPECT_FLOAT_EQ(matrix[6], 0.0f);
  EXPECT_FLOAT_EQ(matrix[7], 0.0f);
  EXPECT_FLOAT_EQ(matrix[8], 1.0f);

  EXPECT_EQ(status, cudaSuccess);
  status = cudaMemcpy(
    matrix,
    gpuMatrixOutput->points,
    sizeof(float) * 9,
    cudaMemcpyDeviceToHost
  );
  EXPECT_EQ(status, cudaSuccess);

  EXPECT_NEAR(matrix[0],  0.666667f, EPSILON);
  EXPECT_NEAR(matrix[1],  3.0f, EPSILON);
  EXPECT_NEAR(matrix[2],  -1.66667f, EPSILON);
  EXPECT_NEAR(matrix[3],  -0.0f, EPSILON);
  EXPECT_NEAR(matrix[4],  -2.0f, EPSILON);
  EXPECT_NEAR(matrix[5],  1.0f, EPSILON);
  EXPECT_NEAR(matrix[6], -0.3333333f, EPSILON);
  EXPECT_NEAR(matrix[7],  0.0f, EPSILON);
  EXPECT_NEAR(matrix[8],  0.3333333f, EPSILON);

  cudaFree(error);
  cudaFree(gpuMatrix);
  cudaFree(gpuMatrixOutput);
}

TEST(LIN_ALG, MATRIX_MULTIPLICATION_ORDERING){
  SquareMatrix<2> matrix{1.0f,2.0f,3.0f,4.0f};
  Point<2> point{5.0f,6.0f};

  Point<2> one_way = point * matrix;
  Point<2> other_way = matrix * point;

  Point<2> answer_1{17.0f, 39.0f};
  Point<2> answer_2{23.0f, 34.0f};

  EXPECT_TRUE(one_way == answer_1);
  EXPECT_TRUE(other_way == answer_2);
}

TEST(LIN_ALG, POINTS_ARE_ZERO_INITIALIZED){
  Point<16> p;

  for(int i = 0; i < 16; i++){
    EXPECT_FLOAT_EQ(p[i], 0.0f);
  }
}