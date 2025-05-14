#pragma once

#include<stdint.h>

#include <gtest/gtest.h>
#include"../gpu_code/dicom_node_gpu.cuh"

TEST(BOUNDING_BOX, Box_25x25x25_with_10_11_12_20_19_18){
  cudaError_t status;
  const size_t elements = 25 * 25 * 25;

  auto idx = [](int i, int j, int k) constexpr {
    return k * 25 * 25 + j * 25 + i;
  };

  const size_t data_size = elements * sizeof(uint8_t);
  uint8_t* host_data = (uint8_t*)malloc(data_size);
  EXPECT_NE(host_data, (void*)NULL);
  memset(host_data, (uint8_t)0, data_size);
  host_data[idx(10,11,12)] = 1;
  host_data[idx(20,19,18)] = 1;
  BoundingBox_3D out;

  status = reduce<1, BoundingBoxOP_3D<uint8_t>, uint8_t, BoundingBox_3D, Extent<3>>(
    host_data, data_size, &out, {25,25,25}
  );

  EXPECT_EQ(status, cudaSuccess);
  EXPECT_EQ(out.x_min, 10);
  EXPECT_EQ(out.x_max, 20);
  EXPECT_EQ(out.y_min, 11);
  EXPECT_EQ(out.y_max, 19);
  EXPECT_EQ(out.z_min, 12);
  EXPECT_EQ(out.z_max, 18);

  free(host_data);
}
