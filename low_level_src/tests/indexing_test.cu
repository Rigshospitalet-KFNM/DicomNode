/**
 * @file indexing_test.cu
 * @author Demiguard (cjen0668@regionh.dk)
 * @brief Testing for gpu_code/core/indexing
 * @version 0.1
 * @date 2024-12-18
 *
 * @copyright Copyright (c) 2024
 *
 */

#include"../gpu_code/dicom_node_gpu.cuh"

#include<gtest/gtest.h>

#include<cuda/std/optional>

TEST(INDEXING, CREATION_TEST_MULTIPLE_ARGS){
  const Domain<3> domain(3,4,5);

  EXPECT_EQ(domain.x(), 5);
  EXPECT_EQ(domain.y(), 4);
  EXPECT_EQ(domain.z(), 3);

  const Index<3> index(1,2,3);

  EXPECT_EQ(index.x(), 1);
  EXPECT_EQ(index.y(), 2);
  EXPECT_EQ(index.z(), 3);
}

TEST(INDEXING, CREATION_TEST_LIST){
  const Domain<3> domain({3,4,5});

  EXPECT_EQ(domain.x(), 5);
  EXPECT_EQ(domain.y(), 4);
  EXPECT_EQ(domain.z(), 3);

  const Index<3> index({1,2,3});

  EXPECT_EQ(index.x(), 1);
  EXPECT_EQ(index.y(), 2);
  EXPECT_EQ(index.z(), 3);
}

TEST(INDEXING, INDEXING_INTO_DATA_STRUCTURES){
  const Domain<3> domain({3,4,5});

  EXPECT_EQ(domain.x(), domain[2]);
  EXPECT_EQ(domain.y(), domain[1]);
  EXPECT_EQ(domain.z(), domain[0]);

  const Index<3> index({1,2,3});

  EXPECT_EQ(index.x(), index[0]);
  EXPECT_EQ(index.y(), index[1]);
  EXPECT_EQ(index.z(), index[2]);
}

TEST(INDEXING, CONTAINS_TAKES_REVERSED_INDEXES_INTO_ACCOUNT){
  const Domain<3> domain(3,4,5);

  const Index<3> index_in(4,3,2);
  const Index<3> index_out(2,3,4);

  EXPECT_TRUE(domain.contains(index_in));
  EXPECT_FALSE(domain.contains(index_out));
}

TEST(INDEXING, TO_FLAT_INDEX){
  const Domain<3> domain(3,4,5);

  cuda::std::optional<uint64_t> flat_index = domain.flat_index(1,1,2);

  EXPECT_TRUE(flat_index.has_value());
  if(flat_index.has_value()){
    const uint64_t& flat_value = flat_index.value();
    EXPECT_EQ(flat_value,(uint64_t)(2*4*5 + 1 * 5 + 1));
  }
}

TEST(INDEXING, FROM_FLAT_INDEX){
  const Domain<3> domain(3,4,5);

  Index<3> index = domain.from_flat_index(1 + 2 * 5 + 2 * 4 * 5);

  EXPECT_EQ(index.x(), 1);
  EXPECT_EQ(index.y(), 2);
  EXPECT_EQ(index.z(), 2);
}