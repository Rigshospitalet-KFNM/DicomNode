#include<array>

#include<gtest/gtest.h>

#include<vector>

#include<cuda/std/array>

#include"../gpu_code/dicom_node_gpu.cuh"

namespace TEST_EXTENT {
  TEST(EXTENT, FLAT_INDEX){
    Extent<3> extent {4,4,4};

    Index<3> i000{0,0,0};
    Index<3> i001{1,0,0};
    Index<3> i002{2,0,0};
    Index<3> i003{3,0,0};
    Index<3> i010{0,1,0};
    Index<3> i011{1,1,0};
    Index<3> i012{2,1,0};
    Index<3> i013{3,1,0};

    cuda::std::optional<u64> v000 = extent.flat_index(i000);
    cuda::std::optional<u64> v001 = extent.flat_index(i001);
    cuda::std::optional<u64> v002 = extent.flat_index(i002);
    cuda::std::optional<u64> v003 = extent.flat_index(i003);
    cuda::std::optional<u64> v010 = extent.flat_index(i010);
    cuda::std::optional<u64> v011 = extent.flat_index(i011);
    cuda::std::optional<u64> v012 = extent.flat_index(i012);
    cuda::std::optional<u64> v013 = extent.flat_index(i013);

    EXPECT_TRUE(v000.has_value());
    EXPECT_EQ(v000, 0);
    EXPECT_TRUE(v001.has_value());
    EXPECT_EQ(v001, 1);
    EXPECT_TRUE(v002.has_value());
    EXPECT_EQ(v002, 2);
    EXPECT_TRUE(v003.has_value());
    EXPECT_EQ(v003, 3);

    EXPECT_TRUE(v010.has_value());
    EXPECT_EQ(  v010, 4);
    EXPECT_TRUE(v011.has_value());
    EXPECT_EQ(  v011, 5);
    EXPECT_TRUE(v012.has_value());
    EXPECT_EQ(  v012, 6);
    EXPECT_TRUE(v013.has_value());
    EXPECT_EQ(  v013, 7);
  }
}


namespace TEST_IMAGE {

template<typename T>
__global__ void dummy_indexing(
    Image<3,T>* image,
    Index<3>* indicies,
    T* out,
    const size_t n
){
  u64 gid = get_gid();

  if(gid < n){
    out[gid] = image->volume.at(indicies[gid]);
  }
}

template<typename T, size_t N>
__global__ void dummy_indexing(
    Image<3,T>* image,
    cuda::std::array<Point<3>, N>* indicies,
    cuda::std::array<T, N>* out
){
  u64 gid = get_gid();

  cuda::std::array<Point<3>, N>& indicies_ = *indicies;
  cuda::std::array<T, N>& out_ = *out;

  if(gid < N){
    out_[gid] = image->volume.interpolate_at_index_point(indicies_[gid]);
  }
}

void check(cudaError_t error){
  if(error != cudaSuccess){
    std::cout << "THERE AN ERROR!\n";
  }
}

TEST(IMAGE, CUDA_INDEXING){
  constexpr size_t z = 3;
  constexpr size_t y = 4;
  constexpr size_t x = 3;

  constexpr Extent<3> extent(z,y,x);

  f32 data[ z * y * x ] = {
    10.0f, 30.0f, 40.0f,
    20.0f, 50.0f, 70.0f,
    310.0f, 130.0f, 240.0f,
    320.0f, 150.0f, 270.0f,

    110.0f, 130.0f, 140.0f,
    120.0f, 150.0f, 170.0f,
    10.0f, 30.0f, 40.0f,
    20.0f, 50.0f, 70.0f,

    -10.0f, 160.0f, -40.0f,
    -20.0f, 150.0f, -720.0f,
    -5.0f, 350.0f, 40.0f,
    -50.0f, -150.0f, -720.0f,
  };

  constexpr size_t data_size = extent.elements() * sizeof(f32);

  const Space<3> image_space {
    .starting_point = Point<3>{
      0.0f, 0.0f, 0.0f
    },

    .basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
    .inverted_basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      },
    },
    .extent = Extent<3>{z,y,x}
  };

  Image<3, f32> host_image{
    image_space, Volume<3, f32>{
      .data=nullptr,
      .m_extent=extent,
      .default_value=1.0f
    }
  };

  check(cudaMalloc(&(host_image.volume.data), data_size));
  check(cudaMemcpy(host_image.volume.data, data, data_size, cudaMemcpyDefault));

  constexpr u64 num_tests = 3;
  constexpr u64 index_size = num_tests * sizeof(Index<3>);
  constexpr u64 out_size = num_tests * sizeof(f32);

  Image<3,f32>* device_image = nullptr;
  check(cudaMalloc(&device_image, sizeof(Image<3,f32>)));
  check(cudaMemcpy(device_image, &host_image, sizeof(Image<3,f32>), cudaMemcpyDefault));

  std::array<Index<3>, num_tests> host_indicies{{
    Index<3>(0,0,0),
    Index<3>(2,2,1),
    Index<3>(2,3,1)
  }};

  Index<3>* dev_indicies = nullptr;
  f32* dev_outs = nullptr;

  check(cudaMalloc(&dev_indicies, index_size));
  check(cudaMemcpy(dev_indicies, host_indicies.data(), index_size, cudaMemcpyDefault));
  check(cudaMalloc(&dev_outs, out_size));

  dummy_indexing<f32><<<1, num_tests>>>(device_image, dev_indicies, dev_outs, num_tests);

  std::array<f32, num_tests> host_outs;

  check(cudaMemcpy(host_outs.data(), dev_outs, out_size, cudaMemcpyDefault));

  Volume<3, f32> host_volume{
    .data=data,
    .m_extent=extent,
    .default_value = 15.0
  };

  for(u8 i = 0; i < num_tests; i++){
    EXPECT_FLOAT_EQ(host_outs[i], host_volume.at(host_indicies[i]));
  }

  cudaFree(host_image.volume.data);
  cudaFree(device_image);
  cudaFree(dev_indicies);
  cudaFree(dev_outs);
}

TEST(IMAGE, CUDA_POINT_INDEXING){
  constexpr size_t z = 3;
  constexpr size_t y = 4;
  constexpr size_t x = 3;

  constexpr Extent<3> extent(z,y,x);

  f32 data[ z * y * x ] = {
    10.0f, 30.0f, 40.0f,
    20.0f, 50.0f, 70.0f,
    310.0f, 130.0f, 240.0f,
    320.0f, 150.0f, 270.0f,

    110.0f, 130.0f, 140.0f,
    120.0f, 150.0f, 170.0f,
    10.0f, 30.0f, 40.0f,
    20.0f, 50.0f, 70.0f,

    -10.0f, 160.0f, -40.0f,
    -20.0f, 150.0f, -720.0f,
    -5.0f, 350.0f, 40.0f,
    -50.0f, -150.0f, -720.0f,
  };

  constexpr u64 data_size = z * y * x * sizeof(f32);

  const Space<3> image_space {
    .starting_point = Point<3>{
      0.0f, 0.0f, 0.0f
    },

    .basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
    .inverted_basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      },
    },
    .extent = Extent<3>{z,y,x}
  };

  Image<3, f32> ghost_image{
    image_space, Volume<3, f32>{
      .data=nullptr,
      .m_extent=extent,
      .default_value=1.0f
    }
  };

  Image<3, f32> host_image{
    image_space, Volume<3, f32>{
      .data=data,
      .m_extent=extent,
      .default_value=1.0f
    }
  };

  check(cudaMalloc(&(ghost_image.volume.data), data_size));
  check(cudaMemcpy(ghost_image.volume.data, data, data_size, cudaMemcpyDefault));
  Image<3,f32>* device_image = nullptr;
  check(cudaMalloc(&device_image, sizeof(Image<3,f32>)));
  check(cudaMemcpy(device_image, &ghost_image, sizeof(Image<3,f32>), cudaMemcpyDefault));

  constexpr u64 num_tests = 4;

  cuda::std::array<Point<3>, num_tests> host_points{{
    {0.0f,0.0f,0.0f},
    {0.5f,0.0f,0.0f},
    {0.5f,0.5f,0.0f},
    {0.5f,0.5f,0.5f},
  }};

  EXPECT_FLOAT_EQ(host_image.volume.interpolate_at_index_point(host_points[0]), 10.0f);
  EXPECT_FLOAT_EQ(host_image.volume.interpolate_at_index_point(host_points[1]), 20.0f);
  EXPECT_FLOAT_EQ(host_image.volume.interpolate_at_index_point(host_points[2]), 27.5f);
  EXPECT_FLOAT_EQ(host_image.volume.interpolate_at_index_point(host_points[3]), 77.5f);


  constexpr u64 point_size = sizeof(cuda::std::array<Point<3>, num_tests>);
  constexpr u64 out_size = sizeof(cuda::std::array<f32, num_tests>);
  cuda::std::array<f32, num_tests>* dev_outs = nullptr;
  cuda::std::array<Point<3>, num_tests>* dev_points = nullptr;

  cudaMalloc(&dev_outs, out_size);
  cudaMalloc(&dev_points, point_size);
  cudaMemcpy(dev_points, host_points.data(), point_size, cudaMemcpyDefault);

  dummy_indexing<f32, num_tests><<<1, num_tests>>>(device_image, dev_points, dev_outs);

  cuda::std::array<f32, num_tests> host_outs;
  cudaMemcpy(host_outs.data(), dev_outs, out_size, cudaMemcpyDefault);

  EXPECT_FLOAT_EQ(host_outs[0], 10.0f);
  EXPECT_FLOAT_EQ(host_outs[1], 20.0f);
  EXPECT_FLOAT_EQ(host_outs[2], 27.5f);
  EXPECT_FLOAT_EQ(host_outs[3], 77.5f);

  cudaFree(ghost_image.volume.data);
  cudaFree(device_image);
  cudaFree(dev_outs);
  cudaFree(dev_points);
}

} // end of namespace

namespace TEST_VOLUME {


TEST(VOLUME, SUB_VOLUME_HOST){
  constexpr static u32 z = 4;
  constexpr static u32 y = 4;
  constexpr static u32 x = 4;
  constexpr static Extent<3> extent{z,y,x};


  std::array<f32, extent.elements()> data = {{
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,

    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,

    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,

    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0
  }};

  Volume<3, f32> og_volume{
    .data=data.data(),
    .m_extent=extent,
    .default_value=-1.0f
  };

  Extent<3> new_extent{2,2,2};
  Index<3> offset_index{1,1,1};
  std::array<f32, 8> result_data;

  auto the_sub_volume = sub_volume(
    og_volume,
    result_data.data(),
    new_extent,
    offset_index
  );

  EXPECT_FLOAT_EQ(result_data[0],1.0f);
  EXPECT_FLOAT_EQ(result_data[1],1.0f);
  EXPECT_FLOAT_EQ(result_data[2],1.0f);
  EXPECT_FLOAT_EQ(result_data[3],1.0f);

  EXPECT_FLOAT_EQ(result_data[4],2.0f);
  EXPECT_FLOAT_EQ(result_data[5],2.0f);
  EXPECT_FLOAT_EQ(result_data[6],2.0f);
  EXPECT_FLOAT_EQ(result_data[7],2.0f);
}

constexpr u64 ImageSize = sizeof(Image<3, f32>);

__global__ void sub_volume_gpu_kernel(Volume<3, f32> volume, f32* out, Extent<3> new_extent, Index<3> offset){
  sub_volume(
    volume, out, new_extent, offset
  );
}

TEST(VOLUME, SUB_VOLUME_GPU_SMALL){
  constexpr static u32 z = 4;
  constexpr static u32 y = 4;
  constexpr static u32 x = 4;
  constexpr static Extent<3> extent{z,y,x};


  std::array<f32, extent.elements()> data = {{
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,

    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,

    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,

    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0
  }};

  Volume<3, f32> og_volume{
    .data=data.data(),
    .m_extent=extent,
    .default_value=-1.0f
  };


  Volume<3, f32> ghost_volume{
    .data=nullptr,
    .m_extent=extent,
    .default_value=-1.0f
  };

  CUDA_CHECK(cudaMalloc(&(ghost_volume.data), sizeof(f32) * extent.elements()));
  CUDA_CHECK(cudaMemcpy(ghost_volume.data, data.data(), sizeof(f32) * extent.elements(), cudaMemcpyDefault));

  f32* dev_out_pointer = nullptr;
  Extent<3> new_extent{2,2,2};
  Index<3> offset{1,1,1};

  const u64 out_size = sizeof(f32) * new_extent.elements();
  CUDA_CHECK(cudaMalloc(&dev_out_pointer, out_size));

  sub_volume_gpu_kernel<<<1, 1024>>>(ghost_volume, dev_out_pointer, new_extent, offset);
  EXPECT_EQ(cudaSuccess, cudaGetLastError());

  std::vector<f32> host_out(new_extent.elements());

  CUDA_CHECK(cudaMemcpy(
    host_out.data(),
    dev_out_pointer,
    out_size,
    cudaMemcpyDefault
  ));

  EXPECT_FLOAT_EQ(host_out[0],1.0f);
  EXPECT_FLOAT_EQ(host_out[1],1.0f);
  EXPECT_FLOAT_EQ(host_out[2],1.0f);
  EXPECT_FLOAT_EQ(host_out[3],1.0f);

  EXPECT_FLOAT_EQ(host_out[4],2.0f);
  EXPECT_FLOAT_EQ(host_out[5],2.0f);
  EXPECT_FLOAT_EQ(host_out[6],2.0f);
  EXPECT_FLOAT_EQ(host_out[7],2.0f);

  cudaFree(ghost_volume.data);
  cudaFree(dev_out_pointer);
}

TEST(VOLUME, SUB_VOLUME_GPU_SMALL_OUTSIDE){
  constexpr static u32 z = 4;
  constexpr static u32 y = 4;
  constexpr static u32 x = 4;
  constexpr static Extent<3> extent{z,y,x};


  std::array<f32, extent.elements()> data = {{
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,

    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,

    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0,

    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0,
    3.0, 3.0, 3.0, 3.0
  }};

  Volume<3, f32> og_volume{
    .data=data.data(),
    .m_extent=extent,
    .default_value=-1.0f
  };


  Volume<3, f32> ghost_volume{
    .data=nullptr,
    .m_extent=extent,
    .default_value=-1.0f
  };

  CUDA_CHECK(cudaMalloc(&(ghost_volume.data), sizeof(f32) * extent.elements()));
  CUDA_CHECK(cudaMemcpy(ghost_volume.data, data.data(), sizeof(f32) * extent.elements(), cudaMemcpyDefault));

  f32* dev_out_pointer = nullptr;
  Extent<3> new_extent{2,2,2};
  Index<3> offset{3,3,3};

  const u64 out_size = sizeof(f32) * new_extent.elements();
  CUDA_CHECK(cudaMalloc(&dev_out_pointer, out_size));

  sub_volume_gpu_kernel<<<1, 1024>>>(ghost_volume, dev_out_pointer, new_extent, offset);
  EXPECT_EQ(cudaSuccess, cudaGetLastError());

  std::vector<f32> host_out(new_extent.elements());

  CUDA_CHECK(cudaMemcpy(
    host_out.data(),
    dev_out_pointer,
    out_size,
    cudaMemcpyDefault
  ));

  EXPECT_FLOAT_EQ(host_out[0],3.0f);
  EXPECT_FLOAT_EQ(host_out[1],-1.0f);
  EXPECT_FLOAT_EQ(host_out[2],-1.0f);
  EXPECT_FLOAT_EQ(host_out[3],-1.0f);

  EXPECT_FLOAT_EQ(host_out[4],-1.0f);
  EXPECT_FLOAT_EQ(host_out[5],-1.0f);
  EXPECT_FLOAT_EQ(host_out[6],-1.0f);
  EXPECT_FLOAT_EQ(host_out[7],-1.0f);

  cudaFree(ghost_volume.data);
  cudaFree(dev_out_pointer);
}


} // end of namespace
