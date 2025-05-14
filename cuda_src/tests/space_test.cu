#include<gtest/gtest.h>
#include"../gpu_code/dicom_node_gpu.cuh"

using Space3 = Space<3>;
using Index3 = Index<3>;
using Point3 = Point<3>;


TEST(SPACE_INDEXING, REAL_LIFE_1){
  const Space3 dest_space{
    .starting_point = {-362.4010009765625f,-546.052978515625f, -805.7930297851562f},
    .basis = {1.649999976158142f, 0.0f, 0.0f,
              0.0f, 1.649999976158142f, 0.0f,
              0.0f, 0.0f, 1.6455700397491455f},
    .inverted_basis = {
      0.60606061f, 0.0f, 0.0f,
      0.0f , 0.60606061f, 0.0f,
      0.0f , 0.0f, 0.60769215f
    },
    .extent = {645,440,440}
  };

  const float host_basis_xy_length = 1.5234375f;

  const Space3 host_space{
    .starting_point = {-389.23828125,-573.23828125, -1133.0},
    .basis = {host_basis_xy_length, 0.0f, 0.0f,
              0.0f, host_basis_xy_length, 0.0f,
              0.0f, 0.0f, 3.0f},
    .inverted_basis = {
      0.65641026f, 0.0f, 0.0f,
      0.0f , 0.65641026f, 0.0f,
      0.0f , 0.0f, 0.33333333f
    },
    .extent = {354,512,512}
  };

  const Index3 host_central_idx{256, 256, 177};

  const Point3 host_central_point = host_space.at_index(host_central_idx);

  const float host_x = 256.0f * host_basis_xy_length + -389.23828125f;
  const float host_y = 256.0f * host_basis_xy_length + -573.23828125f;
  const float host_z = 177.0f * 3.0f + -1133.0f;

  EXPECT_FLOAT_EQ(host_x, host_central_point[0]);
  EXPECT_FLOAT_EQ(host_y, host_central_point[1]);
  EXPECT_FLOAT_EQ(host_z, host_central_point[2]);

  const Point3 inverse_index = host_space.interpolate_point(host_central_point);

  EXPECT_FLOAT_EQ(256.0f, inverse_index[0]);
  EXPECT_FLOAT_EQ(256.0f, inverse_index[1]);
  EXPECT_FLOAT_EQ(177.0f, inverse_index[2]);
}