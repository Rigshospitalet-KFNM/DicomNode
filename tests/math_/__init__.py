# Python standard library
from unittest import skipIf

# Third party modules
import numpy
import nibabel

# Dicomnode modules
from dicomnode import library_paths
from dicomnode.math import switch_ordering, CUDA
from dicomnode.math.image import Image
from dicomnode.math.space import Space

# Test modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

ct_image_path = library_paths.report_data_directory / "CT_nifti" / "CT.nii"
ct_brain_path = library_paths.report_data_directory / "CT_nifti" / "segmentation" / "brain.nii.gz"



class MathTestCases(DicomnodeTestCase):
  def test_row_to_column(self):
    shape = (4,3,2)

    input_array = numpy.arange(numpy.prod(shape)).reshape(shape)
    test_array = switch_ordering(input_array)

    self.assertEqual(test_array.shape, tuple(reversed(shape)))
    to_list = [[[ elem for elem in subsublist] # Convert to build-in lists
                for subsublist in sublist]
                for sublist in test_array ]
    self.assertListEqual(to_list, [
      [[ 0.0,  6.0,  12.0, 18.0],
       [ 2.0,  8.0, 14.0, 20.0],
       [ 4.0, 10.0, 16.0, 22.0],
      ],
      [[ 1.0,  7.0, 13.0, 19.0],
       [ 3.0,  9.0, 15.0, 21.0],
       [ 5.0, 11.0, 17.0, 23.0],
      ]
    ])



  @skipIf(not CUDA, "You need GPU for this test")
  def test_center_of_gravity(self):
    image_data =  numpy.array([
      [[1,0,0,1],
       [0,0,0,0],
       [0,0,0,0],
       [1,0,0,1]],

      [[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0]],

      [[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0]],

      [[1,0,0,1],
       [0,0,0,0],
       [0,0,0,0],
       [1,0,0,1]],
    ], dtype=numpy.float32)

    image = Image(
      image_data, Space(
        numpy.eye(3), [0,0,0], image_data.shape
      )
    )

    from dicomnode.math import _cuda
    success, cog = _cuda.center_of_gravity(image.raw)

    self.assertEqual(cog[0], 1.5)
    self.assertEqual(cog[1], 1.5)
    self.assertEqual(cog[2], 1.5)

  @skipIf(True, "This is just here to performance test - You need GPU")
  def test_center_of_gravity_performance_test(self):
    import time
    from dicomnode.dicom.series import extract_image
    from dicomnode.math.image import mask_image
    from dicomnode.math import center_of_gravity, cpu_center_of_gravity
    nifti: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(ct_image_path) # type: ignore
    image = extract_image(nifti)

    seg_nifti: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(ct_brain_path) # type: ignore

    seg_image = extract_image(seg_nifti)
    masked_image = mask_image(image, seg_image)

    runtimes_gpu = []
    runtimes_cpu = []

    for i in range(10):
      start = time.perf_counter()
      gpu_cog = center_of_gravity(masked_image)
      end = time.perf_counter()

      runtimes_gpu.append(end - start)

    for i in range(10):
      start = time.perf_counter()
      cpu_cog =  cpu_center_of_gravity(masked_image)
      end = time.perf_counter()

      runtimes_cpu.append(end - start)

    print(numpy.mean(runtimes_cpu))
    print(numpy.mean(runtimes_gpu))
    self.assertTrue((numpy.array(gpu_cog) - numpy.array(cpu_cog) < 0.0001).all()) # type: ignore





from . import tests_affine
from . import tests_bounding_box
from . import tests_mirror
from . import tests_image
from . import tests_types
from . import tests_space
