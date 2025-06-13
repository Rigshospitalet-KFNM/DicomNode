# Python Standard library

# Third party Library
import numpy
from pydicom import Dataset, DataElement
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode Modules
from dicomnode.lib.exceptions import MissingDatasets, InvalidDataset
from dicomnode import dicom
from dicomnode.math.space import Space
from dicomnode.math.image import Image

# Testing helpers
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class DicomTests(DicomnodeTestCase):
  def test_add_private_tag_errors_on_public_tags(self):
    test_dataset = Dataset()
    private_de = DataElement(0x0010_0010, 'PN', "HELLO WOLRD")
    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)

    self.assertEqual(len(test_dataset), 0)

  def test_add_private_tag_errors_on_private_group_allocation(self):
    test_dataset = Dataset()
    private_de = DataElement(0x1337_0001, 'LO', "DICOMNODE")
    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)
    self.assertEqual(len(test_dataset), 0)

  def test_add_private_tag_errors_on_reserved_tags(self):
    test_dataset = Dataset()
    private_de = DataElement(0x1337_01FE, 'LO', "DICOMNODE")
    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)
    self.assertEqual(len(test_dataset), 0)
    private_de = DataElement(0x1337_01FF, 'LO', "DICOMNODE")
    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)
    self.assertEqual(len(test_dataset), 0)

  def test_add_private_tags_to_empty_dataset(self):
    test_dataset = Dataset()

    private_de = DataElement(0x1337_0101, 'IS', 1512)

    dicom.add_private_tag(test_dataset, private_de)

    self.assertIn(private_de.tag, test_dataset)

    self.assertIn(0x1337_0001, test_dataset)
    self.assertIn(0x1337_01FE, test_dataset)
    self.assertIn(0x1337_01FF, test_dataset)

  def test_add_private_tags_adding_multiple_things(self):
    test_dataset = Dataset()

    private_de_1 = DataElement(0x1337_0101, 'IS', 1512)
    private_de_2 = DataElement(0x1337_0102, 'IS', 1512)
    private_de_3 = DataElement(0x1337_0103, 'IS', 1512)

    dicom.add_private_tag(test_dataset, private_de_1)
    dicom.add_private_tag(test_dataset, private_de_2)
    dicom.add_private_tag(test_dataset, private_de_3)

    self.assertIn(private_de_1.tag, test_dataset)
    self.assertIn(private_de_2.tag, test_dataset)
    self.assertIn(private_de_3.tag, test_dataset)

    self.assertIn(0x1337_0001, test_dataset)
    self.assertIn(0x1337_01FE, test_dataset)
    self.assertIn(0x1337_01FF, test_dataset)

    self.assertEqual(test_dataset[0x1337_01FE].VM, 3)

  def test_add_private_tags_failures(self):
    test_dataset = Dataset()
    private_de_1 = DataElement(0x1337_0101, 'IS', 1512)
    dicom.add_private_tag(test_dataset, private_de_1)

    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de_1)

    test_dataset[0x0009_0001] = DataElement(0x0009_0001, 'LO', 'SIEMENS')

    self.assertRaises(
      ValueError,
      dicom.add_private_tag,
      test_dataset,
      DataElement(0x0009_0101, 'LO', 'BLAH')
    )

    test_dataset[0x0009_0001] = DataElement(0x0009_0001, 'LO', 'DICOMNODE: Blah blah')

    self.assertRaises(
      ValueError,
      dicom.add_private_tag,
      test_dataset,
      DataElement(0x0009_0101, 'LO', 'BLAH')
    )


  def test_is_private_group_tag(self):
    self.assertTrue(dicom.is_private_group_tag(0x1337_0001))
    self.assertFalse(dicom.is_private_group_tag(0x1337_0100))
    self.assertFalse(dicom.is_private_group_tag(0x1336_0001))

  def test_creating_coordinate_system_with_space(self):
    testing_space = Space(
      [[6,0,0], [0,7,0], [0,0,8]], #type: ignore
      [0,0,0],
      [2,3,4]
    )

    points, (x_dim, y_dim, z_dim), orientation, starting_point = dicom.create_dicom_coordinate_system(testing_space)

    self.assertEqual(x_dim, 6)
    self.assertEqual(y_dim, 7)
    self.assertEqual(z_dim, 8)

    self.assertListEqual(orientation, [1,0,0,0,1,0])
    self.assertListEqual([p for p in starting_point], [0,0,0])

  def test_creating_coordinate_system_with_space_errors_on_wrong_args(self):
    testing_space = Space(
      [[6,0,0], [0,7,0], [0,0,8]], #type: ignore
      [0,0,0],
      [2,3,4]
    )

    with self.assertRaises(TypeError):
      dicom.create_dicom_coordinate_system(testing_space, rm_shape=(2,3,4))


  def test_creating_coordinate_system_with_image(self):
    testing_space = Space(
      [[6,0,0], [0,7,0], [0,0,8]], #type: ignore
      [0,0,0],
      [2,3,4]
    )

    testing_image = Image(numpy.arange(2*3*4).reshape((2,3,4)), testing_space)

    points, (x_dim, y_dim, z_dim), orientation, starting_point = dicom.create_dicom_coordinate_system(testing_image)

    self.assertEqual(x_dim, 6)
    self.assertEqual(y_dim, 7)
    self.assertEqual(z_dim, 8)

    self.assertListEqual(orientation, [1,0,0,0,1,0])
    self.assertListEqual([p for p in starting_point], [0,0,0])

  def test_creating_coordinate_system_with_image_errors_on_wrong_args(self):
    testing_space = Space(
      [[6,0,0], [0,7,0], [0,0,8]], #type: ignore
      [0,0,0],
      [2,3,4]
    )

    testing_image = Image(numpy.arange(2*3*4).reshape((2,3,4)), testing_space)

    with self.assertRaises(TypeError):
      dicom.create_dicom_coordinate_system(testing_image, rm_shape=(2,3,4))

  def test_creating_coordinate_system_with_ndarray_rm(self):
    coordinate_system = numpy.array([
      [6,0,0,3],
      [0,7,0,4],
      [0,0,8,5],
      [0,0,0,1],
    ])

    points, (x_dim, y_dim, z_dim), orientation, starting_point = dicom.create_dicom_coordinate_system(
      coordinate_system, rm_shape=(10,11,12)
    )

    self.assertEqual(x_dim, 6)
    self.assertEqual(y_dim, 7)
    self.assertEqual(z_dim, 8)

    self.assertListEqual([x for x in orientation],[1,0,0,0,1,0])
    self.assertListEqual([x for x in starting_point],[3,4,5])

  def test_creating_coordinate_system_with_ndarray_cm(self):
    coordinate_system = numpy.array([
      [6,0,0,3],
      [0,7,0,4],
      [0,0,8,5],
      [0,0,0,1],
    ])

    points, (x_dim, y_dim, z_dim), orientation, starting_point = dicom.create_dicom_coordinate_system(
      coordinate_system, cm_shape=(12,11,10)
    )

    self.assertEqual(x_dim, 6)
    self.assertEqual(y_dim, 7)
    self.assertEqual(z_dim, 8)

    self.assertListEqual([x for x in orientation],[1,0,0,0,1,0])
    self.assertListEqual([x for x in starting_point],[3,4,5])

  def test_creating_coordinate_system_with_ndarray_error_on_args(self):
    coordinate_system = numpy.array([
      [6,0,0,3],
      [0,7,0,4],
      [0,0,8,5],
      [0,0,0,1],
    ])

    with self.assertRaises(TypeError):
      dicom.create_dicom_coordinate_system(
        coordinate_system, cm_shape=(12,11,10), rm_shape=(10,11,12)
      )
    with self.assertRaises(TypeError):
      dicom.create_dicom_coordinate_system(
        numpy.arange(4).reshape((2,2)), cm_shape=(12,11,10)
      )

  def test_creating_coordinate_system_error_on_unknown_args(self):
    self.assertRaises(TypeError, dicom.create_dicom_coordinate_system, "Hello world")

  def test_comparing_datasets(self):
    dataset_1 = Dataset()
    dataset_2 = Dataset()

    # Both have the element
    dataset_1.SOPClassUID = SecondaryCaptureImageStorage
    dataset_2.SOPClassUID = SecondaryCaptureImageStorage

    # dataset 1 has element but dataset 2 doesn't
    dataset_1.PatientID = "2000331122"

    # dataset 2 has element but dataset 1 doesn't
    dataset_2.Rows = 100

    dataset_1.Columns = 100


    iteration = 0
    for tag_1, tag_2 in dicom.ComparingDatasets(dataset_1, dataset_2):
      iteration += 1
      if iteration == 1:
        if tag_1 is None or tag_2 is None:
          self.fail("tag 1 or tag 2 is None")

        self.assertEqual(tag_1.tag, 0x0008_0016)
        self.assertEqual(tag_2.tag, 0x0008_0016)
      elif iteration == 2:
        if tag_1 is None or tag_2 is not None:
          self.fail("Tag 1 is none or tag 2 is not None")

        self.assertEqual(tag_1.tag, 0x0010_0020)
      elif iteration == 3:
        if tag_1 is not None or tag_2 is None:
          self.fail("Tag 2 is none or tag 1 is not None")

        self.assertEqual(tag_2.tag, 0x0028_0010)

      elif iteration == 4:
        if tag_1 is None or tag_2 is not None:
          self.fail("Tag 2 is none or tag 1 is not None")

        self.assertEqual(tag_1.tag, 0x0028_0011)

    iteration = 0
    for tag_1, tag_2 in dicom.ComparingDatasets(dataset_2, dataset_1):
      iteration += 1
      if iteration == 1:
        if tag_1 is None or tag_2 is None:
          self.fail("tag 1 or tag 2 is None")

        self.assertEqual(tag_1.tag, 0x0008_0016)
        self.assertEqual(tag_2.tag, 0x0008_0016)
      elif iteration == 2:
        if tag_2 is None or tag_1 is not None:
          self.fail("Tag 1 is none or tag 2 is not None")

        self.assertEqual(tag_2.tag, 0x0010_0020)
      elif iteration == 3:
        if tag_2 is not None or tag_1 is None:
          self.fail("Tag 2 is none or tag 1 is not None")

        self.assertEqual(tag_1.tag, 0x0028_0010)

      elif iteration == 4:
        if tag_2 is None or tag_1 is not None:
          self.fail("Tag 2 is none or tag 1 is not None")

        self.assertEqual(tag_2.tag, 0x0028_0011)

  def test_assess_single_series_errors_on_empty_container(self):
    self.assertRaises(MissingDatasets,dicom.assess_single_series, [])

  def test_assess_single_series_errors_on_multiple_containers(self):
    dataset_1 = Dataset()
    dataset_1.SeriesInstanceUID = dicom.gen_uid()
    dataset_2 = Dataset()
    dataset_2.SeriesInstanceUID = dicom.gen_uid()

    self.assertRaises(InvalidDataset, dicom.assess_single_series, [dataset_1, dataset_2])

  def test_sanity_check_dataset(self):
    empty_dataset = Dataset()

    self.assertFalse(dicom.sanity_check_dataset(empty_dataset))

from . import tests_nifti
from . import tests_series
from . import tests_series
