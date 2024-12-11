"""This test is mostly for rt struct

The base example is a triangle:

 |
 |       (4,5,1:6)
 |           /\
 |          /  \
 |         /    \
 |        /      \
 |       /        \
 |      /          \
 |     /____________\
 | (1,1,1:6)     (7,1,1:6)
 |_________________________

"""

# Python Standard library

# Third party modules
from pydicom import Dataset, Sequence
from rt_utils import RTStruct

# Dicomnode modules
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.dicom import gen_uid
from dicomnode.dicom.series import DicomSeries
from dicomnode.dicom.rt_structs import get_mask, get_contour_sequence_by_name

# Test helper functions
from helpers import generate_numpy_datasets
from helpers.dicomnode_test_case import DicomnodeTestCase

class RTStructTestCases(DicomnodeTestCase):
  def setUp(self):
    self.number_of_slices = 9
    self.rows = 9
    self.cols = 9

    self.frame_of_reference = gen_uid()
    # Lingo helper roi = Region of interest
    self.ROI_number = 1337
    self.ROI_name = "testROI"

    self.series = DicomSeries(
      [ds for ds in generate_numpy_datasets(self.number_of_slices,
                                            Cols=self.rows,
                                            Rows=self.cols,
                                            starting_image_position=[0,0,0],
                                            image_orientation=[1,0,0,0,1,0],
                                            pixel_spacing=[1,1],
                                            slice_thickness=1)]
    )
    self.series["FrameOfReferenceUID"] = self.frame_of_reference

    self.rt_struct = Dataset()

    self.referenced_frame_of_reference_dataset = Dataset()
    self.structure_set_roi_sequence_dataset = Dataset()
    self.roi_contour_sequence_dataset = Dataset()
    self.contour_sequence_dataset = Dataset()
    self.contour_image_sequence_datasets = [Dataset() for _ in self.series]

    self.referenced_frame_of_reference_dataset.FrameOfReferenceUID = self.frame_of_reference

    for contour_image_dataset, source_dataset in zip(self.contour_image_sequence_datasets, self.series):
      contour_image_dataset.ReferencedSOPClassUID = source_dataset.SOPClassUID
      contour_image_dataset.ReferencedSOPInstanceUID = source_dataset.SOPInstanceUID

    self.contour_sequence_dataset.ContourImageSequence = Sequence(self.contour_image_sequence_datasets)
    self.contour_sequence_dataset.ContourGeometricType = "CLOSED_PLANAR"
    self.contour_sequence_dataset.NumberOfContourPoints = 3
    self.contour_sequence_dataset.ContourData = [
      1,1,1,
      4,1,1,
      4,7,1,
    ]

    self.roi_contour_sequence_dataset.ReferencedROINumber = self.ROI_number
    self.roi_contour_sequence_dataset.ContourSequence = Sequence([
      self.contour_sequence_dataset
    ])

    self.structure_set_roi_sequence_dataset.ROINumber = self.ROI_number
    self.structure_set_roi_sequence_dataset.ROIName = self.ROI_name
    self.structure_set_roi_sequence_dataset.ReferencedFrameOfReferenceUID = self.frame_of_reference

    self.rt_struct.FrameOfReferenceUID = self.frame_of_reference
    self.rt_struct.ReferencedFrameOfReferenceSequence = Sequence([
      self.referenced_frame_of_reference_dataset
    ])

    self.rt_struct.ROIContourSequence = Sequence([
      self.roi_contour_sequence_dataset
    ])

    self.rt_struct.StructureSetROISequence = Sequence([
      self.structure_set_roi_sequence_dataset
    ])

  def test_simple_rt_struct_construction(self):
    rt_struct = RTStruct(self.series.datasets, self.rt_struct) # type: ignore

    # note that there might be a problem with (x,y) = (y,x)
    mask = rt_struct.get_roi_mask_by_name(self.ROI_name)
    my_mask = get_mask(rt_struct, self.ROI_name)

    self.assertEqual(mask.shape, my_mask.raw.shape)

    # So I still think there might be an error here some where
    for i in range(self.number_of_slices):
      self.assertTrue((mask[:,:,i] == my_mask[i] ).all())

  def test_extracting_data_invalid_rt_struct_raises(self):
    self.assertRaises(InvalidDataset, get_contour_sequence_by_name, Dataset(), "Bah")
    self.assertRaises(InvalidDataset, get_contour_sequence_by_name, self.rt_struct, "Bah")
