__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from datetime import datetime, date, time
from logging import ERROR
from typing import Any, List, Tuple
from random import randint
from unittest import TestCase

# Third Party Modules
import numpy
from pydicom import DataElement, Dataset
from pydicom.tag import Tag
from pydicom.uid import SecondaryCaptureImageStorage, CTImageStorage

# Dicomnode modules
from dicomnode.constants import DICOMNODE_IMPLEMENTATION_UID
from dicomnode.lib.logging import get_logger
from dicomnode.dicom import gen_uid
from dicomnode.dicom.dicom_factory import CopyElement, DicomFactory, DiscardElement, FunctionalElement,\
  Blueprint, SeriesElement, StaticElement, InstanceCopyElement, \
  InstanceEnvironment, SequenceElement, get_pivot
from dicomnode.dicom.blueprints import add_UID_tag
from dicomnode.dicom.series import DicomSeries

from dicomnode.lib.exceptions import IncorrectlyConfigured

class BlueprintTestCase(TestCase):
  def setUp(self) -> None:
    self.virtual_study_date = StaticElement(0x00080020, 'DA', date.today())
    self.virtual_patient_name = StaticElement(0x00100010, 'PN', 'Face^Testy^Mac')
    self.virtual_patient_id = StaticElement(0x00100020, 'LO', '1502799995')

    self.virtual_patient_id_conflict = StaticElement(0x00100020, 'LO', '0404942445')
    self.virtual_patient_sex = StaticElement(0x00100040, 'CS', 'M')

    self.blueprint_1 = Blueprint([
      self.virtual_patient_name,
      self.virtual_patient_id,
    ])

    self.blueprint_2 = Blueprint([
      self.virtual_patient_id_conflict,
      self.virtual_patient_sex,
    ])

  def tearDown(self) -> None:
    pass

  def test_iter(self):
    for i,virtual_element in enumerate(self.blueprint_1):
      if i == 0:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_name))
      if i == 1:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_id))
      if i == 2:
        self.assertTrue(False) # pragma: no cover

  def test_add(self):
    bluer_print = self.blueprint_1 + self.blueprint_2
    self.assertFalse(id(bluer_print) == id(self.blueprint_1))
    self.assertFalse(id(bluer_print) == id(self.blueprint_2))

    for i,virtual_element in enumerate(self.blueprint_1):
      if i == 0:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_name))
      if i == 1:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_id))
      if i == 2:
        self.assertTrue(False) # pragma: no cover

    for i,virtual_element in enumerate(self.blueprint_2):
      if i == 0:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_id_conflict))
      if i == 1:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_sex))
      if i == 2:
        self.assertTrue(False) # pragma: no cover

    for i,virtual_element in enumerate(bluer_print):
      if i == 0:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_name))
      if i == 1:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_id_conflict))
      if i == 2:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_sex))
      if i == 3:
        self.assertTrue(False) # pragma: no cover

  def test_iadd(self):
    blueprint = Blueprint([
      self.virtual_patient_name,
      self.virtual_patient_id,
    ])

    blueprint += Blueprint([
      self.virtual_patient_id_conflict,
      self.virtual_patient_sex,
    ])

    for i,virtual_element in enumerate(blueprint):
      if i == 0:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_name))
      if i == 1:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_id_conflict))
      if i == 2:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_sex))
      if i == 3:
        self.assertTrue(False) # pragma: no cover

  def test_contains(self):
    self.assertTrue(0x00100010 in self.blueprint_1)
    self.assertTrue(0x00100020 in self.blueprint_1)

  def test_getitem(self):
    self.assertEqual(id(self.virtual_patient_name), id(self.blueprint_1[0x00100010]))
    self.assertEqual(id(self.virtual_patient_id), id(self.blueprint_1[0x00100020]))

  def test_delitem(self):
    blueprint = Blueprint([
      self.virtual_patient_name,
      self.virtual_patient_id,
    ])
    del blueprint[0x00100010]
    self.assertRaises(KeyError, blueprint.__getitem__, 0x00100010)


  def test_setting_tags(self):
    blueprint = Blueprint([
      self.virtual_patient_name,
      self.virtual_patient_id,
    ])

    blueprint[0x00100020] = self.virtual_patient_id_conflict
    blueprint[Tag(0x00100020)] = self.virtual_patient_id_conflict

  def test_set_item_mismatch(self):
    self.assertRaises(ValueError, self.blueprint_1.__setitem__, 0x00100020, self.virtual_patient_name)
    self.assertRaises(TypeError, self.blueprint_1.__setitem__, 0x00100010, DataElement(0x00100010, 'PN', 'Face^Mace^to'))

  def test_blueprint_length(self):
    blueprint = Blueprint([
      self.virtual_patient_name,
      self.virtual_patient_id,
      self.virtual_patient_sex,
    ])

    self.assertEqual(len(blueprint), 3)

  def test_blueprint_get_required_tags_empty(self):
    self.assertListEqual(self.blueprint_1.get_required_tags(), [])

  def test_blueprint_get_required_tags_many(self):
    blueprint = Blueprint([
      CopyElement(0x00100030),
      CopyElement(0x00100040, Optional=True),
      InstanceCopyElement(0x00100050, 'IS')
    ])

    self.assertListEqual(blueprint.get_required_tags(), [0x00100030, 0x00100050])

  def test_get_pivot_of_dataset(self):
    test_dataset = Dataset()
    self.assertIs(get_pivot(test_dataset),test_dataset)

  def test_blueprint_with_long_customer_name(self):
    with self.assertLogs(get_logger()) as cm:
      StaticElement(0x0011_1099, 'IS',1, name="a" * 1000)

    self.assertEqual(len(cm.output), 1)


class DicomFactoryTestCase(TestCase):
  def setUp(self) -> None:
    self.factory = DicomFactory()

    self.patient_name = "test^patient"
    info_uint16 = numpy.iinfo(numpy.uint16)
    def gen_dataset(i: int) -> Dataset:
      ds = Dataset()
      self.factory.store_image_in_dataset(ds, numpy.random.randint(0,info_uint16.max, (11,12), numpy.uint16))
      ds.InstanceNumber = i + 1
      ds.PatientName = self.patient_name
      return ds

    self.parent_series = DicomSeries([
      gen_dataset(i) for i in range(13)
    ])



  def tearDown(self) -> None:
    pass

  def test_CopyElementCorporealialize(self):
    copy_element = CopyElement(0x00100020)
    optional_copy_element = CopyElement(0x00100020, Optional=True)

    dataset = Dataset()
    self.assertRaises(KeyError, copy_element.corporealialize, [dataset])
    self.assertIsNone(optional_copy_element.corporealialize([dataset]))
    cpr = '1502799995'
    dataset.PatientID = cpr

    data_element = copy_element.corporealialize([dataset])
    data_element_optional = optional_copy_element.corporealialize([dataset])

    if data_element is not None and data_element_optional is not None:
      self.assertIsInstance(data_element, DataElement)
      self.assertIsInstance(data_element_optional, DataElement)

      self.assertEqual(data_element.tag, 0x00100020)
      self.assertEqual(data_element_optional.tag, 0x00100020)

      self.assertEqual(data_element.value, cpr)
      self.assertEqual(data_element_optional.value, cpr)
    else:
      self.assertEqual(1,2) # pragma: ignore

  def test_discard_element(self):
    discard_element = DiscardElement(0x00100020)
    dataset = Dataset()
    cpr = '1502799995'
    dataset.PatientID = cpr
    self.assertIsNone(discard_element.corporealialize([dataset]))

  def test_series_element(self):
    series_element_no_arg = SeriesElement(0x0020000E, 'UI', gen_uid)

    dataset = Dataset()
    de = series_element_no_arg.corporealialize([dataset])

    self.assertEqual(de.tag, 0x0020000E)
    self.assertEqual('UI', de.VR)
    self.assertTrue(str(de.value.name).startswith(DICOMNODE_IMPLEMENTATION_UID + '.'))

    def bmiCalc(ds : Dataset):
      height = ds.PatientSize
      weight = ds.PatientWeight

      return weight / height ** 2

    series_element = SeriesElement(0x00101022, 'DS', bmiCalc)

    dataset.PatientWeight = 90
    dataset.PatientSize = 2

    de = series_element.corporealialize([dataset])

    self.assertEqual(de.tag, 0x00101022)
    self.assertEqual(de.VR, 'DS')
    self.assertEqual(de.value, 90 / 2 ** 2)


  def test_write_private_tags(self):
    # Assemble
    blueprint = Blueprint()

    # Act
    blueprint.add_virtual_element(
      StaticElement(0x00115000, 'IS', 124, "name")
    )

    # Assert
    self.assertIn(0x00115000, blueprint)
    self.assertIn(0x001150FE, blueprint)
    self.assertIn(0x001150FF, blueprint)
    self.assertIn(0x00110050, blueprint)


  def test_overwrite_private_tag(self):
    blueprint = Blueprint()
    blueprint.add_virtual_element(StaticElement(0x00115000, 'IS', 124, name="name"))

    blueprint.add_virtual_element(
      StaticElement(0x00115000, 'LO', "A Test string", name="overwritten name")
    )

  def test_attempt_write_reserved_tag(self):
    blueprint = Blueprint()
    with self.assertLogs('dicomnode', ERROR) as cm:
      tag = StaticElement(0x00110050, 'IS', 124, name="name")
      self.assertRaises(IncorrectlyConfigured,blueprint.add_virtual_element, tag)
    self.assertIn('ERROR:dicomnode:Dicom node will automatically allocate private tag ranges', cm.output)

    with self.assertLogs('dicomnode', ERROR) as cm:
      tag = StaticElement(0x001150FE, 'IS', 124, name="name")
      self.assertRaises(IncorrectlyConfigured,blueprint.add_virtual_element, tag)
    self.assertIn('ERROR:dicomnode:You are trying to add a private tag, that have been reserved by Dicomnode', cm.output)

    with self.assertLogs('dicomnode', ERROR) as cm:
      tag = StaticElement(0x001150FF, 'IS', 124, name="name")
      self.assertRaises(IncorrectlyConfigured,blueprint.add_virtual_element, tag)
    self.assertIn('ERROR:dicomnode:You are trying to add a private tag, that have been reserved by Dicomnode', cm.output)

  def test_build_a_private_Sequence(self):
    blueprint = Blueprint([StaticElement(0x0008_0016, 'UI', SecondaryCaptureImageStorage)])
    sequence_blueprint = Blueprint([StaticElement(0x0008_0016, 'UI', SecondaryCaptureImageStorage)])
    sequence_sequence_blueprint = Blueprint([StaticElement(0x0008_0016, 'UI', SecondaryCaptureImageStorage)])

    blueprint.add_virtual_element(SequenceElement(0x00115000, [sequence_blueprint], "Sequence 1"))
    sequence_blueprint.add_virtual_element(SequenceElement(0x00115000, [sequence_sequence_blueprint], "Sequence 2"))
    sequence_sequence_blueprint.add_virtual_element(StaticElement(0x00135011,'LO', 'Sequence'))

    dataset = Dataset()
    build_dataset = self.factory.build_instance(dataset, blueprint)

  def test_factory_build_with_copy(self):
    dataset = Dataset()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.PatientName = "TestName"
    dataset.PatientID = "12345678"

    blueprint = Blueprint()
    blueprint.add_virtual_element(StaticElement(0x0008_0016, 'UI', CTImageStorage))
    blueprint.add_virtual_element(CopyElement(0x00100010))
    blueprint.add_virtual_element(CopyElement(0x00100020))

    build_dataset = self.factory.build_instance(dataset,
                                                blueprint)
    self.assertIn(0x00100010, build_dataset)
    self.assertIn(0x00100020, build_dataset)

  def test_build_a_series_with_rescaling(self):
    test_blueprint = Blueprint([
      CopyElement(0x00100010),
      StaticElement(0x0008_0016, 'UI', SecondaryCaptureImageStorage),
      FunctionalElement(0x01115001, 'IS', lambda env: env.instance_number + 34),
    ])

    # Note that images are store z,y,x
    test_image: numpy.ndarray[Tuple[int,int,int], Any] = numpy.random.normal(0,1,(13, 12, 11))

    produced_series = self.factory.build_series(
      test_image,
      test_blueprint,
      self.parent_series
    )

    self.assertEqual(len(produced_series), 13)
    for dataset in produced_series.datasets:
      self.assertEqual(dataset.Columns,11)
      self.assertEqual(dataset.Rows,12)
      self.assertEqual(dataset.PatientName, self.patient_name)
      self.assertIn(0x7fe00010, dataset)
      self.assertEqual(dataset.SmallestImagePixelValue, 0)
      # Some numeric unstably might cause Numpy to round down.
      #self.assertGreaterEqual(dataset.LargestImagePixelValue, 65534)

  def test_create_dataset_with_private_tags(self):
    bp = Blueprint([])

    bp.add_virtual_element(StaticElement(0x0008_0016, 'UI', SecondaryCaptureImageStorage))
    bp.add_virtual_element(FunctionalElement(0x0008_0018, 'UI', add_UID_tag))
    bp.add_virtual_element(StaticElement(0x0020_0013, 'IS', 1))

    bp.add_virtual_element(StaticElement(0x3003_0199, 'IS', 14710))
    bp.add_virtual_element(StaticElement(0x3003_0101, 'IS', 14710))
    bp.add_virtual_element(StaticElement(0x3003_0102, 'IS', 14710))
    bp.add_virtual_element(StaticElement(0x3005_0101, 'IS', 14710))
    bp.add_virtual_element(StaticElement(0x3003_0182, 'IS', 14710))
    bp.add_virtual_element(StaticElement(0x3003_0181, 'IS', 14710))

    bp.add_virtual_element(StaticElement(0x3005_0199, 'IS', 14710))
    bp.add_virtual_element(StaticElement(0x3005_0102, 'IS', 14710))
    bp.add_virtual_element(StaticElement(0x3005_01AE, 'IS', 14710))
    bp.add_virtual_element(StaticElement(0x3003_01AE, 'IS', 14710))

    factory = DicomFactory()
    pivot = Dataset()
    build_dataset = factory.build_instance(pivot, bp)
    prev_tag = None

    for str_ in build_dataset[0x3003_01FE]:
      if prev_tag is None:
        prev_tag = str_
        continue
      self.assertLess(prev_tag, str_)
      prev_tag = str_

    prev_tag = None
    for str_ in build_dataset[0x3005_01FE]:
      if prev_tag is None:
        prev_tag = str_
        continue
      self.assertLess(prev_tag, str_)
      prev_tag = str_
