from datetime import datetime, date, time



from pydicom import DataElement, Dataset
from pydicom.tag import Tag
from pydicom.uid import SecondaryCaptureImageStorage
from typing import Any, List
from unittest import TestCase

from dicomnode.constants import DICOMNODE_IMPLEMENTATION_UID
from dicomnode.lib.dicom import gen_uid
from dicomnode.lib.dicomFactory import AttrElement, CopyElement, DicomFactory, DiscardElement, FillingStrategy, \
  general_series_study_header, SeriesHeader, Blueprint, SeriesElement, StaticElement, SOP_common_header
from dicomnode.lib.exceptions import InvalidTagType, IncorrectlyConfigured

class HeaderBlueprintTestCase(TestCase):
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
        self.assertTrue(False)

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
        self.assertTrue(False)

    for i,virtual_element in enumerate(self.blueprint_2):
      if i == 0:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_id_conflict))
      if i == 1:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_sex))
      if i == 2:
        self.assertTrue(False)

    for i,virtual_element in enumerate(bluer_print):
      if i == 0:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_name))
      if i == 1:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_id_conflict))
      if i == 2:
        self.assertEqual(id(virtual_element), id(self.virtual_patient_sex))
      if i == 3:
        self.assertTrue(False)

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
        self.assertTrue(False)

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


class HeaderTestCase(TestCase):
  def setUp(self) -> None:
    self.header = SeriesHeader()
    self.de_1 = DataElement(0x00100010, 'PN', 'Face^Mace^to')
    self.tag_list = [self.de_1]

  def test_header_with_args_iter(self):
    header = SeriesHeader(self.tag_list) # type: ignore The type checker is high here

    self.assertEqual(id(self.de_1), id(header[0x00100010]))
    for i, tag in enumerate(header):
      if i == 0:
        self.assertEqual(id(self.de_1), id(tag))
      else:
        self.assertTrue(False)

  def test_header_InvalidTagType(self):
    self.assertRaises(InvalidTagType, self.header.add_tag,StaticElement(0x00100010, 'PN', 'Face^Mace^To'))

  def test_header_wrong_element(self):
    self.assertRaises(ValueError, self.header.__setitem__, 0x00100020, self.de_1)

  def test_header_set_element(self):
    self.header[0x00100010] = self.de_1

class testFactory(DicomFactory):
  def make_series(self, header: SeriesHeader, image: Any) -> List[Dataset]:
    return super().make_series(header, image)

class DicomFactoryTestClass(TestCase):
  def setUp(self) -> None:
    self.my_desc = "My description"
    self.factory = testFactory(filling_strategy=None)
    self.factory.series_description = self.my_desc

  def tearDown(self) -> None:
    pass

  def test_make_series_dont_call_super(self):
    self.assertRaises(NotImplementedError, self.factory.make_series, SeriesHeader(), None)

  def test_AttributeElementCorporealialize(self):
    attribute_element = AttrElement(0x0008103E, 'LO','series_description')

    de = attribute_element.corporealialize(self.factory, Dataset())

    self.assertIsInstance(de, DataElement)
    self.assertEqual(de.tag, 0x0008103E)
    self.assertEqual(de.VR, 'LO')
    self.assertEqual(de.value, self.my_desc)

  def test_CopyElementCorporealialize(self):
    copy_element = CopyElement(0x00100020)
    optional_copy_element = CopyElement(0x00100020, Optional=True)

    dataset = Dataset()
    self.assertRaises(KeyError, copy_element.corporealialize, self.factory, dataset)
    self.assertIsNone(optional_copy_element.corporealialize(self.factory, dataset))
    cpr = '1502799995'
    dataset.PatientID = cpr

    data_element = copy_element.corporealialize( self.factory, dataset)
    data_element_optional = optional_copy_element.corporealialize(self.factory, dataset)

    if data_element is not None and data_element_optional is not None:
      self.assertIsInstance(data_element, DataElement)
      self.assertIsInstance(data_element_optional, DataElement)

      self.assertEqual(data_element.tag, 0x00100020)
      self.assertEqual(data_element_optional.tag, 0x00100020)

      self.assertEqual(data_element.value, cpr)
      self.assertEqual(data_element_optional.value, cpr)
    else:
      self.assertEqual(1,2)

  def test_discard_element(self):
    discard_element = DiscardElement(0x00100020)
    dataset = Dataset()
    cpr = '1502799995'
    dataset.PatientID = cpr
    self.assertIsNone(discard_element.corporealialize(self.factory, dataset))

  def test_series_element(self):
    series_element_no_arg = SeriesElement(0x0020000E, 'UI', gen_uid)

    dataset = Dataset()
    de = series_element_no_arg.corporealialize(self.factory, dataset)

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

    de = series_element.corporealialize(self.factory, dataset)

    self.assertEqual(de.tag, 0x00101022)
    self.assertEqual(de.VR, 'DS')
    self.assertEqual(de.value, 90 / 2 ** 2)

  def test_static_element(self):
    patient_name = 'Face^Booty^Mac'
    static_element = StaticElement(0x00100010, 'PN', patient_name)
    de = static_element.corporealialize(self.factory, Dataset())
    if isinstance(de, DataElement):
      self.assertEqual(de.tag, 0x00100010)
      self.assertEqual(de.VR, 'PN')
      self.assertEqual(de.value, patient_name)
    else:
      self.assertTrue(False)


  def test_create_header(self):
    dataset = Dataset()
    self.assertRaises(IncorrectlyConfigured, self.factory.make_series_header, dataset)

    headerBP = general_series_study_header
    self.assertRaises(IncorrectlyConfigured, self.factory.make_series_header, dataset, **{'elements' :  headerBP})

    dataset.Modality = 'OT'
    dataset.PatientSize = 50

    header = self.factory.make_series_header(dataset, headerBP, FillingStrategy.DISCARD)
    headerCopy = self.factory.make_series_header(dataset, headerBP, FillingStrategy.COPY)

    self.assertNotIn(0x00101020, header)
    self.assertIn(0x00101020, headerCopy)

