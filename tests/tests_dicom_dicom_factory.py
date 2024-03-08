__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from datetime import datetime, date, time
from logging import ERROR

# Third Party Modules
from pydicom import DataElement, Dataset
from pydicom.tag import Tag
from pydicom.uid import SecondaryCaptureImageStorage
from typing import Any, List
from unittest import TestCase

# Dicomnode modules
from dicomnode.constants import DICOMNODE_IMPLEMENTATION_UID
from dicomnode.dicom import gen_uid
from dicomnode.dicom.dicom_factory import CopyElement, DicomFactory, DiscardElement, FunctionalElement, FillingStrategy, \
  general_series_blueprint, SeriesHeader, Blueprint, SeriesElement, StaticElement, SOP_common_blueprint, image_plane_blueprint, InstanceCopyElement, _add_InstanceNumber, \
  InstanceEnvironment, SequenceElement
from dicomnode.lib.exceptions import InvalidTagType, IncorrectlyConfigured, HeaderConstructionFailure

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
        self.assertTrue(False) # pragma: no cover

  def test_header_InvalidTagType(self):
    self.assertRaises(InvalidTagType, self.header.add_tag,StaticElement(0x00100010, 'PN', 'Face^Mace^To'))

  def test_header_wrong_element(self):
    self.assertRaises(ValueError, self.header.__setitem__, 0x00100020, self.de_1)

  def test_header_set_element(self):
    self.header[0x00100010] = self.de_1

  def test_virtual_name_truncated_name(self):
    CopyElement(0x10110011, name="a" * 65)

  def test_static_element_string(self):
    self.assertEqual(str(StaticElement(0x00100010, 'PN', 'Test Name')), 
                     "<StaticElement: (0010, 0010) PN Test Name>")


class testFactory(DicomFactory):
  def build_from_header(self, header: SeriesHeader, image: Any) -> List[Dataset]:
    return super().build_from_header(header, image)

class DicomFactoryTestClass(TestCase):
  def setUp(self) -> None:
    self.my_desc = "My description"
    self.factory = testFactory()
    self.factory.series_description = self.my_desc

  def tearDown(self) -> None:
    pass

  def test_make_series_dont_call_super(self):
    self.assertRaises(NotImplementedError, self.factory.build_from_header, SeriesHeader(), None)


  def test_CopyElementCorporealialize(self):
    copy_element = CopyElement(0x00100020)
    optional_copy_element = CopyElement(0x00100020, Optional=True)

    dataset = Dataset()
    self.assertRaises(KeyError, copy_element.corporealialize, self.factory, [dataset])
    self.assertIsNone(optional_copy_element.corporealialize(self.factory, [dataset]))
    cpr = '1502799995'
    dataset.PatientID = cpr

    data_element = copy_element.corporealialize( self.factory, [dataset])
    data_element_optional = optional_copy_element.corporealialize(self.factory, [dataset])

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
    self.assertIsNone(discard_element.corporealialize(self.factory, [dataset]))

  def test_series_element(self):
    series_element_no_arg = SeriesElement(0x0020000E, 'UI', gen_uid)

    dataset = Dataset()
    de = series_element_no_arg.corporealialize(self.factory, [dataset])

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

    de = series_element.corporealialize(self.factory, [dataset])

    self.assertEqual(de.tag, 0x00101022)
    self.assertEqual(de.VR, 'DS')
    self.assertEqual(de.value, 90 / 2 ** 2)

  def test_static_element(self):
    patient_name = 'Face^Booty^Mac'
    static_element = StaticElement(0x00100010, 'PN', patient_name)
    de = static_element.corporealialize(self.factory, [Dataset()])
    if isinstance(de, DataElement):
      self.assertEqual(de.tag, 0x00100010)
      self.assertEqual(de.VR, 'PN')
      self.assertEqual(de.value, patient_name)
    else:
      self.assertTrue(False)


  def test_create_header(self):
    dataset = Dataset()
    headerBP = general_series_blueprint

    dataset.Modality = 'OT'
    dataset.PatientSize = 50
    dataset.PatientPosition = 'FFP'

    header = self.factory.make_series_header([dataset], headerBP, FillingStrategy.DISCARD)
    headerCopy = self.factory.make_series_header([dataset], headerBP, FillingStrategy.COPY)

    self.assertNotIn(0x00101020, header)
    self.assertIn(0x00101020, headerCopy)

  def test_create_instance_copy(self):
    datasets = []

    for dataset_index in reversed(range(1,11,1)):
      dataset = Dataset()
      dataset.InstanceNumber = dataset_index
      dataset.ImagePositionPatient = [0,0,dataset_index - 1]
      datasets.append(dataset)

    blueprint = Blueprint([
      FunctionalElement(0x00200013, 'IS', _add_InstanceNumber),
      InstanceCopyElement(0x00200032, 'DS')
    ])

    header = self.factory.make_series_header(datasets, blueprint)

    instance_copy_element = header[0x00200032]
    # This is mostly to make my type checker happy
    if not isinstance(instance_copy_element, InstanceCopyElement):
      raise AssertionError

    instance_copy_element_refence_2 = instance_copy_element.corporealialize(self.factory, datasets)
    self.assertIs(instance_copy_element, instance_copy_element_refence_2)
    for i in range(1,11,1):
      instance_environment = InstanceEnvironment(instance_number=i)
      data_element = instance_copy_element.produce(instance_environment)

      self.assertIsInstance(data_element, DataElement)
      self.assertEqual(data_element.tag, 0x00200032)
      self.assertListEqual(list(data_element.value), [0,0, i - 1])


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
    blueprint = Blueprint()
    sequence_blueprint = Blueprint()
    sequence_sequence_blueprint = Blueprint()

    blueprint.add_virtual_element(SequenceElement(0x00115000, [sequence_blueprint], "Sequence 1"))
    sequence_blueprint.add_virtual_element(SequenceElement(0x00115000, [sequence_sequence_blueprint], "Sequence 2"))
    sequence_sequence_blueprint.add_virtual_element(StaticElement(0x00135011,'LO', 'Sequence'))

    dataset = Dataset()
    build_dataset = self.factory.build(dataset, blueprint)

  def test_corporealialize_into_nothing(self):
    ce = CopyElement(0x00100010)
    self.assertRaises(ValueError, ce.corporealialize, self.factory, [])

  def test_header_string(self):
    datasets = []

    for dataset_index in reversed(range(1,11,1)):
      dataset = Dataset()
      dataset.InstanceNumber = dataset_index
      dataset.ImagePositionPatient = [0,0,dataset_index - 1]
      datasets.append(dataset)

    blueprint = Blueprint([
      FunctionalElement(0x00200013, 'IS', _add_InstanceNumber),
      InstanceCopyElement(0x00200032, 'DS')
    ])

    header = self.factory.make_series_header(datasets, blueprint)
    self.assertEqual(str(header), "SeriesHeader with 2 tags:\n    Tag: (0020, 0013) - FunctionalElement\n    Tag: (0020, 0032) - InstanceCopyElement\n")

  def test_get_default_blueprint(self):
    self.assertIsInstance(self.factory.get_default_blueprint(), Blueprint)

  def test_get_series_header_empty_series(self):
    self.assertRaises(ValueError, self.factory.make_series_header, [], self.factory.get_default_blueprint())

  def test_make_series_header_fails_to_copy(self):
    dataset = Dataset()
    #dataset.PatientName = "TestName"

    blueprint = Blueprint()
    blueprint.add_virtual_element(CopyElement(0x00100010))

    self.assertRaises(HeaderConstructionFailure,self.factory.make_series_header, [dataset], blueprint)

  def test_factory_build_with_copy(self):
    dataset = Dataset()
    dataset.PatientName = "TestName"
    dataset.PatientID = "12345678"

    blueprint = Blueprint()
    blueprint.add_virtual_element(CopyElement(0x00100010))

    build_dataset = self.factory.build(dataset, blueprint, filling_strategy=FillingStrategy.COPY)
    self.assertIn(0x00100010, build_dataset)
    self.assertIn(0x00100020, build_dataset)

