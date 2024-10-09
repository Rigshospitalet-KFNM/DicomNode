import logging
import shutil
from random import randint
from pathlib import Path
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from unittest import TestCase

from dicomnode.dicom.dimse import Address
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.server.output import DicomOutput, FileOutput
from tests.helpers import get_test_ae


class OutputTests(TestCase):
  def setUp(self) -> None:
    self.path = Path(self._testMethodName)
    self.path.mkdir()

    self.nodePort = randint(1025,65535)
    self.endpointPort = randint(1025,65535)
    while self.nodePort == self.endpointPort:
      self.endpointPort = randint(1025,65535)
    self.endpoint =get_test_ae(
      self.endpointPort,
      self.nodePort,
      logging.getLogger("dicomnode")
    )
    self.endpointAddress = Address('localhost', self.endpointPort, self.endpoint.ae_title)

    self.dataset_1 = Dataset()
    self.dataset_1.SOPClassUID = SecondaryCaptureImageStorage
    self.dataset_1.PatientID = "1623910515"
    self.dataset_1.StudyInstanceUID = gen_uid()
    self.dataset_1.SeriesInstanceUID = gen_uid()
    make_meta(self.dataset_1)

    self.datasets = [self.dataset_1]

  def tearDown(self) -> None:
    self.endpoint.shutdown()
    shutil.rmtree(self.path)


  def test_file_output(self):
    output = FileOutput([(self.path, self.datasets)])
    self.assertTrue(output.send())
    self.assertTrue(Path(self.path / self.dataset_1.StudyInstanceUID.name / self.dataset_1.SeriesInstanceUID.name / (self.dataset_1.SOPInstanceUID.name + ".dcm")).exists())


  def test_dicom_output_send(self):
    with self.assertLogs("dicomnode", logging.DEBUG) as cm:
      output = DicomOutput([(self.endpointAddress, self.datasets)], "PIPELINE_AE")
      self.assertTrue(output.send())
    self.assertIn('INFO:dicomnode:Received C Store',cm.output)

  def test_dicom_output_send_failure(self):
    address = Address('localhost', 150, "WrongAE")
    with self.assertLogs("dicomnode", logging.DEBUG) as cm:
      output = DicomOutput([(address, self.datasets)], "PIPELINE_AE")
      self.assertFalse(output.send())
    self.assertIn("ERROR:dicomnode:Could not send to images to WrongAE", cm.output)
