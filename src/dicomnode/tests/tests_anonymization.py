from unittest import TestCase

from pydicom import Dataset, DataElement, Sequence
from pydicom.valuerep import VR
from pydicom.uid import generate_uid, MediaStorageDirectoryStorage
from dicomnode.lib.anonymization import anonymize_dataset

from dicomnode.lib.studyTree import DicomTree, IdentityMapping

class Lib_anonymization(TestCase):
  def setUp(self) -> None:
    self.patientID_1 = "testID_1"
    self.patientID_2 = "testID_2"
    self.patientID_3 = "testID_æøå"

    self.patientIDs = [self.patientID_1, self.patientID_2, self.patientID_3]

    self.patientName_1 = "TestP1"
    self.patientName_2 = "TestP2"
    self.patientName_3 = "TestæøåP3"

    self.patientNames = [self.patientID_1, self.patientName_2, self.patientName_3]

    self.studyUID_1 = generate_uid()
    self.studyUID_2 = generate_uid()
    self.studyUID_3 = generate_uid()

    self.seriesUID_1 = generate_uid()
    self.seriesUID_2 = generate_uid()
    self.seriesUID_3 = generate_uid()

    self.dataset_1 = Dataset()
    self.dataset_1_SOPInstanceUID = generate_uid()
    self.dataset_1.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    self.dataset_1.SOPInstanceUID = self.dataset_1_SOPInstanceUID
    self.dataset_1.SeriesInstanceUID = self.seriesUID_1
    self.dataset_1.StudyInstanceUID = self.studyUID_1
    self.dataset_1.PatientID = self.patientID_1
    self.dataset_1.PatientName = self.patientName_1
    self.dataset_1.ensure_file_meta()


    self.dataset_2 = Dataset()
    self.dataset_2_SOPInstanceUID = generate_uid()
    self.dataset_2.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    self.dataset_2.SOPInstanceUID = self.dataset_2_SOPInstanceUID
    self.dataset_2.SeriesInstanceUID = self.seriesUID_1
    self.dataset_2.StudyInstanceUID = self.studyUID_1
    self.dataset_2.PatientID = self.patientID_1
    self.dataset_2.PatientName = self.patientName_1
    self.dataset_2.ensure_file_meta()

    self.dataset_3 = Dataset()
    self.dataset_3_SOPInstanceUID = generate_uid()
    self.dataset_3.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    self.dataset_3.SOPInstanceUID = self.dataset_3_SOPInstanceUID
    self.dataset_3.SeriesInstanceUID = self.seriesUID_2
    self.dataset_3.StudyInstanceUID = self.studyUID_1
    self.dataset_3.PatientID = self.patientID_1
    self.dataset_3.PatientName = self.patientName_1
    self.dataset_3.ensure_file_meta()
    self.dataset_3_seq_ds_1 = Dataset()
    self.dataset_3_seq_ds_1.ReviewerName = "Should^be^anonymized!"
    self.dataset_3_seq_ds_2 = Dataset()
    self.dataset_3_seq_ds_2.ReviewerName = "Should^be^anonymized!"
    self.dataset_3.ReferencedPatientSequence = Sequence([self.dataset_3_seq_ds_1, self.dataset_3_seq_ds_2])

    self.dataset_4 = Dataset()
    self.dataset_4_SOPInstanceUID = generate_uid()
    self.dataset_4.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    self.dataset_4.SOPInstanceUID = self.dataset_4_SOPInstanceUID
    self.dataset_4.SeriesInstanceUID = self.seriesUID_1
    self.dataset_4.StudyInstanceUID = self.studyUID_2
    self.dataset_4.PatientID = self.patientID_2
    self.dataset_4.PatientName = self.patientName_2
    self.dataset_4[0x00101001] = DataElement(0x00101001, VR.PN, "Should^be^anonymized!")
    self.dataset_4.ensure_file_meta()

    self.dataset_5 = Dataset()
    self.dataset_5_SOPInstanceUID = generate_uid()
    self.dataset_5.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    self.dataset_5.SOPInstanceUID = self.dataset_5_SOPInstanceUID
    self.dataset_5.SeriesInstanceUID = self.seriesUID_1
    self.dataset_5.StudyInstanceUID = self.studyUID_2
    self.dataset_5.PatientID = self.patientID_2
    self.dataset_5.PatientName = self.patientName_2
    self.dataset_5.ResponsiblePerson = "Should^be^removed"
    self.dataset_5.ensure_file_meta()

    self.dataset_6 = Dataset()
    self.dataset_6_SOPInstanceUID = generate_uid()
    self.dataset_6.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    self.dataset_6.SOPInstanceUID = self.dataset_5_SOPInstanceUID
    self.dataset_6.SeriesInstanceUID = self.seriesUID_3
    self.dataset_6.StudyInstanceUID = self.studyUID_3
    self.dataset_6.PatientID = self.patientID_3
    self.dataset_6.PatientName = self.patientName_3

    self.datasets = [self.dataset_1, self.dataset_2, self.dataset_3, self.dataset_4, self.dataset_5]
    self.dt = DicomTree(self.datasets)
    self.im = IdentityMapping()
    self.im.fill_from_DicomTree(self.dt)

  def test_anonymization(self):
    anonymization_function = anonymize_dataset(self.im)
    self.dt.apply_mapping(anonymization_function, self.im)

    for ds in self.datasets:
      self.assertNotIn(ds.PatientName,self.patientNames)
      self.assertNotIn(ds.PatientID, self.patientIDs)

    self.assertEqual(self.dataset_3_seq_ds_1.ReviewerName, "Anon_Reviewer Name")
    self.assertEqual(self.dataset_3_seq_ds_2.ReviewerName, "Anon_Reviewer Name")
    self.assertEqual(self.dataset_4.OtherPatientNames, "Anon_Other Patient_Names")
    self.assertEqual(self.dataset_5.ResponsiblePersone, "Anon_Responsible Person")
