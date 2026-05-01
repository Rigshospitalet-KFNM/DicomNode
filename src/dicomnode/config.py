from __future__ import annotations

__author__ = "Demiguard"

# Python standard library
from dataclasses import dataclass
from logging import INFO
from pathlib import Path

from typing import TextIO, TYPE_CHECKING

# Third party modules

# Dicomnode modules

if TYPE_CHECKING:
  from dicomnode.lib.io import Directory, File
  from dicomnode.dicom import DicomIdentifier
  from dicomnode.data_structures.optional import OptionalPath


@dataclass
class DicomnodeConfigRaw:
  STUDY_EXPIRATION_DAYS : int | None = None
  PATIENT_IDENTIFIER_TAG : int | None = None
  LAZY_STORAGE : bool | None = None

  # AE CONFIG
  AE_TITLE : str | None = None
  IP : str | None = None
  PORT : int | None = None

  REQUIRED_CALLED_AET : bool | None = None

  # PATHS
  ARCHIVE_DIRECTORY : str | None = None
  PROCESSING_DIRECTORY : str | None = None
  RUN_FILE : str | None = None

  # LOGGING
  LOG_OUTPUT : TextIO | Path | str | None = None
  LOG_WHEN : str | None = None
  LOG_LEVEL : int | None = None
  LOG_FORMAT : str | None = None
  LOG_DATE_FORMAT : str | None = None
  LOG_NUMBER_OF_BACK_UPS: int | None = None


@dataclass
class DicomnodeConfig:
  STUDY_EXPIRATION_DAYS : int
  PATIENT_IDENTIFIER_TAG : int
  LAZY_STORAGE : bool

  IDENTIFIER : DicomIdentifier

  # AE CONFIG
  AE_TITLE : str
  IP : str
  PORT : int


  REQUIRED_CALLED_AET : bool

  # PATHS
  ARCHIVE_DIRECTORY : OptionalPath
  PROCESSING_DIRECTORY : OptionalPath
  RUN_FILE : File | None

  # LOGGING
  LOG_OUTPUT : TextIO | str | None
  LOG_WHEN : str
  LOG_LEVEL : int
  LOG_FORMAT : str
  LOG_DATE_FORMAT : str
  LOG_NUMBER_OF_BACK_UPS : int



def default_to(value, default):
  return value if value is not None else default

def config_from_raw(config=DicomnodeConfigRaw()) -> DicomnodeConfig:
  from dicomnode.dicom import DicomIdentifier
  from dicomnode.data_structures.optional import OptionalPath

  study_expiration_days  = default_to(config.STUDY_EXPIRATION_DAYS, 14)
  patient_identifier_tag = default_to(config.PATIENT_IDENTIFIER_TAG, 0x0010_0020)
  lazy_storage = default_to(config.LAZY_STORAGE, False)

  identifier = DicomIdentifier(identifying_tag=patient_identifier_tag)

  ae_title = default_to(config.AE_TITLE, "DICOMNODE")
  ip = default_to(config.IP, "127.0.0.1")
  port = default_to(config.PORT, 104)

  required_called_aet = default_to(config.REQUIRED_CALLED_AET, False)

  archive_directory = OptionalPath(config.ARCHIVE_DIRECTORY)
  processing_directory = OptionalPath(config.PROCESSING_DIRECTORY)
  run_file = File(config.RUN_FILE) if config.RUN_FILE is not None else None

  log_output = None

  log_when = default_to(config.LOG_WHEN, "w0")
  log_level = default_to(config.LOG_LEVEL, INFO)
  log_format = default_to(config.LOG_FORMAT, "[%(asctime)s] |%(thread_id)d| %(name)s - %(levelname)s - %(message)s")
  log_date_format = default_to(config.LOG_DATE_FORMAT, "%Y/%m/%d %H:%M:%S")
  log_number_of_back_ups = default_to(config.LOG_NUMBER_OF_BACK_UPS, 8)

  return DicomnodeConfig(
    study_expiration_days,
    patient_identifier_tag,
    lazy_storage,
    identifier,
    ae_title,
    ip,
    port,
    required_called_aet,
    archive_directory,
    processing_directory,
    run_file,
    log_output,
    log_when,
    log_level,
    log_format,
    log_date_format,
    log_number_of_back_ups
  )