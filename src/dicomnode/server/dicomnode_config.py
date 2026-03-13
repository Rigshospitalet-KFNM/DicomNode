""""""

__author__ = "Demiguard"

# Python standard library
from dataclasses import dataclass
from logging import INFO

# Third party modules

# Dicomnode modules
from dicomnode.lib.io import Directory, File


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
  LOG_OUTPUT : str | None = None
  LOG_WHEN : str | None = None
  LOG_LEVEL : int | None = None
  LOG_FORMAT : str | None = None


@dataclass
class DicomnodeConfig:
  STUDY_EXPIRATION_DAYS : int
  PATIENT_IDENTIFIER_TAG : int
  LAZY_STORAGE : bool

  # AE CONFIG
  AE_TITLE : str
  IP : str
  PORT : int

  REQUIRED_CALLED_AET : bool

  # PATHS
  ARCHIVE_DIRECTORY : Directory | None
  PROCESSING_DIRECTORY : Directory | None
  RUN_FILE : File | None

  # LOGGING
  LOG_OUTPUT : str | None
  LOG_WHEN : str | None
  LOG_LEVEL : int | None
  LOG_FORMAT : str | None



def default_to(value, default):
  return value if value is not None else default

def config_from_raw(config=DicomnodeConfigRaw()) -> DicomnodeConfig:
  study_expiration_days  = default_to(config.STUDY_EXPIRATION_DAYS, 14)
  patient_identifier_tag = default_to(config.PATIENT_IDENTIFIER_TAG, 0x0010_0020)
  lazy_storage = default_to(config.LAZY_STORAGE, False)

  ae_title = default_to(config.AE_TITLE, "DICOMNODE")
  ip = default_to(config.IP, "127.0.0.1")
  port = default_to(config.PORT, 104)

  required_called_aet = default_to(config.REQUIRED_CALLED_AET, False)

  archive_directory = Directory(config.ARCHIVE_DIRECTORY) if config.ARCHIVE_DIRECTORY is not None else None
  processing_directory = Directory(config.PROCESSING_DIRECTORY) if config.PROCESSING_DIRECTORY is not None else None
  run_file = File(config.RUN_FILE) if config.RUN_FILE is not None else None

  log_output = None


  log_when = default_to(config.LOG_WHEN, "w0")
  log_level = default_to(config.LOG_LEVEL, INFO)
  log_format = default_to(config.LOG_FORMAT, "[%(asctime)s] |%(thread_id)d| %(name)s - %(levelname)s - %(message)s")

  return DicomnodeConfig(
    study_expiration_days,
    patient_identifier_tag,
    lazy_storage,
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
    log_format
  )