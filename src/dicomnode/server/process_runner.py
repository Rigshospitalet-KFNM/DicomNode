# Python standard library
from dataclasses import dataclass
from logging import getLogger
from os import chdir
from pathlib import Path
from typing import Type

# Dicomnode imports
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.lib.io import TemporaryWorkingDirectory
from dicomnode.lib.logging import set_logger, LoggerConfig
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import PipelineOutput, NoOutput

@dataclass
class ProcessRunnerArgs:
  input_container : InputContainer
  log_config : LoggerConfig
  process_path : Path | None
  patient_id : str


class ProcessRunner():
  """Runner class for the processing.
  """
  def __init__(self, args : ProcessRunnerArgs) -> None:
    self.logger = getLogger(DICOMNODE_LOGGER_NAME)
    set_logger(self.logger, args.log_config)

    if args.process_path is None:
      self._main(args.patient_id, args.input_container)
    else:
      with TemporaryWorkingDirectory(args.process_path):
        self._main(args.patient_id, args.input_container)

  def process_signal_handler_SIGINT(self, signal_, frame): #pragma: no cover
    # Same as above
    self.logger.critical("Process killed by Parent")
    exit(1)

  def _main(self,patient_id, input_container: InputContainer):
    try:
      result = self.process(input_container)

      if self._dispatch(result):
        self.logger.info(f"Process has handled {patient_id}")
      else:
        self.logger.error(f"Process was unable to dispatch {patient_id}")
    except Exception as exception:
      self.logger.critical(f"Encountered an exception {exception} in process function")

  def process(self, input_container: InputContainer) -> PipelineOutput:
    return NoOutput()

  def _dispatch(self, output: PipelineOutput) -> bool:
    """This function is responsible for triggering exporting of data and handling errors.
      You should consider if it's possible to create your own output
      rather than overwriting this function

      Args:
        output: PipelineOutput - the output to be exported
      Returns:
        bool - If the output was successful in exporting the data.
    """
    try:
      success = output.send()
    except Exception as exception:
      self.logger.critical(f"Exception raised {exception} in user Output Send Function")
      success = False
    return success
