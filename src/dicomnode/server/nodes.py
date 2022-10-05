from pathlib import Path
from pynetdicom import AE, evt, AllStoragePresentationContexts

from dicomnode.server.inputDataClass import AbstractInputDataClass


class AbstractPipeline():
  """Abstract Class for a Pipeline, which acts a SCP

  Requires the following attributes before it can be instanciated.
    * ae_title : str
    * config_path : Union[str, PathLike]
    * process : Callable

  """

  # Required overwritten Properties
  ae_title = "Your_AE_TITLE"
  InputDataClass = AbstractInputDataClass
  log_path = ""

  # AE configuration tags
  ip = ''
  port = 104
  supported_contexts = AllStoragePresentationContexts


  def __init__(self) -> None:

    # Validates that Pipeline is configured correctly.
    self.ae = AE(ae_title = self.ae_title)

    self.ae.supported_contexts = self.supported_contexts

    self.ae.start_server((self.ip, self.port), evt_handlers=[
      (evt.EVT_C_STORE, self.__handle_store, [self])
    ])

  def __handle_store(event, self):
    self.log()
    if not self.filter(event):
      return 0x0000

  def log(self):
    pass

  def filter(self, event) -> bool:
    return False

  def process(self, InputDataClass):
    pass

  def dispatch(self, process_return_value):
    pass

  def post_init(self) -> None:
    """This function is called just before the server is started.
      The idea being that a user change this function to run some arbitrary code before the Dicom node starts.
      This would often be 


    """
    pass