from pathlib import Path
from pynetdicom import AE, evt, AllStoragePresentationContexts

from dicomnode.server.inputDataClass import AbstractInputDataClass

from sys import stdout
import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler

class AbstractPipeline():
  """Abstract Class for a Pipeline, which acts a SCP

  Requires the following attributes before it can be instantiated.
    * ae_title : str
    * config_path : Union[str, PathLike]
    * process : Callable

  """

  # Input configuration
  input = {}
  instance_identifier_tag = 0x00100020 # Patient ID


  # AE configuration tags
  ae_title = "Your_AE_TITLE"
  ip = ''
  port = 104
  supported_contexts = AllStoragePresentationContexts
  require_called_aet = True
  require_calling_aet = []

  #Logging Configuration
  backup_weeks = 8
  log_path = None
  log_level = logging.DEBUG
  log_format = "%(asctime)s %(name)s %(message)s"
  disable_pynetdicom_logger = True

  def close(self) -> None:
    """Closes all connections active connections.
      If your application includes additional connections, you should overwrite this method,
      And close any connections and call the super function.
    """
    self.ae.shutdown()

  def open(self, *args, blocking=True, **kwargs) -> None:
    """Opens all connections active connections.
      If your application includes additional connections, you should overwrite this method,
      And open any connections and call the super function.

      Keyword Args:
        blocking (bool) : if true, this functions doesn't return.
    """
    self.ae.require_called_aet = self.require_called_aet
    self.ae.require_calling_aet = self.require_calling_aet

    #self.logger.debug("Starting Server")

    self.ae.start_server(
      (self.ip,self.port),
      block=blocking,
      evt_handlers=self.evt_handlers)



  def __init__(self, start=True) -> None:
    # It's easiest to define it here, since functions are not defined until class is instantiated
    # Also you shouldn't be touching this
    self.evt_handlers = [
      (evt.EVT_C_STORE, self.__handle_store),
      (evt.EVT_ACCEPTED, self.__association_accepted),
      (evt.EVT_RELEASED, self.__association_released)
    ]
    self.imageReceived = {

    }

    # logging
    if self.log_path:
      logging.basicConfig(
        level=self.log_level,
        format=self.log_format,
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[TimedRotatingFileHandler(
          filename=self.log_path,
          when='W0',
          backupCount=self.backup_weeks,
        )]
      )
    else:
      logging.basicConfig(
        level=self.log_level,
        format=self.log_format,
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[StreamHandler(
          stream=stdout
        )]
      )
    self.logger = logging.getLogger("dicomnode")
    if self.disable_pynetdicom_logger:
      logging.getLogger("pynetdicom").setLevel(logging.ERROR)

    # Validates that Pipeline is configured correctly.
    self.ae = AE(ae_title = self.ae_title)
    self.ae.supported_contexts = self.supported_contexts

    self.post_init(start=start)
    if start:
      self.open()

  def __handle_store(self, event):
    self.filter(event)

    return 0x0000

  def __association_accepted(self, event : evt.Event):
    self.logger.info(f"Association with {event.assoc.requestor.ae_title} - {event.assoc.requestor.address} Accepted")
    for requested_context in event.assoc.requestor.requested_contexts:
      if requested_context.abstract_syntax.startswith("1.2.840.10008.5.1.4.1.1"):
        self.imageReceived[event.assoc.name] = 0

  def __association_released(self, event : evt.Event):
    if event.assoc.name in self.imageReceived:
      log_message = f"Association with {event.assoc.requestor.ae_title} Released. Send {self.imageReceived[event.assoc.name]}"
    else:
      log_message = f"Association with {event.assoc.requestor.ae_title} Released"
    self.logger.info(log_message)


  def log(self):
    pass

  def filter(self, event) -> bool:
    """_summary_

    Args:
        event (_type_): _description_

    Returns:
        bool: _description_
    """
    return False

  def process(self, InputDataClass):
    raise NotImplemented

  def dispatch(self, process_return_value):
    pass

  def post_init(self, start : bool) -> None:
    """This function is called just before the server is started.
      The idea being that a user change this function to run some arbitrary code before the Dicom node starts.
      This would often be

    Args:
        start (bool): Indicatation if the server should start

    """
    pass