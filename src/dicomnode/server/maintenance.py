"""Contains the module for the maintience thread used by the dicomnode

  MaintienceThread is a thread, that removes old studies and perform other cleanup.
"""

__author__ = "Christoffer"

# Python3 standard Library
from datetime import datetime, timedelta
from threading import Thread, Event
from typing import Any, Callable, Iterable, Mapping

# Thrid party Packages

# Dicomnode packages
from dicomnode.server.pipeline_tree import PipelineTree

class MaintenanceThread(Thread):
  """This thread ensures that old studies are removed from the input
  pipeline tree.

  Should be stopped upon server closure.
  """
  _seconds_in_a_day = 86400

  def __init__(self,
               pipeline_tree: PipelineTree,
               study_expiration_days: int,
               group: None = None,
               name: str | None = None,
               args: Iterable[Any] = ...,
               kwargs: Mapping[str, Any] | None = None,
               *,
               daemon: bool | None = None) -> None:
    super().__init__(group, None, name, args, kwargs, daemon=daemon)
    self.pipeline_tree = pipeline_tree
    self.study_expiration_days = study_expiration_days
    self.__running = True
    self.waiting_event = None


  def run(self):
    while self.__running:
      self.waiting_event = Event()
      waiting = self.waiting_event.wait(
        self.calculate_seconds_to_next_maintenance())
      if waiting:
        break
      else:
        self.maintenance()


  def stop(self):
    """Wakes the thread and kills it"""
    self.__running = False
    if self.waiting_event is not None:
      self.waiting_event.set()


  def calculate_seconds_to_next_maintenance(self, now=datetime.now()) -> float:
    """Calculates the time in seconds to the next scheduled clean up"""
    tomorrow = now + timedelta(days=1)
    clean_up_datetime = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0,0,0,0, tzinfo=now.tzinfo)
    time_delta = clean_up_datetime - now
    return time_delta.days * self._seconds_in_a_day + float(time_delta.seconds) # I guess you could add micro seconds here but WHO CARES


  def maintenance(self, now = datetime.now()) -> None:
    """Removes old studies in the pipeline tree to ensure GDPR compliance
    """
    # Note this might cause some bug, where a patient is being processed, and at the same time removed
    # This is considered so unlikely, that it's a bug I accept in the code
    expiry_datetime = now - timedelta(days=self.study_expiration_days)
    self.pipeline_tree.remove_expired_studies(expiry_datetime)
