"""Contains the module for the maintience thread used by the dicomnode

  MaintienceThread is a thread, that removes old studies and perform other cleanup.
"""

__author__ = "Christoffer"

# Python3 standard Library
from datetime import datetime, timedelta
from threading import Thread, Event
from typing import Any, Iterable, Mapping, Optional

# Thrid party Packages
import psutil

# Dicomnode packages
from dicomnode.lib.utils import human_readable_byte_count
from dicomnode.lib.logging import get_logger
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
               name: Optional[str] = "Maintenance",
               args: Iterable[Any] = [],
               kwargs: Optional[Mapping[str, Any]] = None,
               *,
               daemon: Optional[bool]= None) -> None:
    super().__init__(group, None, name, args, kwargs, daemon=daemon)
    self.pipeline_tree = pipeline_tree
    self.study_expiration_days = study_expiration_days
    self._running = True
    self.waiting_event = None
    self.logger = get_logger()


  def run(self): # pragma: no cover
    while self._running:
      self.waiting_event = Event()
      waiting = self.waiting_event.wait(
        self.calculate_seconds_to_next_maintenance())
      if waiting:
        break

      self.maintenance()


  def stop(self):
    """Wakes the thread and kills it"""
    self._running = False
    if self.waiting_event is not None:
      self.waiting_event.set()


  def calculate_seconds_to_next_maintenance(self, input_now:Optional[datetime] = None) -> float:
    """Calculates the time in seconds to the next scheduled clean up"""
    if input_now is None:
      now = datetime.now()
    else:
      now = input_now

    if(now.hour == 23 and now.minute == 59):
      return self._seconds_in_a_day

    tomorrow = now + timedelta(days=1)
    clean_up_datetime = datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                                 0,0,0,0, tzinfo=now.tzinfo)
    time_delta = clean_up_datetime - now
    # I guess you could add micro seconds here but WHO CARES
    return time_delta.days * self._seconds_in_a_day + float(time_delta.seconds)


  def maintenance(self, input_now: Optional[datetime] = None) -> None:
    """Removes old studies in the pipeline tree to ensure GDPR compliance
    """
    if input_now is None:
      now = datetime.now()
    else:
      now = input_now # pragma: no cover


    # Note this might cause some bug,
    # where a patient is being processed, and at the same time removed
    # This is considered so unlikely, that it's a bug I accept in the code
    expiry_datetime = now - timedelta(days=self.study_expiration_days)
    self.pipeline_tree.remove_expired_studies(expiry_datetime)
    self.logger.info("Performed Maintenance, current pipeline tree is:")
    self.logger.info(str(self.pipeline_tree))

    process = psutil.Process()
    with process.oneshot():
      mem_info = process.memory_info()
      self.logger.info(f"Process is using {human_readable_byte_count(mem_info.rss)} Memory")

__all__ = [
  'MaintenanceThread'
]
