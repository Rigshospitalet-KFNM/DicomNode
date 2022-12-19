from threading import Thread
from typing import Dict, List, Optional

from pynetdicom.events import Event


from dicomnode.server.nodes import AbstractPipeline

class AbstractThreadedPipeline(AbstractPipeline):
  threads: Dict[Optional[int],List[Thread]] = {}

  def _handle_store(self, event: Event) -> int:
    thread: Thread = Thread(target=super()._handle_store, args=[event], daemon=True)
    thread.start()
    if event.assoc.native_id in self.threads:
      self.threads[event.assoc.native_id].append(thread)
    else:
      self.threads[event.assoc.native_id] = [thread]
    return 0x0000

  def _join_threads(self, assoc_name:Optional[int] = None) -> None:
    if assoc_name is None:
      for thread_list in self.threads.values():
        for thread in thread_list: # pragma: no cover
          thread.join() # pragma: no cover
      self.threads = {}
    else:
      thread_list = self.threads[assoc_name]
      for thread in thread_list:
        thread.join()
      del self.threads[assoc_name]

  def _association_released(self, event: Event):
    self._join_threads(event.assoc.native_id)
    return super()._association_released(event)

