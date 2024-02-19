"""A Cluster is an object where you have multiple dicom nodes running in the
same endpoint. This is useful if you have multiple "quick to execute" nodes.
Or if you have multiple nodes which must share a resource (Think GPU)
"""

# Python Standard Library
from typing import List, Optional, Type

# Third party packages

# Dicomnode packages
from dicomnode.lib.logging import get_logger
from dicomnode.lib.exceptions import IncorrectlyConfigured
from dicomnode.server.nodes import AbstractPipeline, AbstractQueuedPipeline

class Cluster:
  nodes: Optional[List[AbstractPipeline, Type[AbstractPipeline]]] = None
  _nodes: List[AbstractPipeline]


  read_queued_warning = False

  def __init_nodes(self, nodes: List[AbstractPipeline, AbstractQueuedPipeline]):
    self._nodes = []
    multiple_queues = False
    for node in nodes:
      if isinstance(node, Type[AbstractPipeline]):
        node: AbstractPipeline = node()
      if isinstance(node, AbstractQueuedPipeline):
        if multiple_queues:
          if not self.read_queued_warning:
            self.logger.warning()
        else:
          multiple_queues = True
      self._nodes.append(node)

  def __init__(self, nodes: Optional[List[AbstractPipeline, Type[AbstractPipeline]]] = None) -> None:
    self.logger = get_logger()
    if nodes is not None:
      self.__init_nodes(nodes)
    elif self.nodes is not None:
      self.__init_nodes(nodes)
    else:
      raise IncorrectlyConfigured
