"""A Cluster is an object where you have multiple dicom nodes running in the
same endpoint. This is useful if you have multiple "quick to execute" nodes.
Or if you have multiple nodes which must share a resource (Think GPU)
"""

# Python Standard Library
from typing import Dict, List, Optional, Type, TypeAlias, Union

# Third party packages

# Dicomnode packages
from dicomnode.lib.logging import get_logger
from dicomnode.lib.exceptions import IncorrectlyConfigured
from dicomnode.server.nodes import AbstractPipeline, AbstractQueuedPipeline

cluster_node_list: TypeAlias = List[Union[AbstractPipeline,
                                          Type[AbstractPipeline]]]


class Cluster:
  nodes: Optional[cluster_node_list] = None
  ae_titles: List[str]
  _nodes: Dict[str,AbstractPipeline]

  read_queued_warning = False

  def __init_nodes(self, nodes: cluster_node_list):
    self._nodes = {}

    multiple_queues = False
    for node in nodes:
      if isinstance(node, Type):
        node = node()
      if isinstance(node, AbstractQueuedPipeline):
        if multiple_queues:
          if not self.read_queued_warning:
            self.logger.warning("")
        else:
          multiple_queues = True
      self._nodes[node.ae_title] = node

  def __init__(self, nodes: Optional[cluster_node_list] = None) -> None:
    self.logger = get_logger()
    if nodes is not None:
      self.__init_nodes(nodes)
    elif self.nodes is not None:
      self.__init_nodes(self.nodes)
    else:
      raise IncorrectlyConfigured

__all__ = [
  'Cluster'
]
