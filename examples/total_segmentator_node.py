""""""

# Python standard library



# Third party packages
from pydicom.uid import CTImageStorage

# Dicomnode packages
from dicomnode.server.grinders import NiftiGrinder
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.output import DicomOutput, PipelineOutput
from dicomnode.server.pipeline_tree import InputContainer

class CTInput(AbstractInput):
  required_values = {
    0x0008_0016 : CTImageStorage
  }  

  def validate(self) -> bool:
    valid = True
    for image in self:
      valid &= image.InstanceNumber < self.images
    return valid

  image_grinder = NiftiGrinder()

class TotalSegmentatorNode(AbstractQueuedPipeline):
  port = 11113
  ae_title = "TOTALSEGMENTATOR"

  def process(self, input_data: InputContainer) -> PipelineOutput:
    
    
    
    return 

