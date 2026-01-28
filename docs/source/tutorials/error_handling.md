# Error Handling

Part of building a robust program is handling unexpected errors. The main
problem with dealing with an unexpected error is that it is unexpected, and
therefore per definition you cannot be prepared to handle such an error. Unless
you are aware of potential error source, but you assume it so unlikely that you
do not program around this. Think of a hard drive failure where you can no
longer read or write to the drive, or there is no longer any memory on the
device.

It could also be a programming error, where some package throws an exception for
a specific set of dicom images, or there was some precondition that you assumed,
but didn't check. This could happen when you switch scanner or use custom
datasets that you didn't develop for.

Your pipeline may also have custom errors. For instance if your model can only
handle patient laying on their back, but a patient is unable to lay on their
back due a crooked spine. What should your pipeline do when inevitable happens.

The first step is figuring out the custom error, that you have and check them.
This is because Dicomnode doesn't know about your special cases. The way you can
check if the patient is laying on the side is that ImageOrientation tag will be
different from the standard `[1,0,0,0,1,0]` and then you can throw an exception
with `raise Exception` you can even create your own fancy exception to help the
debugging process afterwards:
```python
class FancyException(Exception):
  pass

def custom_error(input_container) -> bool:
  ...

class MyPipeline(AbstractPipeline):
  class Processor(AbstractProcessor)
    def process(self, input_container):
      if custom_error(input_container):
        raise FancyException

      # Rest of the pipeline
```
