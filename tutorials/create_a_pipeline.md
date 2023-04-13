---
title: Create a pipeline
author: Christoffer Vilstrup Jensen
---

## Introduction & Definitions

Throughout the library and tutorials the word: "pipeline" and "dicomnode" are used interchangeably.
First and foremost it's important to define what is a "pipeline". In this library it's defined as:
A dicom SCP which performs some post processing on the images and the output is shipped to some endpoint. To pass data to a pipeline is by sending dicom images to it using DIMSE messages rather than some CLI.

#### Workflow of the pipeline

In board terms a pipeline is a server program running through the following stages:

1. Wait for data
2. Filter data, if there's sufficient data move to step 3 otherwise move to step 1
3. Extract data
4. Process data
5. Export data
6. Clean up data, move to step 1.

This library heavily uses inheritance, to allow users flexibility and replace any functionality that is unwanted or insufficient.

## Building the your first pipeline

### The main class

Create a file and import the `AbstractPipeline` class from `dicomnode.server.node` module.

Superclass it and open it with the following code:

```python
from dicomnode.server.node import AbstractPipeline

class MyPipeline(AbstractPipeline):
  pass

if __name__ == '__main__':
  pipeline = MyPipeline()
  pipeline.open()
```

The `MyPipeline` class creates a SCP with no functionality. To change this you should overwrite properties and methods from the `AbstractPipeline` class.

For instance you might wish to change the AE title of the pipeline or the port it's hosted on, and can be done like so:

```python
class MyPipeline(AbstractPipeline):
  ip='0.0.0.0'
  port=4321
  ae_title="YourFancyAEtitle"
  ...
```

A full overview available configuration of the `AbstractPipeline` and other classes look at [Configuration overview](tutorials/configuration_overview.md)

### Step 2 and 3: Inputs for the pipeline

Your pipeline is going to need some inputs and for this you'll superclass another class: `AbstractInput` found in the `server.input` module.

You should consider each Abstract input as an container for a dicom series.
Each input is should filter out datasets you don't want. This is done by overwriting the `required_tags` and `required_values` attributes.

* `required_tags` - List of tags which is represented as an `int`. The tag is required to be in DICOM object but the value of the tag is irrelevant to determine if an image is valid.
* `requires_values` - Dict which is a mapping between tags and values. An image is only valid if it matches the given value.

You must implement a method to check if the input have all the images it needs.

And finally you must provide a method for transforming the dicom images into a format used by your processing step.

#### Filtering example

Say you need a CT- and a PET series as inputs to your pipeline. You would create two inputs and let them have different `required_values`:


```python
from dicomnode.server.input import AbstractInput

class MyCTInput(AbstractInput):
  required_tags: List[int] = [
    ... # List of tags used
  ]
  required_values: Dict[int, Any] = {
    0x00080060 : "CT"
  }

  ...

class MyPETInput(AbstractInput):
  required_tags: List[int] = [
    ... # List of tags used
  ]
  required_values: Dict[int, Any] = {
    0x00080060 : "PT"
  }

  ...
```

A pipeline attempt to store a picture in ALL of its inputs so an image can be stored in multiple inputs. If the Pipeline fails to store an image in at least 1 input, it will return status code: `0xB006`.

#### Validation

After each storage connection is released, each input should check if it contains sufficient data to start processing. This is done by a `validate` function call, where an input should inspect itself and determine this and return `True` if it contains sufficient data and `False` if not.

This is should be done by inspecting the `data` and `images` attributes.

Determining if you have all slices of a dicom series is non-trivial task, however as an example:

```python
class MyInput(AbstractInput):
  def validate(self) -> bool:
    max_instance_number = 0
    for dataset in self.data.values():
      max_instance_number = max(dataset.InstanceNumber, max_instance_number)
    return self.images == max_instanceNumber
```

#### Data extraction

After an input have validated, most medical image processing programs often work with a different file format to overcome the fractured nature of dicom images. So the input transforms its dicom images into some other format using a "Grinder" function.

##### Grinders

A Grinder is a glorified function, that does just that, they can be found in `dicomnode.server.grinders`.

For instance the `NumpyGrinder` outputs a numpy array of dicom images.

```python
from dicomnode.server.grinders import Grinder, NumpyGrinder

class MyInput(AbstractInput):
  image_grinder: Grinder = NumpyGrinder()
```

Note that some grinder might have additional installation requirement and can be found in those directories instead.

#### Importing Inputs into the pipeline

Once you have configured your input classes, you need to configure your pipeline to create inputs you have created by overwriting the `input` attribute, with a dict containing the classes of input and an input.

**Note you must pass the type of the class, not an instance of the class!**

Now your pipeline needs a method to separate two unrelated dicom series, while grouping two related dicom series together. By default this distinction is made using the Patient ID attribute of the dicom series, but it can configured to another attribute by overwriting the `patient_identifier_attribute` of the `AbstractPipeline`

Considering our Pet and CT example the code would look like:

```python
class MyCTInput(AbstractInput):
  ...

class MyPETInput(AbstractInput):
  ...

class MyPipeline(AbstractPipeline):
  ...
  input = {
    'CT' : MyCTInput,
    'PET' : MyPetInput,
  }
  ...
```

### Processing

The processing is handled by the process method, so you need to overwrite it with your own post-processing, however it must have a specific call structure.

It must accept an `InputContainer` and return a `PipelineOutput`.


```python
class MyPipeline(AbstractPipeline):
  ...
  def process(self, input_container: InputContainer) -> PipelineOutput:
    ...
```

The `PipelineOutput` is related to exporting data and will be explained in the next section.

The `InputContainer` is a glorified `Dict[str, Any]` where the keys are matching the keys of the `input` attribute of the pipeline and values is what the `AbstractInputs` Grinders returned.

So in the PET and CT example from above: `input_container['CT']` would return the CT image and `input_container['PET']` would return the pet image.

An `InputContainer` also may contain:

* `header` - A `Dicomnode.lib.dicom_factory.SeriesHeader` - which is an object useful for recreating dicom series and next subsection.
* `response_address` - A `Dicomnode.lib.dimse.Address` - which represent the last association to send picture to this patient.
Take the running example, if our PET and CT picture originate from two different sources, the response address is last to add studies to the Patient.

#### Building new Dicom Series

After applying some post processing to an image, you need to return to the Dicom format. For this you need a header and a specialized factory.

Both base classes can be found in `dicomnode.lib.dicom_factory`.

* `Blueprint` - A static blueprint for a dicom series
* `DicomFactory` - Factory that build dicom series.

Building a new dicom series happens in three steps.

1. A `Blueprint` and Specialized `DicomFactory` is given to an AbstractPipeline
2. A Father dicom Series and a `Blueprint` produce a `SeriesHeader`
3. A `SeriesHeader` and an image produces the new Dicom Series.

You'll have to perform step 3 in the processing function.

For a more in-depth tutorial, see [Create a dicom Series](tutorials/create_a_dicom_series.md)


### Exporting Data

The final step of a pipeline is to send data to an endpoint. This is done by
`PipelineOutput`-objects, but you have to create them in the processing.

Different type of `PipelineOutput` sends 

```python

from dicomnode.lib.dimse import Address

from dicomnode.server.output import DicomOutput

class MyPipeline(AbstractPipeline):
  ...
  def process(self, input_container: InputContainer) -> PipelineOutput

  ...

  return DicomOutput([Address(ip='', port=104, ae_title=""), datasets])
```

If you have performed the steps above you now have a functional pipeline.

