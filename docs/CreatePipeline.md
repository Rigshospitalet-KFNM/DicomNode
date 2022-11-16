---
title: Create a pipeline
author: Christoffer Vilstrup Jensen
---
# Creating a pipeline

## Introduction & Definitions

First and foremost it's important to define what is a "pipeline".
For this library it's defined as: A program which have some number of inputs, and
when all inputs are present, they are consumed and some output is produced and store externally.

This library is optimized towards hiding all the communication, file storage and other things that isn't the processing of the input pictures.
Most of the base tools are found in the server submodule, however the lib module is might also be useful to have a look through.

If you're unfamiliar with the DICOM protocol, please read the **DicomForDummies.md** document before proceeding.
This tutorial also assumes you're familiar with some python software engineering conventions, if not please read **PythonSoftwareEngineering.md**

## Building the your first pipeline

### The main class

The pipeline is abstracted away into a single class which you should inherit from. This class is found in the `server.node` submodule and is called `AbstractPipeline`.

Create a file and fill it with the following code:

```python
from dicomnode.server.node import AbstractPipeline

class MyPipeline(AbstractPipeline):
  pass

if __name__ == '__main__':
  MyPipeline()
```

So the `MyPipeline` class is where you'll configure the server by overwriting some properties and methods from the `AbstractPipeline` class.
The last line is creating and instance of your pipeline. Which will try and open on localhost:104 with a default AE title.
Note that you might not be allowed to open ports below 1024, without super user permissions.
This is probably not what you want so you need to configure the pipe line add the following lines to your pipeline class:

```python
class MyPipeline(AbstractPipeline):
  ip='0.0.0.0'
  port=4321
  ae_title="YourFancyAEtitle"
  ...
```

### Inputs for the pipeline

Your pipeline is going to need some inputs and for this you'll need another class: `AbstractInput` found in the `server.input` module.
Similar to the pipeline you need to create an input class.

Add this code before your `MyPipeline` class definition

```python
from dicomnode.server.input import AbstractInput

class MyInput(AbstractInput):
  required_tags: List[int] = [0x00080018, 0x7FE00010] # SOPInstanceUID, Pixel Data
  required_values: Dict[int, Any] = {}
  image_grinder: Callable[[Iterator[Dataset]], GrindType] = staticmethod(grinder_function)

  def validate(self) -> bool:
    ...
```

Before inserting the input into the pipeline, you should configure the input.
As the pipeline receives C-stores it will attempt to store it in these AbstractInput classes.
A pipeline attempt to store a picture in ALL of its inputs and an input can be stored in multiple inputs.
To determine if a input accepts a picture it looks at its class attributes: `required_tags` and `required_values`.

* `required_tags` - List of tags which is represented as an `int`. The tag is required to be in DICOM object but the value of the tag is irrelevant to determine if an image is valid.
* `requires_values` - Dict which is a mapping between tags and values. An image is only valid if it matches the given value.

The way you should use this is by putting the tags that you use in your processing in the `required_tags` attribute and the tags and values in the `required_values` which makes sure it's the correct type of image. While the entire image gets stored, your process function should not depend tags
that you have not checked exists.

After each connection is released the pipeline checks each patient which received data under that connection, if the there's sufficient data to start processing.
The pipeline determines this by iterating over all the inputs for the patient, by calling the `validate` function.

This function looks at the inputs `data` and `images` attributes and determines if input is ready for processing. This is a user defined function, meaning it's your job to write a function, which determines this. It should return `True` if the input have the required data for processing and `False` if it don't.

If all inputs for that patient validates, each input's `get_data` method is called. By default this is a call to the image_grinder function.
A grinder is a function which pre-processes the data, extracting the relevant data and then passing it on the the process function in the main pipeline.
You can look at the `lib.grinders` to see all available grinders, or write you own if libraries grinders doesn't work for your project.

### Putting it all together

Now that you have defined your input it's time put it into the pipeline:

```python
class MyPipeline(AbstractPipeline):
  ...
  input = {'argument_name' : MyInput }
  ...
```

If you need multiple inputs, just add additional entries into the input directory. Note that it's the class itself you're passing not an instance of the class.

Add the process method to the class, this is the image processing function that is the reason why you wanted the pipeline in the first place.

```python
class MyPipeline(AbstractPipeline):
  ...
  def process(self, input_data: Dict[str, Any]) -> Iterable[Dataset]:
    ...
```

The `input_data` dictionary contains keys matching the keys of the `input` and values what the inputs `get_data` method returned, which unless you overwrote it is the just the return value of the grinder. Your function should return a iterable (list) of datasets of identical modality.

### Exporting Data

The final step of a pipeline is to send data to an endpoint. To define an endpoint you need to overwrite another attribute called endpoint.
This should be a list of addresses, which is a data class found in `lib.dimse` module:

```python

from dicomnode.lib.dimse import Address

class MyPipeline(AbstractPipeline):
  ...
  endpoints = [Address(ip='', port=104, ae_title="")]
```

If you have performed the steps above you now have a functional pipeline.

### Configuring the pipeline

So far we have only scratched the surface of configuration. To get the full overview, I suggest reading the `nodes.py` file, but some common configuration can be found below:

#### Permanent file storage

By default all the dataset is kept in memory. That means if the program is stopped, all data is lost.
To make the pipeline save a copy to disk you need to overwrite an attribute:

```python
class MyPipeline(AbstractPipeline):
  ...
  root_data_directory: str | Path = Path('path/to/data/directory')
  ...
```

The pipeline will now save a copy of each dataset, and delete them when it's done with the dataset.
The file structure produced looks like this:

```text
  root_data_directory / {\$patient_identifier_tag} / {\$input_arg_name_1} / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                                                ... / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                   / {\$input_arg_name_2} / Image_{\$image.modality}_{\$instance_number}.dcm
                                                                                ... / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                               ... / ...
                      / {\$patient_identifier_tag} / {\$input_arg_name_1} / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                                                ... / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                   / {\$input_arg_name_2} / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                                                                ... / Image_{\$image.modality}_{\$image.instance_number}.dcm
                                               ... / ...
                  ... / ...
```

The `patient_identifier_tag` is another pipeline attribute, which the pipeline uses to separate images belonging to differing batches.
The tag defaults to the tag PatientID. The `input_arg_name` is a key in the input directory, the files are the images stored in the input instance.

#### Logging

If you have tried and run your pipeline in a terminal, you'll without a doubt noticed some messages. These are logging messages, and by default they are passed to print on the screen, and doesn't really create a permanent record. To get a "permanent log" you need to set a path, where the log can be stored:

```python
class MyPipeline(AbstractPipeline):
  ...
  root_data_directory: str | Path = Path('path/to/log')
  ...
```

The pipeline will now create a rotating timed log, that means every week at monday midnight it's going to move the file into a backup file. The pipeline keeps up to 8 backup logs, or 2 months worth of logs. This is done to prevent, that the logs grow massively and fill the server the pipeline is running on.
