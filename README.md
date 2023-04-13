# DicomNode

DicomNode is a framework for building post-processing pipeline of medical images.
It specializes in converted dicom images into formats usable formats (e.g. Nifti, Minc) and back to dicom.

## Installation

The Library require:

* g++
* cmake
* pybind11
* Python developer tools

To install this library
> `pip install git+https://github.com/Rigshospitalet-KFNM/DicomNode.git`

## Toolkit Usage

The omnitool is an extendable toolkit for some common functionality.

* anonymize - Anonymizes a dicom file or directory
* show - Displaying a Dicom file
* store - Sends DIMSE C-Store to target dicom-node

To use the toolkit use:
> `omnitool $tool $tool_arguments`

## Setting up a Image pipeline

This library contains modules to set up a dicom SCP optimized for a data pipeline.
A number of tutorials can be found in the tutorial folder.

I recommend reading the [createPipeline.md](tutorials/CreatePipeline.md) document first.
After this look through a couple of examples for instance: [plusOneNode](examples/plusOneNode.py)
The library contains a bunch of classes for common problems, consider looking through the [ClassesOverview.md](tutorials/ClassOverview.md) to determine if there's something that could help you.
The classes can be configured in different ways. Look through [ConfigurationOverview.md](tutorials/ConfigurationOverview.md) to see the flexibility of the classes.

## Tests

To run tests you need to install the extra packages for testing with:
> `pip install git+https://github.com/Rigshospitalet-KFNM/DicomNode.git[test]`

Then run the tests with:
> `coverage run runtests.py && coverage report --show-missing`
