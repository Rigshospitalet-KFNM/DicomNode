# DicomNode

Dicom node is a toolkit and library for the various scripts and tools used for dicom communication
It contains the tools to set up an post processing pipeline of medical images.

## Installation

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

I recommend looking in the [createPipeline.md](tutorials/CreatePipeline.md) for first look

Then look at an example in the examples folder [example](examples/plusOneNode.py)

To gain a good overview of the different classes, look at [ClassesOverview.md](tutorials/ClassOverview.md)

## Tests

To run tests you need to install the extra packages for testing with:
> `pip install git+https://github.com/Rigshospitalet-KFNM/DicomNode.git[test]`

Then run the tests with:
> `coverage run runtests.py && coverage report --show-missing`
