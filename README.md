# DicomNode

DicomNode is a framework for building post-processing pipeline of medical
images. The purpose of the framework is to provide an easy and simple way of
transforming a research project into a functional dicom node that can be used in
clinic.

In addition to provide the infrastructure of a server, it provides tools for
common tasks that all pipelines needs to handle, such as:
dicom -> ndarray -> dicom



## Installation

The Library require:

* Python developer tools

Using apt:
> `sudo apt install g++ cmake python3-dev`

To install this library
> `pip install git+https://github.com/Rigshospitalet-KFNM/DicomNode.git`

### Reports

The generated reports use Latex to compile pdf reports. This is done using
defaults for pylatex which rely on installed latex compilers.

To install compilers

Using apt:
> `sudo apt install texlive`

If you are using custom fonts:

> `sudo apt install texlive-xetex`

## Toolkit Usage

The omnitool is an extendable toolkit for some common functionality.

* anonymize - Anonymizes a dicom file or directory
* show - Displaying a Dicom file
* store - Sends DIMSE C-Store to target dicom-node

To use the toolkit use:
> `omnitool $tool $tool_arguments`

## Docs

Docs are deployed at: [Read the docs](https://dicomnode.readthedocs.io/en/latest/)

## Tests

To run tests you need to install the extra packages for testing with:
> `pip install git+https://github.com/Rigshospitalet-KFNM/DicomNode.git[test]`

Then run the tests with:
> `coverage run runtests.py && coverage report --show-missing`

## Windows support

[For windows support](https://ubuntu.com/tutorials/install-ubuntu-desktop)

## Citations

<a id="1">[1]</a>
D. P. Playne and K. Hawick
"A New Algorithm for Parallel Connected-Component Labelling on GPUs"
in IEEE Transactions on Parallel and Distributed Systems, vol. 29, no. 6, pp. 1217-1230,
1 June 2018,
doi: 10.1109/TPDS.2018.2799216.
