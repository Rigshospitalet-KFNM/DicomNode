# DicomNode

Dicom node is a toolkit for the various scripts and tools used for dicom communication

## Installation

To install this library
> `pip install git+https://github.com/Rigshospitalet-KFNM/DicomNode.git`

## Usage

The omnitool is an extendable toolkit for some common functionality.

* anonymize - Anonymizes a dicom file or directory
* show - Displaying a Dicom file
* store - Sends DIMSE C-Store to target dicom-node

To use the toolkit use:
> `omnitool $tool $tool_arguments`

## Setting up a Dicom node

This library contains modules to set up a dicom SCP optimized for a data pipeline.
Before creating your pipeline it's a good idea to read the CreatePipeline.md in the docs folder.

## Contributing

To contribute with your own feature do the following:

1. Fork this directory
2. Create a branch with:
   >`git branch $feature_name && git checkout $feature_name`
3. Commit your changes to the forked Directory
4. Create a pull request with your changes against the branch

### Adding additional tools to the omnitool

If you have a script that you wish to add to the omnitool. Then create a python file in *src/dicomnode/tools* and import it in the *\_\_init\_\_.py* file
the omnitool will call the functions
> `get_parser(parser : argparse._SubParsersAction)`
> `entry_func(args : argparse.Namespace)`

The first function will create the parser for your tool, where as the second function should be your actual script.

## Tests

To run tests you need to install the extra packages for testing with:
> `pip install git+https://github.com/Rigshospitalet-KFNM/DicomNode.git[test]`

Then run the tests with:
> `coverage run runtests.py && coverage report --show-missing`

### References & Credit

* CT test data - Albertina, B., Watson, M., Holback, C., Jarosz, R., Kirk, S., Lee, Y., … Lemmerman, J. (2016). Radiology Data from The Cancer Genome Atlas Lung Adenocarcinoma [TCGA-LUAD] collection. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2016.JGNIHEP5
* Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. (paper)
