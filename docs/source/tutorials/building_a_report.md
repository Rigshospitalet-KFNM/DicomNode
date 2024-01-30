# Building a report

## Disclaimer

Most of these code example will not work out of the box and might require you
to extract various values.

## Motivation

Dicomnode uses pyLaTeX and therefore LaTex as the underlying engine for its
report generation. This grants the freedoms that the user can make any report
that they desire, while at the same time open the window for standardization
with build in components.

Note that PyLaTeX and this library have very philosophies about each of its
classes. PyLaTeX maps each of the it's classes to LaTeX commands/environments
while dicomnode uses it classes to be blueprints.

## Starting point

Lets assume that we have a pipeline, that should produce some images and a
report with key values, we can imagine the code to look something like this:

```python
class MyPipeline(AbstractPipeline):
  def process(self, input_data):
    # data extration from input data
    ...

    # Mathematical modeling
    modeled_images = ...

    # Report Generation
    report = generate_report(modeled_images, input_data)

    # Factory
    modeled_datasets = self.dicom_factory.build_from_header(input_data.header, blueprint)
    encoded_report = self.dicom_factory.encode_pdf(report, modeled_datasets)

    return DicomOutput([(PACS_ADDRESS, modeled_dataset), (PACS_ADDRESS, encoded_report)])
```

The generated report is a placeholder. (Although I highly recommend that you try
and keep the process function at a high level like the example.) In this
tutorial we explore the content of that function.

## The PyLaTeX pipeline

As dicomnode is build on top of PyLaTeX it is necessary to understand how that
library works at surface level.

The idea is that the library provides some objects that can be combined, which
then produces a .tex file, which then can be compiled by your standard LaTeX
compiler. This is why you have to install them to make this work as they are
not python modules.

Such as: `sudo apt install texlive` Naturally this might not sufficient since
the LaTeX compiler might need extra packages to compile the extra document.
For instance you need to use `XeLatex` or `LuaLatex` if you wish to compile
with a custom font. The library defaults to `XeLatex` which can be installed by
`sudo apt install texlive-xetex`.

### Files

One 'disadvantage' to this is that we are interacting with the file system and
other program. Which means that an error might originate in an other program
and it can be very hard to recover from such an error. It also means that
other threads or programs can mess with the files that your program depend on,
which leads to some bugs witch is dependant on the environment.

The next thing is that files needs a directory to be in. Dicomnode have a
couple of paths that is needed. These paths can be controlled with Environment
variables and stored in the `dicomnode.library_paths`.

* (Python Attribute) - (Environment variable) - (default)
* processing_directory - DICOMNODE_ENV_PROCESSING_PATH - /tmp/dicomnode/
* report_directory - DICOMNODE_ENV_REPORT_PATH - /tmp/dicomnode/report
* report_data_directory - DICOMNODE_ENV_REPORT_DATA_PATH - /tmp/dicomnode/report_data
* figure_directory - DICOMNODE_ENV_FIGURE_PATH - /tmp/dicomnode/figures

The processing directory is the directory that a dicomnode uses a root for
relative paths. I recommend this directory is clean as this can make it
explicit which files are needed, and therefore make it easier to deploy the
node in a different environment.

The Report directory is the directory that reports could be compiled to.
Note that LaTeX produces several auxiliary files for compilation. PyLaTex
cleans these files up after a successful compilation, but leaves them if a
compilation have failed.

The Report data directory is a directory you can place files needed for report
compilation. Examples are .sty files or static images.

The figure directory is a directory intended to put your matplotlib figures in.

### What Dicomnode does for you

Dicomnode provides: A Report base class, some build in plots and components
that you can use or can inspire you to write your own components.

Dicom transfer mechanism (DIMSE) cannot handle raw pdf or tex files, they need
to be encoded into a dicom object before it can be send. This is what the
`encode_pdf` method does for you.

## Dicomnode Report generation

So this section is a exploration build in components and the way I would
recommend building a report.

### Report

So to start with we need the root object that we use to contain all the other
objects. For this we use the document, we can add content to it by using the
append method or the create method.

```Python
from pylatex import Section, MiniPage

from dicomnode.report import Report

# Rest of the pipeline

def generate_report(images, input_data):
  # Notice lack of file extension as the libraries handle thi
  report = Report(f'{PatientID}')

  report.append(Section('This is the first section')) # The first method

  with report.create(MiniPage()) as mini_page:
    ... # Fill the mini page with content

  ... # Rest of Report generation

  return report
```

### Patient Information

This component displays relevant patient information

### Report Header

### Table


