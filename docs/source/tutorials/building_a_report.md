# Building a report

## Motivation

Dicomnode uses pyLaTeX and therefore LaTex as the underlying engine for its
report generation. This grants the freedoms that the user can make any report
that they desire, while at the same time open the window for standardization
with build in components.

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
    encoded_report = self.dicom_factory.encode_pdf(report)

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
variables.

* working_directory
* report_directory
* report_data_directory
* figure_directory

### What Dicomnode does for you

So you could just use PyLaTeX yourself or some other package to produce the pdf
document, however there's value in standardization. So I recommend using the
library components over rolling your own.

The second thing is that DIMSE cannot handle raw pdf or tex files, they need to
be encoded into a dicom object before it can be send. This is what the
`encode_pdf` method does for you. Note that you might need to update some
information



## Dicomnode Report generation

So this section is a exploration build in components and the way I would
recommend building a report.

### Report

So to start with we need the root object that we use to contain all the other
objects.

```Python

from dicomnode.report import Report

# Rest of the pipeline

def generate_report(images, input_data):
  report = Report(f'{}')
```



