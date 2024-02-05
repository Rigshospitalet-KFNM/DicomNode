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
    modeled_datasets = self.dicom_factory.build_from_header(input_data.header,
                                                            blueprint)
    encoded_report = self.dicom_factory.encode_pdf(report, modeled_datasets)

    return DicomOutput([(PACS_ADDRESS, modeled_dataset), (PACS_ADDRESS,
                                                          encoded_report)])
```

The generated report is a placeholder. (Although I highly recommend that you try
and keep the process function at a high level like the example.) In this
tutorial we explore the content of that function.

### A general outline

With this top level view you should consider the report generation in the
following steps:

1. Data generation (Most of the normal pipeline)
2. Data Extraction: Not all data might be needed. Isolate the data you need to
generate all reports possible, and pass it to a new environment (read a
function call)
3. Create the report section wise, ie: Create the header, create information
about the patient, print some picture... etc. etc.
4. (C) If you have some conditional content, ie: if you should display
different information based on the extracted data, then extract out to its own
function and be clear try to specify the different types of reports generated
using an enum. This will help other understand your code better.
5. Return the report, encode it to dicom using the dicom factory.

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
* `processing_directory` - **DICOMNODE_ENV_PROCESSING_PATH** - */tmp/dicomnode/*
* `report_directory` - **DICOMNODE_ENV_REPORT_PATH** - */tmp/dicomnode/report*
* `report_data_directory` - **DICOMNODE_ENV_REPORT_DATA_PATH** - */tmp/dicomnode/report_data*
* `figure_directory` - **DICOMNODE_ENV_FIGURE_PATH** - */tmp/dicomnode/figures*

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

### Build-in components

#### Report

So to start with we need the root object that we use to contain all the other
objects. For this we use the document, we can add content to it by using the
append method or the create method.

```Python
from pylatex import Section, MiniPage

from dicomnode.report import Report

# Rest of the pipeline

def generate_report(images, input_data):
  # Notice lack of file extension as the libraries handle this
  report = Report(f'{PatientID}')

  report.append(Section('This is the first section')) # The first method

  with report.create(MiniPage()) as mini_page:
    ... # Fill the mini page with content

  ... # Rest of Report generation

  return report
```

#### Patient Information

This component displays relevant patient information i a framed box:

* Patient Name
* Patient ID
* Study Name
* Series Name
* Study Date

#### Report Header

This add a header to the study with information about the hospital and the
performing department. It includes a icon which I recommend you place as a
static image in the `report_data` directory

#### Table

This is a table with a few build in styles. I can also recommend PyLatex's
Tabular and Tabularx are recommended alternatives.

#### Plot & Plots

Dicomnode also includes some standardization for plots, which is build on top
of `matplotlib`. The relevant base class is `dicomnode.report.plot.Plot` which
is responsible for saving your image and appending it the report.

Now the library assumes that you wish to plot from a three dimensional volume
which poses natural problems unless you have some really cool paper. There's
three ways you can traverse an volume:

* Corornal - From front to back
* Sagittal - From side to side
* Transverse - From top to bottom

which is encapsulated in the `Plot.AnatomicalPlane` enum, which you can use to
create a `Plot.PlaneImages` sequence which takes a plane as argument and allows
you to transverse through the volume. Because you are limited to 2 dimensions,
you need to select an image from the volume, which is done by:
`Dicomnode.report.base_classes.Selector`. A selector is just a glorified
function which select an image (or range of images). Finally you might wish to
apply some transformation function.

##### Anatomical Plot

This is the "base plot" of dicomnode, it and other plots of Dicomnode can be
configured by passing a `<PlotType>.Options` in its constructor. It displays a
single slice

##### Triple Plot

This is just 3 Anatomical plots next to each other.

### Rolling your own

Naturally these components might not be exactly what you want. I highly suggest
that you create "Blueprints" similar to `ReportHeader`, `PatientInformation`

I suggest that you implement them as subclasses of
`dicomnode.report.base_classes.LaTeXComponent`. This required you to implement
2 methods:

1. `append_to` - This method add the content the blueprint to the report
2. `from dicom` - This construct an instance of your blueprint from a dicom
picture or series.

Note that the main idea is that the blueprint understand what should be added
to the report, while the report is ignorant of the implementation of the
blueprint.

#### Create new PyLaTeX primitives.

PyLaTeX as library also provide some out the box components such as the
minipage Component. Lets grab an example:
```python
mini_page = MiniPage(width=r"0.8\textwidth")
mini_page.append("Bla bla bla")
```

Becomes the equivalent latex code:

```latex
\begin{minipage}{0.8\textwidth}%
bla bla bla%
\end{minipage}%
```

Lets work backwards how this code was generated:

1. the `\begin` and `\end` is generated because mini page inherits from
`pylatex.base_classes.Environment` which specifies that should included.
2. The mini_page has the class name "MiniPage". When you apply the lower
function, you get the posted `minipage`. This is because the MiniPage class
doesn't have the defined the constant `_latex_name`. If that is undefined it
will take and use the name of the class.
3. Then because the mini page contains content, the content get recursively
added.

Now some components requires packages for instance the `framed` component a
package. If you create a `pylatex.Package` for framed so the python code:

```python
frame = Framed()
Framed.append("content")
```

Becomes the LaTeX:

```latex
\usepackage{framed}

... % Rest of the Document before framed

\begin{framed}
content
\end{framed}

... % Rest of the Document after framed
```

Sometimes you might want more header commands for instance if you have custom
environments or commands. Sadly you are going to need a tad more foot work to
make this work.

For this example lets look at
`dicomnode.src.report.latex_components.DicomFrame` which is such a case. It is
a box which wraps content, however different than framed and mdframe is that
it only wraps content length not the entire line.

This is achieved with the following latex code:

```latex
% Header code
% Packages
\usepackage{mdframed}
\usepackage{xcolor}
\usepackage{color}
\usepackage{varwidth}
\usepackage{environ}
\usepackage{calc}

% Custom
\definecolor{navy}{HTML}{0000AA}
\newlength{\frameTweak}%
\mdfdefinestyle{FrameStyle}{%
    linecolor=navy,
    outerlinewidth=5pt,
    innertopmargin=5pt,
    innerbottommargin=5pt,
    innerrightmargin=5pt,
    innerleftmargin=5pt,
    leftmargin = 5pt,
    rightmargin = 5pt
}

\NewEnviron{prettyFrame}[1][]{%
        \setlength{\frameTweak}{\dimexpr%
        +\mdflength{innerleftmargin}%
        +\mdflength{innerrightmargin}%
        +\mdflength{leftmargin}%
        +\mdflength{rightmargin}%
        }%
    \savebox0{%
        \begin{varwidth}{\dimexpr\linewidth-\frameTweak\relax}%
            \BODY%
        \end{varwidth}%
    }%
    \begin{mdframed}[style=FrameStyle,backgroundcolor=white,userdefinedwidth=\dimexpr\wd0+\frameTweak\relax, #1]%
        \usebox0%
    \end{mdframed}%
}%

% Rest of the document header and so forth

\begin{dicomframe}
  ...
\end{dicomframe}

% Rest of the document
```

The way I solved this is by creating a nested class wrapped by the class:

```Python
class DicomFrame(LaTeXComponent):
  class DicomFrame(Environment):
    packages = []

  def __init__(self):
    super().__init__()
    self._inner = self.DicomFrame

  def append_to(self, document: Report):
    if not self.__class__.__name__ in document.loaded_preambles:
      # Load The preamble
      self.__load_preamble(document.preamble)
      document.loaded_preambles.add(self.__class__.__name__)
    document.append(self._inner)

```

The inner class is a `PyLaTeXObject` from `PyLaTeX` and is what gets appended
to the document. The document class contains a set `loaded_preambles` which
contains the name of all classes which have the loaded preambles. This prevents
the preamble from being loaded twice, and prevent namespace collisions. The
`__load_preamble` function appends the desired content to the preamble. Finally
you should add the following code to the class to make it behave more like a
`PyLaTeXObject`:

```python
from contextlib import contextmanager

... # Class definition from above

  def append(self, other):
    self._inner.append(other)

  @property
  def data(self):
    return self._inner.data

  @data.setter
  def data_set(self, other):
    self._inner.data = other

  @contextmanager
  def create(self, child):
    """Add a LaTeX object to current container, context-manager style.

        Args
        ----
        child: `~.Container`
            An object to be added to the current container
      """

    prev_data = self.data
    self.data_set = child.data  # This way append works appends to the child

    yield child  # allows with ... as to be used as well

    self.data_set = prev_data
    self.append(child)
```

#### Building your own plots

If you wish to build your plots, you can build a `matplotlib.figure.Figure` and
pass it to `dicomnode.report.plot.Plot`, at which point it will place your
figure in the report.
