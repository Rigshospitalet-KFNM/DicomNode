""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from contextlib import contextmanager

# Third Party Packages
from pylatex import Package, NoEscape
from pylatex.base_classes import Environment

# Dicomnode packages
from dicomnode.report import Report
from dicomnode.report.base_classes import LaTeXComponent

class DicomFrame(LaTeXComponent):
  class DicomFrame(Environment):
    packages = [Package('mdframed'), Package('xcolor'), Package('color'), Package('varwidth'), Package('environ'), Package('calc')]
    omit_if_empty = True

  def append_to(self, document: Report):
    if not self.__class__.__name__ in document.loaded_preambles:
      document.preamble.append(NoEscape('\\definecolor{navy}{HTML}{0000AA}'))
      document.preamble.append(NoEscape('\\newlength{\\frameTweak}'))
      document.preamble.append(NoEscape('\\mdfdefinestyle{MyFrame}{'))
      document.preamble.append(NoEscape('    linecolor=navy,'))
      document.preamble.append(NoEscape('    outerlinewidth=5pt,'))
      document.preamble.append(NoEscape('    innertopmargin=5pt,'))
      document.preamble.append(NoEscape('    innerbottommargin=5pt,'))
      document.preamble.append(NoEscape('    innerrightmargin=5pt,'))
      document.preamble.append(NoEscape('    innerleftmargin=5pt,'))
      document.preamble.append(NoEscape('    leftmargin = 5pt,'))
      document.preamble.append(NoEscape('    rightmargin = 5pt'))
      document.preamble.append(NoEscape('}'))
      document.preamble.append(NoEscape('\\NewEnviron{dicomframe}[1][]{'))
      document.preamble.append(NoEscape('    \\setlength{\\frameTweak}{\\dimexpr'))
      document.preamble.append(NoEscape('    +\\mdflength{innerleftmargin}'))
      document.preamble.append(NoEscape('    +\\mdflength{innerrightmargin}'))
      document.preamble.append(NoEscape('    +\\mdflength{leftmargin}'))
      document.preamble.append(NoEscape('    +\\mdflength{rightmargin}'))
      document.preamble.append(NoEscape('    }'))
      document.preamble.append(NoEscape('\\savebox0{'))
      document.preamble.append(NoEscape('    \\begin{varwidth}{\\dimexpr\\linewidth-\\frameTweak\\relax}'))
      document.preamble.append(NoEscape('        \\BODY'))
      document.preamble.append(NoEscape('    \\end{varwidth}'))
      document.preamble.append(NoEscape('}'))
      document.preamble.append(NoEscape('\\begin{mdframed}[style=MyFrame,backgroundcolor=white,userdefinedwidth=\\dimexpr\\wd0+\\frameTweak\\relax, #1]'))
      document.preamble.append(NoEscape('    \\usebox0'))
      document.preamble.append(NoEscape('\\end{mdframed}'))
      document.preamble.append(NoEscape('}'))
      document.loaded_preambles.add(self.__class__.__name__)


    document.append(self._inner)

  def __init__(self) -> None:
    super().__init__()
    self._inner = self.DicomFrame()

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
    print(type(self), dir(self))
    self.data_set = child.data  # This way append works appends to the child

    yield child  # allows with ... as to be used as well

    self.data_set = prev_data
    self.append(child)
