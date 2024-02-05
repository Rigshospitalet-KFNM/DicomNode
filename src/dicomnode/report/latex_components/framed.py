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

  def __init__(self) -> None:
    super().__init__()
    self._inner = self.DicomFrame()

  def append_to(self, document: Report):
    if not self.__class__.__name__ in document.loaded_preambles:
      # Load The preamble
      self.__load_preamble(document.preamble)
      document.loaded_preambles.add(self.__class__.__name__)
    document.append(self._inner)

  def __load_preamble(self, preamble):
      preamble.append(NoEscape('\\definecolor{navy}{HTML}{0000AA}'))
      preamble.append(NoEscape('\\newlength{\\frameTweak}'))
      preamble.append(NoEscape('\\mdfdefinestyle{FrameStyle}{'))
      preamble.append(NoEscape('    linecolor=navy,'))
      preamble.append(NoEscape('    outerlinewidth=5pt,'))
      preamble.append(NoEscape('    innertopmargin=5pt,'))
      preamble.append(NoEscape('    innerbottommargin=5pt,'))
      preamble.append(NoEscape('    innerrightmargin=5pt,'))
      preamble.append(NoEscape('    innerleftmargin=5pt,'))
      preamble.append(NoEscape('    leftmargin = 5pt,'))
      preamble.append(NoEscape('    rightmargin = 5pt'))
      preamble.append(NoEscape('}'))
      preamble.append(NoEscape('\\NewEnviron{dicomframe}[1][]{'))
      preamble.append(NoEscape('    \\setlength{\\frameTweak}{\\dimexpr'))
      preamble.append(NoEscape('    +\\mdflength{innerleftmargin}'))
      preamble.append(NoEscape('    +\\mdflength{innerrightmargin}'))
      preamble.append(NoEscape('    +\\mdflength{leftmargin}'))
      preamble.append(NoEscape('    +\\mdflength{rightmargin}'))
      preamble.append(NoEscape('    }'))
      preamble.append(NoEscape('\\savebox0{'))
      preamble.append(NoEscape('    \\begin{varwidth}{\\dimexpr\\linewidth-\\frameTweak\\relax}'))
      preamble.append(NoEscape('        \\BODY'))
      preamble.append(NoEscape('    \\end{varwidth}'))
      preamble.append(NoEscape('}'))
      preamble.append(NoEscape('\\begin{mdframed}[style=FrameStyle,backgroundcolor=white,userdefinedwidth=\\dimexpr\\wd0+\\frameTweak\\relax, #1]'))
      preamble.append(NoEscape('    \\usebox0'))
      preamble.append(NoEscape('\\end{mdframed}'))
      preamble.append(NoEscape('}'))


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
