# Python standard Library
from dataclasses import dataclass, field

# Third party packages
from pydicom import Dataset
from pylatex import PageStyle, Head, MiniPage, NoEscape, StandAloneGraphic

# Dicomnode packages
from dicomnode import report
from dicomnode.report.base_classes import LaTeXComponent


@dataclass
class ReportHeader(LaTeXComponent):
  icon_path: str
  lines : field(default_factory=list)

  def append_to(self, document: 'report.Report'):
    """Adds a standardized document header to a document

    Args:
        document (Document): Report that this document header is added to.
    """
    header = PageStyle("header", header_thickness=1)

    with header.create(Head('L')) as header_left:
      with header_left.create(MiniPage(width=NoEscape(r"0.49\textwidth"))) as wrapper:
        icon_path = self.icon_path
        wrapper.append(StandAloneGraphic(filename=icon_path,
          image_options=NoEscape("width=120pt")
        ))

    with header.create(Head('R')) as header_right:
      with header_right.create(MiniPage(width=NoEscape(r"0.49\textwidth"), pos='r', align='r')) as wrapper:
        for line in self.lines:
          report.add_line(wrapper, line)

    document.preamble.append(header)
    document.change_document_style("header")

  @classmethod
  def from_dicom(cls, icon_path, dataset: Dataset):
    lines = [
      dataset.InstitutionalDepartmentName,
      dataset.InstitutionName,
      dataset.InstitutionAddress,
    ]

    return cls(icon_path, lines)
