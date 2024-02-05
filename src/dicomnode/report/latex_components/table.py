from dataclasses import dataclass

# Python standard library
from dataclasses import dataclass, field
from enum import Enum
from typing import List

# Third party packages
from pylatex import Package, Tabularx
from pylatex.utils import bold

# Dicomnode packages
from dicomnode import report
from dicomnode.report.base_classes import LaTeXComponent

@dataclass
class Table(LaTeXComponent):
  class TableStyle(Enum):
    FULL = 1
    BORDER = 2
    TOP_BOTTOM = 3
    NONE = 4


  table_style: TableStyle
  withHeader: bool = True
  Alignment: List[str] = field(default_factory=list)
  Rows: List[List[str]] = field(default_factory=list)

  def append_to(self, report: 'report.Report'):
    """Added the table to the Report

    Args:
        table (Table): _description_
    """
    tabularx_package = Package('tabularx')
    array_package = Package('array')
    if tabularx_package not in report.packages:
      report.packages.append(tabularx_package)


    if self.table_style == Table.TableStyle.FULL:
      alignment = "| " + " | ".join(self.Alignment) + " |"
    if self.table_style == Table.TableStyle.BORDER:
      alignment = "| " + " ".join(self.Alignment) + " |"
    else:
      alignment = " ".join(self.Alignment)

    with report.create(Tabularx(alignment)) as tab:
      tab: Tabularx = tab # Just there to make intellisense happy
      if self.table_style in [Table.TableStyle.FULL, Table.TableStyle.BORDER, Table.TableStyle.TOP_BOTTOM]:
        tab.add_hline()

      if self.withHeader:
        headerRow = list(map(bold, self.Rows[0]))
      else:
        headerRow = self.Rows[0]

      tab.add_row(headerRow)

      if self.table_style in [Table.TableStyle.FULL, Table.TableStyle.BORDER, Table.TableStyle.TOP_BOTTOM]:
        tab.add_hline()

      for row in self.Rows[1:]:
        tab.add_row(row)

        if self.table_style == Table.TableStyle.FULL:
          tab.add_hline()

      # Full is missing due to line already being added
      if self.table_style in [Table.TableStyle.BORDER, Table.TableStyle.TOP_BOTTOM]:
        tab.add_hline()
