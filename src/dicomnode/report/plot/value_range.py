# Python Standard Library 
from enum import Enum
from typing import List

# Third party Packages
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Polygon

# Dicomnode packages
from dicomnode.report.plot import Plot

class BarColors(Enum):
  GOOD = (0.15, .90, 0.0)
  CONCERNING = (1.0, 0.95, 0.25)
  PROBLEMATIC = (1.0,0.6, 0.15)
  CRITICAL = (0.9,0.0,0.0)

class ValueRange(Plot):
  @classmethod
  def add_rect(cls, axes: Axes, ranges: List[float], colors: List[BarColors], point: float):
    axes.set_axis_off()

    if not all(ranges[i] <= ranges[i + 1] for i in range(len(ranges) - 1)):
      raise Exception

    if len(ranges) != len(colors) + 1:
      print(len(ranges), len(colors) + 1)
      raise Exception

    aspect_ratio = 1.15 / 5

    #Constants
    line_width = 0.02
    line_width_offset = line_width / 2
    vertical_line_width = line_width * aspect_ratio
    rect_height = 0.5

    min_val = ranges[0]
    max_val = ranges[-1]

    span = max_val - min_val

    bar_lower = 0.5 - rect_height / 2
    bar_upper = 0.5 + rect_height / 2

    text_upper = bar_upper + 0.1
    text_lower = bar_lower - 0.3

    for i in range(len(ranges) - 1):
      color = colors[i].value
      start = (ranges[i] - min_val) / span
      end = (ranges[i +1] - min_val) / span

      rect_width = end - start # these have been normalized by span

      rect_left_bar = Rectangle((start, bar_lower), vertical_line_width, rect_height, facecolor=color)
      rect_right_bar = Rectangle((end - vertical_line_width, bar_lower), vertical_line_width, rect_height, facecolor=color)
      rect_floor_bar = Rectangle((start,bar_lower), rect_width, line_width, facecolor=color)
      rect_roof_bar = Rectangle((start,bar_upper), rect_width, line_width, facecolor=color)

      axes.add_artist(rect_left_bar)
      axes.add_artist(rect_right_bar)
      axes.add_artist(rect_floor_bar)
      axes.add_artist(rect_roof_bar)

      if ranges[i] < point < ranges[i+1]:
        point_x = (point - min_val) / span - line_width_offset
        text_x_offset = len(str(point)) * 0.01 - line_width_offset
        axes.text(point_x - text_x_offset, text_lower, str(point), color=color)

        rect_point = Rectangle((point_x, bar_lower + 0.01),
                               vertical_line_width,
                               rect_height, color=color)
        #axes.add_artist(rect_point)

        arrow_head_y = 0.125
        arrow_head_x = 0.03

        arrow_header_start_y = bar_lower - line_width

        arrow_head = Polygon([[point_x + line_width_offset / 3, arrow_header_start_y],
                              [point_x + line_width_offset / 3 + arrow_head_x / 2, arrow_header_start_y - arrow_head_y],
                              [point_x + line_width_offset / 3, arrow_header_start_y - arrow_head_y / 2],
                              [point_x + line_width_offset / 3 - arrow_head_x / 2, arrow_header_start_y - arrow_head_y]
                         ], color =color) # type: ignore # Umm yes this array isn't made of array, no it doesn't work with tuples
        axes.add_artist(arrow_head)


      text_x_offset = len(str(ranges[i])) * 0.01
      axes.text(start - text_x_offset, text_upper, str(ranges[i]), color=color)
      if i + 2 == len(ranges):
        text_x_offset = len(str(ranges[i + 1])) * 0.01
        axes.text(end - text_x_offset, text_upper, str(ranges[i + 1]), color=color)
