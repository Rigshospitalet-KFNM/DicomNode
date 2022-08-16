""" This is the main entry point for the omnitool, which is a collections of scripts 


"""
__author__ = "Christoffer Vilsturp Jensen"

import argparse
import sys

def entry_func():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "tool",
    type=str,
    help="The tool to be used",
    choices=["show"]
  )

  args = parser.parse_args()

