import argparse

import os
import shutil
from pathlib import Path

from unittest import TextTestRunner, TestSuite, TestLoader

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Testing tool for DicomNode library")
  parser.add_argument("--verbose", type=int, default=1)
  parser.add_argument("-p", "--performance", action='store_true')

  args = parser.parse_args()


  runner = TextTestRunner()
  loader = TestLoader()
  loader.testMethodPrefix = "performance"
  suite: TestSuite = loader.discover("src")

  runner.run(suite)
