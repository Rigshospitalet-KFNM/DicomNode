import argparse

import os
import shutil
from pathlib import Path

from unittest import TextTestRunner, TestSuite, TestLoader

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Testing tool for DicomNode library")
  parser.add_argument("--verbose", type=int, default=1, )

  args = parser.parse_args()


  runner = TextTestRunner()
  loader = TestLoader()
  suite = loader.discover("src")
  cwd = os.getcwd()
  tmpDir = "/tmp/pipeline_tests"
  tmpDirPath = Path(tmpDir)
  if tmpDirPath.exists():
    shutil.rmtree(tmpDir) #pragma: no cover
  os.mkdir(tmpDir, mode=0o777)
  os.chdir(tmpDir)
  runner.run(suite)
  os.chdir(cwd)
  shutil.rmtree(tmpDir)
