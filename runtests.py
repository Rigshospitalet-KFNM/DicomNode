# Python Standard Library
import sys
import argparse
import os
import shutil
from pathlib import Path
from unittest import TextTestRunner, TestSuite, TestLoader

TESTING_TEMPORARY_DIRECTORY = "/tmp/pipeline_tests"
os.environ['DICOMNODE_TESTING_TEMPORARY_DIRECTORY'] = TESTING_TEMPORARY_DIRECTORY
# DICOMNODE_TESTING_TEMPORARY_DIRECTORY must be set before importing

from tests.helpers import testing_logs

PYTHON_3_12_PLUS = 12 <= sys.version_info.minor


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Testing tool for DicomNode library")
  parser.add_argument("test_regex", default="test", nargs="?")
  parser.add_argument("--verbose", "-v", type=int, default=1)
  parser.add_argument("-nc", "--no_clean_up", action='store_true')
  parser.add_argument("-p", "--performance", action='store_true')
  if PYTHON_3_12_PLUS:
    parser.add_argument("-d", "--duration", action='store_true')

  args = parser.parse_args()
  testing_logs()

  if PYTHON_3_12_PLUS:
    if args.duration:
      duration_args = 100000
    else:
      duration_args = None

    runner = TextTestRunner(verbosity=args.verbose, durations=duration_args) #type: ignore
  else:
    runner = TextTestRunner(verbosity=args.verbose)

  loader = TestLoader()
  suite: TestSuite = loader.discover("tests", pattern=f"*{args.test_regex}*.py")
  if args.performance:
    loader.testMethodPrefix = "performance"
    performance_tests = loader.discover("tests")
    suite.addTests(performance_tests)

  cwd = os.getcwd()
  tmpDirPath = Path(TESTING_TEMPORARY_DIRECTORY)

  tmpDirPath.mkdir(mode=0o777, exist_ok=True)

  os.chdir(TESTING_TEMPORARY_DIRECTORY)
  result = runner.run(suite)
  os.chdir(cwd)

  if not args.no_clean_up:
    if tmpDirPath.exists():
      shutil.rmtree(str(tmpDirPath))

  exit(len(result.errors) != 0)