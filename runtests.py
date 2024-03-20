# Python Standard Library
import argparse
import os
import shutil
from pathlib import Path
from unittest import TextTestRunner, TestSuite, TestLoader

TESTING_TEMPORARY_DIRECTORY = "/tmp/pipeline_tests"
os.environ['DICOMNODE_TESTING_TEMPORARY_DIRECTORY'] = TESTING_TEMPORARY_DIRECTORY
# DICOMNODE_TESTING_TEMPORARY_DIRECTORY must be set before importing

from tests.helpers import testing_logs

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Testing tool for DicomNode library")
  parser.add_argument("test_regex", default="test", nargs="?")
  parser.add_argument("--verbose", type=int, default=1)
  parser.add_argument("-nc", "--no_clean_up", action='store_true')
  parser.add_argument("-p", "--performance", action='store_true')

  args = parser.parse_args()
  testing_logs()

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