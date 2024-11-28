# Python Standard Library
import sys
import argparse
import os
import shutil
import re
from pathlib import Path
from unittest import TextTestRunner, TestSuite, TestLoader, TestCase
from typing import  Set,  Union

TESTING_TEMPORARY_DIRECTORY = "/tmp/pipeline_tests"
os.environ['DICOMNODE_TESTING_TEMPORARY_DIRECTORY'] = TESTING_TEMPORARY_DIRECTORY
# DICOMNODE_TESTING_TEMPORARY_DIRECTORY must be set before importing

from tests.helpers import testing_logs

PYTHON_3_12_PLUS = 12 <= sys.version_info.minor

def handle_tests(running_suite: TestSuite,
                 tests: Union[TestCase, TestSuite],
                 pattern: str,
                 added_set: Set[str]):
  if isinstance(tests, TestSuite):
    for sub_tests in tests:
      handle_tests(running_suite, sub_tests, pattern, added_set)
  else:
    test_regex = re.compile(pattern)

    if test_regex.search(tests._testMethodName.lower()):
      key = tests._testMethodName.lower() # this is gonna pwn me later :(
      if key not in  added_set:
        running_suite.addTest(tests)
        added_set.add(key)


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
  running_suite = TestSuite()

  added_tests = set()

  all_suite: TestSuite = loader.discover("tests")
  if args.performance:
    loader.testMethodPrefix = "performance"
    performance_tests = loader.discover("tests")
    all_suite.addTests(performance_tests)

  for file_suite in all_suite:
    handle_tests(running_suite, file_suite, args.test_regex, added_tests)

  cwd = os.getcwd()
  tmpDirPath = Path(TESTING_TEMPORARY_DIRECTORY)

  tmpDirPath.mkdir(mode=0o777, exist_ok=True)

  os.chdir(TESTING_TEMPORARY_DIRECTORY)
  result = runner.run(running_suite)
  os.chdir(cwd)

  if not args.no_clean_up:
    if tmpDirPath.exists():
      shutil.rmtree(str(tmpDirPath))

  exit(len(result.errors) != 0)