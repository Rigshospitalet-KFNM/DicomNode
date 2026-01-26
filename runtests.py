# Python Standard Library
import sys
import argparse
import os
import shutil
import logging
import subprocess
import re
import psutil
import threading
from pathlib import Path
from unittest import TextTestRunner, TestSuite, TestLoader, TestCase
from typing import  Set,  Union

TESTING_TEMPORARY_DIRECTORY = "/tmp/pipeline_tests"
os.environ['DICOMNODE_TESTING_TEMPORARY_DIRECTORY'] = TESTING_TEMPORARY_DIRECTORY
os.environ['DICOMNODE_ENV_REPORT_DATA_PATH'] = os.getcwd() + "/report_data"
# DICOMNODE_TESTING_TEMPORARY_DIRECTORY must be set before importing DICOMNODE



from tests.helpers import testing_logs
from tests.helpers.dicomnode_test_suite import BaseDicomnodeTestSuite

class DicomnodeTestSuite(BaseDicomnodeTestSuite):
  def pre_test(self, test):
    #print(test._testMethodName)
    #root_logger = logging.getLogger()
    #print(f"root node handlers: {root_logger.handlers}")
    #dicomnode_logger = logging.getLogger("dicomnode")
    #print(f"Dicomnode node handlers: {dicomnode_logger.handlers}")
    #process_logger = logging.getLogger("process_dicomnode")
    #print(f"process logger handlers: {process_logger.handlers}")
    pass


  def post_test(self, test):
    #root_logger = logging.getLogger()
    #print(f"root node handlers: {root_logger.handlers}")
    #dicomnode_logger = logging.getLogger("dicomnode")
    #print(f"Dicomnode node handlers: {dicomnode_logger.handlers}")
    #process_logger = logging.getLogger("process_dicomnode")
    #print(f"process logger handlers: {process_logger.handlers}")
    pass


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
      key = tests.__class__.__name__ + tests._testMethodName.lower() # this is gonna pwn me later :(
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
  running_suite = DicomnodeTestSuite()

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

  this_process = psutil.Process()

  print(f"This process has id: {this_process.pid}")

  for process in this_process.children(True):
    # Note that python spawns a resource tracker process, that we skip
    if any(['multiprocessing.resource_tracker' in cmd for cmd in process.cmdline()]):
      continue
    else:
      print(process.cmdline())

    print(f"DEADLOCKED PROCESS! : {process}")
    try:
      pyspy_process = subprocess.run(
          ['py-spy', 'dump', '--pid', str(process.pid)],
          capture_output=True,
          text=True,
          timeout=10
      )
      print(pyspy_process.stdout)
    except Exception as e:
      print(f"Encountered error: {e}", file=sys.stderr)
    # CLICK CLICK MOTHERFUCKER
    process.terminate()

  for thread in threading.enumerate():
    if thread.name != "MainThread" and thread.is_alive():
      print(f"Thread: {thread.name} should have been killed?")

  if not args.no_clean_up:
    if tmpDirPath.exists():
      shutil.rmtree(str(tmpDirPath))

  exit(len(result.errors) != 0)