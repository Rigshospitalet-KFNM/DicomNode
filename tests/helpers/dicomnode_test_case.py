import collections
from enum import IntEnum
from re import compile,Pattern
from typing import List, Optional, Union
from pprint import pformat
import logging
from unittest import TestCase
from unittest.case import _BaseTestCaseContext
from unittest.util import safe_repr

# Dicomnode
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.lib.regex import escape_pattern

RegexAble = Union[Pattern, str]



_LoggingWatcher = collections.namedtuple("_LoggingWatcher",
                                         ["records", "output"])

class _CapturingHandler(logging.Handler):
    """
    A logging handler capturing all (raw and formatted) logging output.
    """

    def __init__(self):
        logging.Handler.__init__(self)
        self.watcher = _LoggingWatcher([], [])

    def flush(self):
        pass

    def emit(self, record):
        self.watcher.records.append(record)
        msg = self.format(record)
        self.watcher.output.append(msg)


class _AssertNonCapturingLogsContext(_BaseTestCaseContext):
    """A context manager for assertNonCapturingLogs() and assertNonCapturingNoLogs() """

    LOGGING_FORMAT = "%(process)s - %(levelname)s:%(name)s:%(message)s"

    def __init__(self, test_case, logger_name, level, no_logs, raise_error=True):
        _BaseTestCaseContext.__init__(self, test_case)
        self.logger_name = logger_name
        if level:
            self.level = logging._nameToLevel.get(level, level)
        else:
            self.level = logging.INFO
        self.msg = None
        self.no_logs = no_logs
        self.raise_error = raise_error

    def __enter__(self):
        if isinstance(self.logger_name, logging.Logger):
          logger = self.logger = self.logger_name
        else:
          logger = self.logger = logging.getLogger(self.logger_name)
        formatter = logging.Formatter(self.LOGGING_FORMAT)
        handler = _CapturingHandler()
        handler.setLevel(self.level)
        handler.setFormatter(formatter)
        self.watcher = handler.watcher
        self.old_handlers = logger.handlers[:]
        self.old_level = logger.level
        self.old_propagate = logger.propagate
        logger.handlers.append(handler)
        logger.setLevel(self.level)
        logger.propagate = False
        if self.no_logs:
            return
        return handler.watcher

    def __exit__(self, exc_type, exc_value, tb):
        self.logger.handlers = self.old_handlers
        self.logger.propagate = self.old_propagate
        self.logger.setLevel(self.old_level)

        if exc_type is not None:
            # let unexpected exceptions pass through
            return False

        if self.no_logs:
            # assertNoLogs
            if len(self.watcher.records) > 0 and self.raise_error:
                self._raiseFailure(
                    "Unexpected logs found: {!r}".format(
                        self.watcher.output
                    )
                )

        else:
            # assertLogs
            if len(self.watcher.records) == 0 and self.raise_error:
                self._raiseFailure(
                    "no logs of level {} or higher triggered on {}"
                    .format(logging.getLevelName(self.level), self.logger.name))


class DicomnodeTestCase(TestCase):
  def tearDown(self) -> None:
    # Clear dicomnode logger:
    dicomnode_logger = logging.getLogger(DICOMNODE_LOGGER_NAME)
    for handler in dicomnode_logger.handlers:
      print(f"test: {self.__class__.__name__}:{self._testMethodName} leaked a dicomnode handler")
      dicomnode_logger.removeHandler(handler)

    process_logger = logging.getLogger(DICOMNODE_PROCESS_LOGGER)
    for handler in process_logger.handlers:
      print(f"test: {self.__class__.__name__}:{self._testMethodName} leaked a process handler")
      process_logger.removeHandler(handler)


  def assertRegexIn(self, regex: RegexAble, container: List[str], msg=None):
    if isinstance(regex, str):
      regex = escape_pattern(regex)

    found_pattern = False
    for str_ in container:
      found_pattern |= regex.search(str_) is not None
      if found_pattern:
        break

    if not found_pattern:
      msg = self._formatMessage(msg, f"Pattern {safe_repr(regex.pattern)} is not in {pformat(container)}")
      self.fail(msg)

  class __BeforeHandlerResponse(IntEnum):
    ORDERING_RESPECTED = 0 # success value
    REGEX_1_MISSING = 1
    REGEX_2_MISSING = 2
    ORDERING_NOT_RESPECTED = 3
    MATCH_TO_SAME_INDEX = 4

  def __beforeHandler(self, regex_1: Pattern, regex_2: Pattern, container: List[str]) -> __BeforeHandlerResponse:
    regex_1_index: Optional[int] = None
    regex_2_index: Optional[int] = None

    for idx, str_ in enumerate(container):
      found_regex_1 = regex_1.search(str_) is not None
      if found_regex_1:
        regex_1_index = idx
      found_regex_2 = regex_2.search(str_) is not None
      if found_regex_2:
        regex_2_index = idx

      if regex_1_index is not None and regex_2_index is not None:
        if regex_1_index < regex_2_index:
          return self.__BeforeHandlerResponse.ORDERING_RESPECTED
        elif regex_1_index > regex_2_index:
          return self.__BeforeHandlerResponse.ORDERING_NOT_RESPECTED
        else:
          return self.__BeforeHandlerResponse.MATCH_TO_SAME_INDEX

    if regex_1_index is None:
      return self.__BeforeHandlerResponse.REGEX_1_MISSING
    else:
      return self.__BeforeHandlerResponse.REGEX_2_MISSING

  def assertRegexBefore(self,
                        regex_1: RegexAble,
                        regex_2: RegexAble,
                        container: List[str], msg=None):
    if isinstance(regex_1, str):
      regex_1 = escape_pattern(regex_1)

    if isinstance(regex_2, str):
      regex_2 = escape_pattern(regex_2)

    success = self.__beforeHandler(regex_1, regex_2, container)

    if success.value == self.__BeforeHandlerResponse.ORDERING_NOT_RESPECTED:
      msg = self._formatMessage(msg, f"Pattern {safe_repr(regex_2.pattern)} appear before Pattern {safe_repr(regex_1.pattern)} in {pformat(container)}")
      self.fail(msg)

    if success.value == self.__BeforeHandlerResponse.REGEX_1_MISSING:
      msg = self._formatMessage(msg, f"Pattern {safe_repr(regex_1.pattern)} is not in {pformat(container)}")
      self.fail(msg)

    if success.value == self.__BeforeHandlerResponse.REGEX_2_MISSING:
      msg = self._formatMessage(msg, f"Pattern {safe_repr(regex_2.pattern)} is not in {pformat(container)}")
      self.fail(msg)

    if success.value == self.__BeforeHandlerResponse.MATCH_TO_SAME_INDEX:
      msg = self._formatMessage(msg, f"Pattern {safe_repr(regex_1.pattern)} and {safe_repr(regex_2.pattern)} match to the same index")
      self.fail(msg)

  def assertNonCapturingLogs(self, logger, level=logging.DEBUG, raise_error=True):
    return _AssertNonCapturingLogsContext(self, logger, level, no_logs=False, raise_error=raise_error)

  def assertNonCapturingNoLogs(self, logger, level=logging.DEBUG, raise_error=True):
    return _AssertNonCapturingLogsContext(self, logger, level, no_logs=True, raise_error=raise_error)
