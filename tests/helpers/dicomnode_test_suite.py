from typing import Iterable
from unittest import TestSuite, TestCase, TestResult


from unittest.suite import _isnotsuite

class BaseDicomnodeTestSuite(TestSuite):
  # type: ignore
  def __init__(self, tests: Iterable[TestCase | TestSuite] = []) -> None:
    super().__init__(tests)

  def run(self, result: TestResult, debug=False):
    topLevel = False
    if getattr(result, '_testRunEntered', False) is False:
      result._testRunEntered = topLevel = True

    for index, test in enumerate(self):
        if result.shouldStop:
            break

        if _isnotsuite(test):
            self._tearDownPreviousClass(test, result)
            self._handleModuleFixture(test, result)
            self._handleClassSetUp(test, result)
            result._previousTestClass = test.__class__

            if (getattr(test.__class__, '_classSetupFailed', False) or
                getattr(result, '_moduleSetUpFailed', False)):
                continue

        self.pre_test(test)

        if not debug:
            test(result)
        else:
            test.debug()

        self.post_test(test)

        if self._cleanup:
            self._removeTestAtIndex(index)

    if topLevel:
        self._tearDownPreviousClass(None, result)
        self._handleModuleTearDown(result)
        result._testRunEntered = False
    return result


  def pre_test(self, test):
    pass

  def post_test(self, test):
    pass