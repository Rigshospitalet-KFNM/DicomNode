# Python Standard library
from logging import DEBUG

# Third party modules

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.lib.parallelism import spawn_process, ProgramOutput, Parallel, ParallelPrimitive

# Test modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

PRINTED_MESSAGE = "GOOOD MORNING VIETNAM"

def process_function(message_to_print):
  print(message_to_print)

class ParallelismTestCase(DicomnodeTestCase):
  def test_parallel_capture_output_from_process(self):
    with self.assertLogs(DICOMNODE_LOGGER_NAME, DEBUG):
      process, queue = spawn_process(process_function, PRINTED_MESSAGE, start=True, capture_output=True)
      process.join()

      if queue is None:
        raise AssertionError("Queue should not be None!")

      output = queue.get()
      queue.close()

      if not isinstance(output, ProgramOutput):
        raise AssertionError("Output Should be ProgramOutput type")

      self.assertIn(PRINTED_MESSAGE, output.stdout)
      self.assertEqual("", output.stderr)

  def test_parallel_primitive_capture_output(self):
    with self.assertLogs(DICOMNODE_LOGGER_NAME, DEBUG):
      parallel = Parallel(
        ParallelPrimitive.PROCESS,
        process_function,
        PRINTED_MESSAGE,
        capture_output=True
      )

      output = parallel.get_output()

      if not isinstance(output, ProgramOutput):
        raise AssertionError("Output Should be ProgramOutput type")

      self.assertIn(PRINTED_MESSAGE, output.stdout)
      self.assertEqual("", output.stderr)