import argparse

from unittest import TextTestRunner, TestSuite, TestLoader

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Testing tool for DicomNode library")
  parser.add_argument("--verbose", type=int, default=1, )

  args = parser.parse_args()


  runner = TextTestRunner()
  loader = TestLoader()
  suite = loader.discover("src")
  runner.run(suite)
