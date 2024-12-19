"""This module contain various function, helping with input validation

The goal is that the cuda code doesn't need to practice defensive coding,
such that the cuda code can be simpler, since dealing with problems is much
easier in python than cuda-cpp.

In other words it's this modules function responsibility to ensure that the cuda
function CANNOT FAIL.
"""