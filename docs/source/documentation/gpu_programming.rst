GPU programming
===============

I tend to write the first draft of these type of documents with snappy,
opinionated and with extra useless commentary. So if you read this, yeah sorry.

Context
*******

I think rather than jumping into documentation, I would rather start with
the context and mindset, there's behind creating an external GPU programming.
The reason for introducing GPU is performance. Which in my experience is very
lacking in medical software as correctness have taken too many seats in the
sofa. Yeah it's the second most important software quality (readability being
first, fucking fight me!), that's doesn't mean you can neglect performance.

I might also have ranted about the usage of external programs. So I'll give the
TLDR of that rant. Other programs implies usage of the file system, which is
fucking slow and error prone.

Obviously writing fast python is what we in the bizz call STUPID, so we have
offload to C/C++ and if we do that, it's a hop skip and a long jump to CUDA.

To put this into perspective, your can get some rather stupid speedups in the
x300 ballpark with GPU programming.

Options
*******

Now that I have convinced GPU programming is important what are the options:

* Pytorch:

  * Pros:

    * The frame work will be common dependency on project that use Dicom Node,
      meaning that the dependency is free.

    * The users can read the library code, since it's all in python land.
      It's not compiled, i.e easy to change source code

    * The code also works for CPU, meaning that there's no duplication of code
      and no need to synchronize the two implementations as the library should
      support non GPU platforms

  * Cons:

    * The frame work will be common dependency on project that use Dicom Node,
      meaning that there could be version conflicts between the dicomnode's
      dependency and the projects dependency.

    * It's not compiled, which will produce slower code.

    * It's at a very high level of abstraction, which will cost performance.
      Now this is sorta something I am pulling out of my ass as I do not know how
      pytorch is implemented, but the main issue is that the high abstraction
      level doesn't provide many constraints, which don't allow many
      optimizations.

    * It doesn't allow allocation and usage of dedicated hardware.

* CuPy For writing JIT kernels.

  * Pros:

    * It could work, maybe?

  * Cons:

    * In general I really really really dislike writing other programming
      languages inside of another language. Especially as sensitive as C++.

    * JIT compilation also takes performance.

    * Honestly I can't wrap my head around how the memory model would work.

* Pybind11, Rawdogging Cuda.

  * Pros:

    * This goes WROOOOOOOM. A compiled binary is the theoretical fastest options
      assuming equal competence among programmers.

    * It gives the library author the most freedom in specifying the level of
      abstraction for the user.

    * It allows the usage of dedicated hardware which can speed things up.

  * Cons:

    * The library author is heavily biased for this option.
    * The Setup is a nightmare.
    * It's another programming language, which means that author need to have a
      more programming languages under their belt. It also means that you end up
      with two different testsuites.
    * It's C++, which is a very hard language to program as it introduces a newbie
      class of problems. Undefined behavior and memory leaks.

As the author is what we in bizz call a FUCKING NEERD, so obviously I selected
pybind11.

The Setup
*********

Alright, so the goal is to produce a library that the python process can hook
into and run CUDA code from there. There's a few libraries for this, however
as I didn't really have a good look of the landscape I picked something that
could get the job done and not the "best" choice.

I would have looked more into boost's python bindings, but it's whatever.

This story starts in the *setup.py* where the script checks if the CUDA compiler
nvcc is installed. If it's the script adds an extension module. From python's
point of view this is just a fancy module without any python source.

The CMakeExtension object really about delegating the responsibility of defining
the extension to Cmake rather than the Setup file. This is mostly because the
compilation step is rather complicated and CMake is a tool for encapsulating
this complexity.

Now there's some short-comings of the current setup. Because if there's no CUDA
compiler, no extension is generated. In the future I might want to have an C++
extension to cover from the missing cuda library. This is however not a trivial
problem, because there's a lot of duplication between a CUDA and C++
implementations. This requires a synchronization between three code bases:
python, CUDA and C++. Also you can't really reuse the code between the C++ and
CUDA because CUDA code is not valid C++, hence you either need two source files
or littering the source code with NVCC flags to determine which code to compile.

It might also fuck with things because NVCC forwards the host code to another
compiler anyways...

Back to the useful stuff: The CMake extension is defined by the CMakeLists.txt
in the root. The file, the directory *low_level_src* and the general structure
of the final python library is closely linked. So let's start with the
directory:

*low_level_src* should contain 3 directories and 2 files:

* gpu_code - This directory contains a 'header only'-library, that's does the
  actual computation. All the kernels should be in here. The code is should be
  oblivious to the fact that it's pybind11 that calls it.

* python - Python contains some objects with the responsibility of extracting
  the data from python objects and then with the extracted data call some
  functions inside of *gpu_code* and then taking that result and putting it back
  into python objects that the python process can use. All the objects should have
  a method hooking it's function to the *_cuda* module. See below:

.. code-block:: cpp

  void apply_XXX(pybind11::module& m);

* tests - tests is for unit tests of the mostly the header library. I found it
  impossible to test the python module without a python instance, which is also
  why I separated the python block and the gpu code.

* python_entry_point.cu - file which is the entrypoint for the CMake. It creates
  the _cuda module and passes it around to the objects of the directory *python*
  and ships it.

* test_initiator.cu - Entry point for the unit tests. It should grab all the
  unit tests of the directory *tests*

There's one special module called utilities.cu, which is an object that contains
function for extracting data from the python object and constructing the
corresponding CUDA object. there's a compile diagram:

.. image:: /_static/compile_tree.svg
  :target: /_static/compile_tree.svg

.. toctree::
    :hidden:
