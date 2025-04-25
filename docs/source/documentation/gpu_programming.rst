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

I'll also apologize for the general structure of the C++ part, as it's my first
project and therefore contractually obligated to make some really fucking dumb
decisions.

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

    * It's not compiled, which will produce slower code. Well I know that they
      have compilation, but I doubt it can reach the same level of speed as nvcc
      and gcc / g++

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

As a rule of thumb, it will be compiled with the latest version of CUDA/GCC/C++
standard. However most of the base is C++20 and using CUDA 11.0 or later.

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
  into python objects that the python process can use. All the objects should
  have a method hooking it's function to the *_cuda* module.

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

This structure isn't ideal, because most of the code changes are in the header
library which in turn causes recompilation of all the modules. This can properly
be fixed by a better separation of submodules but this my first c++ project so
cut me some slack.

Actually Code
*************

Okay so after a 1000 words, we are finally at the actual C++ / CUDA code.
Pybind handles the conversion of objects from python to C++ and back again.

In general build-in types can be transformed to their C++ equivalent without
much brain power, but sadly we need to pass custom classes to the C++ code
because the data only really make sense in a collection. I could just ask the
user pass all the attributes of an object or create a python wrapper function,
but that idea seams kinda stupid.

Python is a dynamically typed language, while C++ is a Static typed language.
So in C++ you have to declare all the types where as In python, you just go and
throw an exception if things don't line up. Everything is an "object."
It's the root of pythons inheritance tree. (I don't even thing you can
meta-class your way out of that.)

So the first thing you need to do is handle that dynamism. The way I do that is
by the function:

.. code-block:: cpp

  dicomNodeError_t is_instance(
    const pybind11::object& python_object,
    const char* module_name,
    const char* instance_type) {
    const pybind11::module_& space_module = pybind11::module_::import(module_name);
    const pybind11::object& space_class = space_module.attr(instance_type);
    if(!pybind11::isinstance(python_object, space_class)){
      return dicomNodeError_t::INPUT_TYPE_ERROR;
    }
      return dicomNodeError_t::SUCCESS;
  }

I am big fan of errors as values, because they allow pretty cool programming
patterns. Also known as monads. (DUN DUN DUN, something something monads is just
a monoid in category of endofunctors)
Rather than embarking some useless monad tutorial I would much rather show you
some CUDA code:

.. code-block:: cpp

  void some_cuda_function(float* host_float){

  cudaError_t error;
  float* device_float;

  error = cudaMalloc(&device_float, sizeof(float));
  if(error){
    // Do some error handling also stop execution
    return;
  }
  error = cudaMemcpy(
    device_float,
    host_float,
    sizeof(float),
    cudaMemcpyDefault
  );

  if(error){
    // Do some error handling, free resources also stop execution
    return;
  }

  some_kernel<<<1,1>>>(device_float);

  error = cudaGetLastError();

  if(error){
    // Do some error handling, free resources also stop execution
    return;
  }

  error = cudaMemcpy(
    host_float,
    device_float,
    sizeof(float),
    cudaMemcpyDefault
  );

  if(error){
    // Do some error handling, free resources also stop execution
    return;
  }

  error = cudaFree(device_float);
  if(error){
    // Do some error handling, free resources also stop execution
    return;
  }

  }

So as programmers, your DRY (don't repeat yourself) sense, should be tickling.
You can see there's multiple "if error return"-statements. The way to get rid
them is by using a monad:

.. code-block:: cpp

  class CudaRunner{
    public:
      CudaRunner(std::function<void(cudaError_t)> error_function)
        : m_error_function(error_function){}

      CudaRunner()
        : m_error_function([](cudaError_t error){}){}

      cudaError_t error() const {
        return m_error;
      }

      CudaRunner& operator|(std::function<cudaError_t()> func) {
        if(m_error == cudaSuccess){
          m_error = func();
          if (m_error != cudaSuccess){
            m_error_function(m_error);
          }
        }
        return *this;
      };

    private:
      std::function<void(cudaError_t)> m_error_function;
      cudaError_t m_error = cudaSuccess;
  };

What this class does is run lambda functions, if the success value is still
success, and runs an error function if it fails. The above code example becomes:

.. code-block:: cpp

  void some_cuda_function(float* host_float){

  float* device_float;
  cudaRunner runner{[&](cudaError_t error){
    // Do something clean up?
  }};

  runner | [&](){ return cudaMalloc(&device_float, sizeof(float)); }
         | [&](){ return cudaMemcpy(
                            device_float,
                            host_float,
                            sizeof(float),
                            cudaMemcpyDefault);
       } | [&](){
        some_kernel<<<1,1>>>(device_float);
        return cudaGetLastError();
       } | [&](){
        return cudaMemcpy(
                  host_float,
                  device_float,
                  sizeof(float),
                  cudaMemcpyDefault);
       } | [&](){
        return cudaFree(device_float);
       };
  }

While I will agree this code looks funky at first, especially if you're not used
to C++ lambda syntax. But it allows you to focus on the happy path and the error
path when applicable.

This concept is a "maybe"-monad if you squint at it sides ways. But that really
doesn't matter what some imaginary haskell lecture from Yale thinks about the
classification of the runner class. What matter is that this works, and it does.

So Instead using the cudaError class I created my own error class, which is
returned from all of dicomnode's C++ functions, that can fail. It also encodes
all cudaError_t by flipping the 32'th bit.

RAII and CUDA
*************

So one of the core concepts of CPP is RAII meaning: Resource acquisition is
initialization. The point of this to automate resource acquisition and
releasing. A consequence of this that you gain quite references in your code,
because you want to avoid creating/copying/destroying resources. Because these
can fail and are pain to deal with, and then you pass the owning object around
instead.
RAII is a really good idea, however the concept sorta breaks with CUDA because:

1. A CUDA program is really just a C/C++ program in disguise. It sets up the
driver context and more important for this problem it does the tear down.
However because there's no order of destruction you can end up needing the
destroyed context to destroy an object. In general Kernel launches and copy
constructors, constructors and destructors really doesn't play nice. There's a
bit on in the `CUDA Standard https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-function-argument-processing`
2. it introduces a separate memory space namely the device memory. You cannot pass
by default a reference to a kernel, because that object live in the CPU's memory
space and the gpu threads cannot access that space. Just like the CPU cannot
access memory on the GPU.

There's a `Medium article https://medium.com/@dmitrijtichonov/cuda-series-memory-and-allocation-fce29c965d37`
which does a pretty good job of explaining the memory model that cuda uses.

While there exists abstraction in the driver that enable the program to overcome
these limitation, they come at the cost of performance. In the authors view
this problem is not something that automation should do. It's the responsibility
of the programer to understand where object exists in what stage of the
execution. This is because the bandwidth between CPU and GPU should be
bottleneck.

If you take a look at the specs of the NVidia 5090 you cna see that it have a
bandwidth of 1.79 TB/s, while it has 104.8 Tflops for 32bit floating point.
Which means you can roughly do 104.8 / (1.79 / 4) ~ 230 floating point
operations per float transferred. That means that if you do less than 230
operation on each float, then your bottleneck will be transfer speed.

Okay sidetrack over. So what do you do now that RAII is out the window.
The plan is to create thin objects, that are copied on kernel invocation. The
kernels have an argument size limit of 256 bytes, so thin object has be smaller
that that. These objects are zero initialized in general.

TESTING
*******

Now it's not really sufficient to have a single test suite in python, as any c++
function really have three parts. An unwrapping of the python construct to c++
readable data, some processing of said data and finally repacking the data for
python. The unit in unit test would prefer it have the units be as small as
possible, hence there's a c++ test suite, that tests just the processing part.

This suite is only build if you build the library manually i.e not by pip.
To build and run it:

.. code-block:: bash

  cd build
  make
  ./cu_tests

Additionally you should run the test with various tools to ensure there are no
memory leaks:

.. code-block::bash

  compute-sanitizer --leak-check full --tool=memcheck ./cu_tests



Conclusion
**********

Honestly I don't really know what else is relevant information, for the GPU
programming.

Yay speed ups!

.. toctree::
    :hidden:
