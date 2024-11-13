*******************
Design of Dicomnode
*******************

*Notice: this document is intended to a place where the overarching design of
dicom node is discussed. It's intended to read by maintainers of the library*

Use case of the library
***********************

When setting up a service, that works with medical images. As communication
protocol, you'll be expected to use DICOM, and dicom is complex and hard.

There exists a significant amount of tool kits and frameworks for handling dicom
images. This tool is to provide a framework for building a server, that receives
medical images, does some processing, and sends the result of the processing
over dicom again.

The library should include any tool or concept, that two dicom servers might
share. It should promote good coding practices and provide a highly stable
server, that is able to help diagnose problems that the user might have made.

The library is easy. It should promote simplicity in user code. It should
attempt to excel in its simplicity. This means that the target use case for this
library is a single server without the whole virtualization shebang.

I would also like to apologize in advance for these documents as they can get
very circle-jerky with my own opinions.

A word on **Complexity**
************************

In my own overinflated opinion, there are two axes of qualifiers to describe the
overall difficulty of a task.

* Easy to hard: A task is easy if it require little effort by the user and is
    hard to fuck up. Where as hard things require effort or is easy to fuck up,
    and therefore precision to complete successfully.
* Simple to complex. This axis describe how many factors have an influence over
    the result, how many outcomes are possible, and how many fail states and
    their resolution exists.

There is not an international standard designating the difficulty of a task,
but these axes are pylons to navigate against when comparing different functions
that solve the same problem. An excellent small exercise is to compare two
programs, that both read a json object and then print it, however: One is
written in C++ and the other is written in python.

Functions are an excellent way to make tasks easy, but making things simpler is
much more difficult and might not possible. There's no way to make program that
invert matrices without a consideration to singular matrices. The only way to
make things simpler is by stripping out complexity. Complexity comes from many
sources. Sometimes added due to laziness others the complexity add some other
value. Consider two functions for matrix multiplication, one is single threaded,
whereas the other is gpu accelerated. The complexity allows for utilization of
hardware resources and therefore better performance. But opens you to bunch of
complexity that can kick you in the nut.

For instance, if you have million function calls of tiny matrices, you will be
IO bound on the gpu, and suffer inferior performance to the simple version. You
also open yourself a million errors from the interacting GPU drivers or your
code might be running on a machine without a GPU.

DICOM, Hard and Complex
=======================

A quick glance at the `_Dicom_Standard` that shows the compressed standard is
180837858 bytes or 180 MB of text and images. Consider the lord of the rings is
by my back of napkin estimation is around 6 MB of text information, DICOM
complexity is well accounted for.

The hardness of DICOM is primary a consequence of how easy it is to construct a
special case is to snipe your program. Good examples are dicom series where the
topogram (scout) from scan is included in the series, series with different
slice thickness, or just series that include some private tags doesn't conform
to the standard.

The point is there's only so much *easy-fication* one can do before some edge
case snipes your program or library. A complexity turd remains a complexity
turd.

Simplification
==============

This library is not the first tool to attempt to encapsulate the complexity of
dicom nor will it be the last. However as a goal it attempt to simplify the
entire process i.e. to strip out unnecessary complexity. One things most
programs do is convert their images from the dicom to a processing friendly
image format. This often requires external programs, which themselves require
files to function. All of this complexity is not needed, and I've yet to a
research project that handles a invalid write to disk.

As such it's the goal of this library to strip these complexities out and have
users rely on good old function call with no interaction with the operating
system, unless there's a good reason for it.


Grand Structure of the library
******************************

The packages is written in python, because many of the applications that the
library try to support are machine learning based and are therefore written in
python. It also have advantage if the program have CLI, the end user can hook
into the library rather that use the CLI, if that's what is desired.

The main component is the `dicomnode.server.nodes.AbstractPipeline` which
represents the actual server. The end user should only fill out a few functions
So far:
* `process`


Inheritance, the bane of software design
========================================

The library uses inheritance as interface to the user. The end user should fill
out a few methods with


.. toctree::
    :hidden:


.. _Dicom_Standard: https://dicom.nema.org/medical/dicom/current/
