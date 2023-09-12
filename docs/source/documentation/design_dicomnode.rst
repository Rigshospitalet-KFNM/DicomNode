*******************
Design of Dicomnode
*******************

*Notice: this document is intended to a place where the overarching design of
dicom node is discussed. It's intended to read by software engineers and
computer scientists.*

Goals of the Library
====================

A significant amount of tool kits are being developed, however many of them are
"simple" programs, that are not production ready. They often use different file
formats that are better suited to processing. For instance a dicom series often
contain hundreds of images meaning hundreds of files, which the programs have
hidden by converting the series to a nifty or minc file.

As such it's the goal of the library to provide tools to convert these image
processing tools into production ready dicom servers. While hiding many of the
unintuitive aspects of the dicom protocol. The servers should hide away as much
as possible of the common aspects between pipelines. For instance many
pipelines convert a dicom series to a different file format, so that should be
part of the library. While the specific functionality of a pipeline is unique
to the pipeline and therefore shouldn't be part of this library. (With the
exception that the pipeline is an example.)


Grand Structure of the library
==============================

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
