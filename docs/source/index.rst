.. dicom node documentation master file, created by
   sphinx-quickstart on Tue Apr 18 08:27:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dicom node's documentation!
======================================

**Dicomnode** is a Python library for creating dicom-to-dicom post processing
dicom nodes. It aims to standardize and provide robustness to a clinical
setting. The target audience are healthcare engineers, researcher who doesn't
have background in software development or software engineering.

It intends to provide a one stop shop for taking research projects and moving
them into a clinical setting. It hides the complexities of dicom protocol from
the user letting them focus on what is important, namely the processing aspect.

If something should be in two different pipelines, then it should belong to this
library.

Tutorials are focus on writing pipelines can be found at
:doc:`tutorials/tutorial_index`

While documentation on high level design of the library and the backend of the
library can be found at :doc:`documentation/documentation`. This is intended for
maintainers rather than users. While at the same times doesn't assume anything
about the skill level of the maintainer.

This software is developed at Rigshospitalet's Department of Clinical Physiology
and Nuclear Medicine.

Source code can be found at: https://github.com/Rigshospitalet-KFNM/DicomNode

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   tutorials/tutorial_index
   documentation/documentation
   dicomnode

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
