# Class overview

This library contains numerous python class for you to use and a few extension to those classes, for common use cases. This files describes the various classes and their use cases.

## Library

This module contains most useful functions for manipulation of dicom objects.

### Dicom Factory

* DicomFactory - Abstract class made for constructing dicom datasets. Cannot instantiate SeriesHeaders.
* Blueprint - Class representing a blueprint for a dataset and is made op of VirtualElements. A blueprint can instantiated to a dataset or a SeriesHeader from another Dicom dataset, through a Dicom factory.
* SeriesHeader - Class representing a partially instantiated dicom dataset. Can be instantiated through the factory to a Series of Dicom images, when given some image data, not on the dicom format.

#### VirtualElements

VirtualElement is base abstract class representing a tag in a blueprint. A Virtual element can produce a Data Element or a method to produce a Data Element by "corporealializing".
There's the following VirtualElements:

* StaticElement - Represent an data element with static value.
* CopyElement - Represent a data element, that will be copied from the handed dataset.
* AttrElement - Reads an attribute from the Factory and copies it. Often useful when restriction are applied to the tag.
* DiscardElement - Element will be discarded from dataset.
* CallElement - Produces a method which will produce a DataElement when called. The value is not shared between datasets in the same series
* SeriesElement - Produces a method which will produce a DataElement when called. The value is shared between datasets in the same series

### NumpyFactory

* NumpyFactory - DicomFactory specialized in numpy arrays
* NumpyCaller - CallElement created to handle numpy arrays.

### Lazy Datasets

* LazyDataset - Dataset on the file system, Can be used as a normal dataset, while miniscule memory footprint until used i.e. it's lazy. 

### Image Tree

* ImageTreeInterface - Interface for creating a tree to store Dicom images. Each subclass must specify how they add images.
* SeriesTree - Contains images from the same series. Throws Error on duplication or when attempting to add images from another series
* StudyTree - Contains images from the same study. Creates SeriesTrees to store images.
* PatientTree - Contains images Belonging to the same patient. Creates StudyTrees to store images.
* DicomTree - Contains PatientTree, Create PatientTree when a new patient is encountered.
* IdentityMapping - A mapping of UIDs, useful for anonymizing.

## Server

This module contains classes for building an medical image processing pipeline.

### Input

* AbstractInput - An ImageTreeInterface abstract class build for containing all the images of an input to a pipeline.
* HistoricAbstractInput - An AbstractInput which sends a C_move upon being instantiated. I.E when the pipeline receives data about the patient for the first time.

### Nodes

* AbstractPipeline - Base class for constructing an image class.
* AbstractThreadedPipeline - AbstractPipeline, but spawns a thread for storing images and always returns successful storage.
* AbstractQueuedPipeline - AbstractPipeline, but commits work to a queue, instead of processing it. Useful when process require resources, that cannot be easily shared such as large neural networks, or when ordering of processing matter.

### Output

* PipelineOutput - Base class for defining an output to a pipeline.
* NoOutput - PipelineOutput for indicating the pipeline produces no output
* DicomOutput - PipelineOutput for sending dicom image using the dicom protocol
* FileOutput - PipelineOutput for saving to the file system. Uses ImageTreeInterface's save method.

### PipelineTree

* InputContainer - A glorified dict with additional data
* PatientContainer - An ImageTreeInterface that manages all Abstract input.
* PipelineTree - An ImageTreeInterface that manages all PatientContainers.
