---
title: Dicom for dummies
author: Christoffer Vilstrup Jensen
---

# Dicom For Dummies

Digital Imaging and Communications in Medicine is a file format and a message protocol owned and partly developed by the National Electrical Manufacturers Association (NEMA).
It's primary file format for medical image equipment. An Analogy is dicom is to medical equipment, what a pdf is to a printer.

The Dicom standard what an should be capable of doing and can be found at:

**https://www.dicomstandard.org/current**

If you click a little bit around in it, you'll find that it's big, verbose and not very readable. This is because it not very restrictive, yet at the same time tries to standardize optional content. It also means there's A LOT of details and finical details that'll be missing for the guide. This guide is mostly focused around usage of Dicom files, and thus many technical details will be left out. 

A warning: Images produced by medical equipment may be in the dicom format, but might not comply with the dicom standard. It's your application responsibility to check, that the images you receive and produce comply with the dicom standard. There's a number of tools in library to help with this job.

## The Dicom file

A dicom file is a dictionary with integers keys and just about anything as values. The standard is this mapping between tags and values. For instance the tag: `0x0010010` means the patients Name. So if a program conform to the dicom standard it'll read and write the patient's name to and from the value associated with the tag `0x00100010`. All tags are in the range of `0` to `4294967295`, or in hex `0xFFFFFFFF`. Most tags have been restricted by the standard.

Along a value associated to a tag is a value representation **(VR)**, which tells how the program should interpret the ones and zeroes forming the value. For instance the patient name tag have a `VR` of `PN`. Which means that the program should assume is formatted as string. However the dicom standard also imposes additional restrictions upon that string: Namely it should be formatted as: family name complex, given name complex, middle name, name prefix, name suffix. Where each component is separated by a '^' character.

It also specifies that a each component is a maximum of 64 characters. 

From experience this is the most often broken part of the dicom standard, where programs simply just store the name as a string, instead of formatting it.

All the Value representation can be found at **https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html**

Finally a tag also have a value multiplicity **(VM)**. This indicates how many instances of the values should exists. For instance the tag `0x00280034` describes how what the aspect ratio of the underlying picture. So if the stored picture is HD, then it has a 16:9 aspect ratio and the values for the tag `0x00280034` should be set to `[16,9]`. A value multiplicity might be a range. For instance if a tag has value multiplicity of 1-n that means there can be any number of values associated with the tag.

Finally a tag might be present but value is associated with that, but more on that later. 

### Sequences

There's a few very special VR, and one of them is the Sequence VR `SQ`. A Sequence is a list of zero or more dicom objects stores inside of the tag. This is often used to store associated values, which doesn't fit within a single tag. It's clearly specified in the standard that each dicom object of a sequence may have different tags, however for your own sanity make sure that each object of a sequence have the same tags.

A Sequence always have a VM of 1. 

### UID & SOP Classes

Most dicom objects have a few unique identifiers **(UID)** which determines something uniquely about the image.


