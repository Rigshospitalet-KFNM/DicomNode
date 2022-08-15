# DicomNode
This is the library for setting up a DicomNode af Rigshospitalet

#### Introduction 

The DICOM node software will be a complete tool for sending and receiving DICOM data that can be implemented on processing servers. The tool will use access control so that only approved data will pe processed.  

#### MDR 

Think of something clever to write here! 

#### Functionality 

The tool handles data transfer of DICOM data and will offer either to store data in a structured file tree or read the data into memory directly upon receival. 

The tool will be able to send response messages after receiving/sending data. 

#### Code 

The tool will be written in python 3+ and kept under version control on GITHUB or similar platforms. 

### Whitelisting 

####AE Title 

Both on the receiving and sending part of the DICOM node a list of allowed sources/destinations should be implemented. This allows for better restriction of data. 

#### Data types 

A list of allowed data types could be based on modality or study/series description etc. 

#### Logging 

A log containing useful information about source/destination with info about what data has been transferred and from where will be kept. 

#### Dispatch 

The tool could also allow for a dispatch option where it receives and immediately sends data to other, perhaps multiple, sources. Could be used for data backup or as a relay node for optimal use of processing resources. 

#### Cleanup 

The tool could additionally allow for a data cleanup functionality that removes data after sending or after a certain time period. This might be a functionality that should be built into the processing pipelines instead. 
