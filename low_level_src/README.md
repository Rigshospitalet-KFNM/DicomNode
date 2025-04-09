# Introduction

Welcome to the fucking jungle. Please read the GPU Programming Document from the
docs. I would also like to apologize for this code. It's my first non trivial
cuda/c++ project. So the structure of the project might be fucked, and I don't
really have a good sparing partner.

Some lingo:
 * An *N* dimensional extent is *N* positive whole numbers, which defines the
limits of the data stores.
 * An Image is a combination of an volume and a Space.
 * A Volume is N-dimensional extent which describes a GPU allocated continuous
 memory region which is inside of the Image.
 * A N-dimensional Space is a N dimensional starting point point and a N
 Dimensional basis.
 * A Texture is a volume, where the data is stored in texture memory

Rules:
  All classes must be default constructable with all zeroes!
