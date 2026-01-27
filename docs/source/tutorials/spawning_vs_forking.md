# The Quick fix

Sadly from version 0.0.21 breaks userspace. Here os the quick fix:

```python

# Version <0.0.21

class Pipeline(AbstractPipeline):
  ...

  attribute = ...

  def process(self, input_container):
    self.attribute
    ...

# Version 0.0.21 and beyond

from dicomnode.server.processor import Processor

ATTRIBUTE = ...


    ...

class Pipeline(AbstractPipeline):
  class Processor(AbstractProcessor): # Very import to name it Processor!
    def process(self, input_container):
      ATTRIBUTE


```

Note that there's some new edge-cases that might hit you, although I consider
this unlikely

## The in-depth explanation

The remain of this document will just explain what changed and why. It contains
no additional "user" documentation, but if you want to know what happens under
the hood, that is what the remaining of this document will devote itself to.

## The root of the problem

A process has a "current working directory" which you can get with a:
`os.getcwd()` call. This is the reference point for all non absolute files look
up a program makes. In other words, if you do a `nibabel.load("file.nii.gz")`
the operating system will look in the current working directory and look for a
file names `file.nii.gz` in the current working directory and return from there.

Now Dicomnode assumes that you do not clean after yourself, meaning if you
produce intermediate files, then dicomnode cleans them up, when you are done
processing the patient.

The way it does this is by creating a directory, then changing its current
working directory to the newly created directory. When the processing is
complete dicomnode deletes the directory and all the intermediate files that
have been produced.

As stated above a process only has a single current working directory that means
if another process sends data to the dicomnode, the archiving will be incorrect
because the newly send files would be stored inside of a temporary directory,
which is bad.

So the way that dicomnode maintains this illusion is by spawning a new process
for processing the data data. That way the program maintains this illusion.

## A small lesson on os-primitives

So how does operating system create a new process? There's two primitives for
doing this:

* `spawn` : Create a new runtime and run the program in it.
* `fork` : The newly created process gets a read only view of the old process.
  If it changes anything, then the operating system will create a new copy of
  the data.

Now `fork` have been the "default" for spawning new processes because it's much
faster. See most programs have large amount of "read-only" data. Think about
libraries like numpy, scipy, matplotlib, but also operating system libraries
like the C standard library, rendering engines and so forth.

To put in measurable terms. A `spawn` call takes an additional ~1.5 seconds per
process created compared to fork, but this number may be even worse if you need
to initialize tensorflow or pytorch for instance.

## The fork problem

Dicomnode have been using `fork` up to version 0.0.20, but this is introduces some
problems.

First of all - It's depreciated, meaning that python will in the future no
longer support it.

But more importantly with `fork` dicomnode might deadlock. When forking,
everything is copied including other threads! Now this is a pretty big problem,
because these threads might be locked. They might be waiting on acquiring a lock
to store a dataset in the dicomnode or post a log record. If the thread is
copied in a locked state the operating system will not wake the thread. This
means, that the child process will never finish, and because dicomnode waits
on the process to finish, it will wait forever and therefore deadlock.

## The fundamental differences between fork and spawn

Sadly `fork` and `spawn` are different in functionality. With the `fork` call
you have access to the memory and the previous instruction so you can do this:

```python

def target(arg):
  arg.call()

class Arg
  def __init__(self):
    def local_function():
      print("hello world")

    self.call = local_function



arg = Arg()

multiprocessing.get_context('fork').process(target=target, args=(arg,)) # OK
multiprocessing.get_context('spawn').process(target=target, args=(arg,)) # NOT OK

```

You cannot send an object that have locks to new processes. Dicomnode have a
neat little function `dicomnode.lib.utils.is_picklable` you can use to check if
you can send an object to another process.
