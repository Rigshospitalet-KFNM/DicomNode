=======================================
Threads, Python and the meaning of pain
=======================================

An introduction to threads
--------------------------

To start with lets get some terminology out of the way. From the top a
program can be viewed as one or more threads and a memory space referred to
as the state of the program. A Thread is the thing that execute statements of
the program. These statements modify the state such that a desired side effect
of the program take effect, be it display a cat face on the screen, write a
file or perform some mathematical computation.

Sometimes programs require some level of multitasking. Take any form of server.
If your program only have one thread, then you can't handle any new requests,
while you're processing a request, which is problematic. That is why in the
server pattern, you will have a number of listener threads, that "handles" a
request by creating a thread and then go back to listening. It is the created
threads job to handle the request. That way a server can handle multiple
requests at the same time.

However that also means that you have multiple threads manipulating the state of
the program at the same time. If a programmer is not careful, then two threads
may modify the same object with a not associative operation. In other words the state
is dependant on the execution order of the threads. This is called a
**race condition**, and you need the greatest wizard hat in existence to say
isn't undefined behavior.

In general threads are difficult work with, because it's very difficult to
determine what the programs state is when you have multiple threads running at
the same time. For example the object a thread is working with might have been
modified by another thread, like another thread inserted an item into dictionary
while the current thread was iterating over it. This causes the main thread to
crash per python.

Python have something called the Global Interpreter Lock, which ensures that
only one thread can modify the state of program at a time. This means we can
think of the possible execution of our program. If we have a single thread, then
the execution path is the ordering of statement as the programmer wrote them.
If you have two threads then you have :math:`\binom(n+m,n)` different execution
paths where :math:`n` and :math:`m` is the number of statement in each thread.
Now this isn't a
problem unless different execution paths leads to different states i.e.
The program has a race condition. The difficult part is a majority of program
executions leads to an expected state, and therefore the race condition are not
guarantied to appear unlike an illegal memory access, or a raised Exception.

Of cause if only one thread is executing instructions at a time, then this
implies a state of the threads, which is either active or sleeping.
Now whether a thread is active or sleeping is ultimately controlled by the
operating system of the machine, when the active thread is switched, we say that
operating system have performed a context switch. They will often happen when
the computer determines the current operation might take a very long time. For
instance the program might request some data on disk, this will likely take a
few milliseconds. Modern CPU is able to process billions of instructions per
seconds. So if you have delay of 5ms and processor running at 3 Ghz, then your
program have 15 million instructions that potentially is wasted. Make use of
these instructions the operating system would perform a context switch to
another thread, that is able to execute and make use of them.

The greatest issue with threading, race conditions and dead locks is that: It is
very difficult to abstract away, and often leave behind some very weird looking
code. The second issue is that problems can be difficult or expensive to
reproduce. Here expensive means require "take a long time to compute".

Tools of threading
******************

While programmers can't control when a thread is executing, we can control
when threads are sleeping. In python we use the `Threading`_ module which
contain threads, locks and barriers.

Threads in Dicomnode
--------------------

Enough boring context. An dicom SCP is a server, and given my rant, therefore
that means a thread is being created on each association, which can give some
problems if both threads are attempting to manipulate the same patient.

This can trigger some unclear exceptions, that normally in a single threaded
environment cannot trigger. For instance if you are iterating over a dictionary,
then python doesn't allow you to modify that dictionary until iteration is done.
While your thread might respect this, if another thread add an element to the
dictionary while you are iterating over it. BOOM.

A Dicomnode pipeline doesn't spawn relevant threads itself directly but instead
relies on pynetdicom to spawn threads relevant for dicom communication. Each
association is a live thread, both then sending and receiving, but we only
concern ourselves with the receiving thread here. Pynetdicom allows us to define
a function that are called when different events happen. Because there are quite
a few events. We'll only go through the default used events however you can see
the full list at `InterventionEvent`_ and `NotificationEvent`_

When a tcp connection is made, a thread is spawned. It is the plan that this
thread should perform everything or more specific: 


* Filtering out unused dataset
* Adding valid ones
* Determining if there is sufficient for processing
* If there is, perform the processing
* Export the result from the processing
* Clean up after a processing



For a dicomnode, where should caution be around the data state or the pipeline
tree. This is the place where datasets are added

.. toctree::
    :hidden:


.. _Threading: https://docs.python.org/3.9/library/threading.html
.. _InterventionEvent: https://pydicom.github.io/pynetdicom/stable/reference/generated/pynetdicom.events.InterventionEvent.html#pynetdicom.events.InterventionEvent
.. _NotificationEvent: https://pydicom.github.io/pynetdicom/stable/reference/generated/pynetdicom.events.NotificationEvent.html#pynetdicom.events.NotificationEvent

