# Responsibilities of the abstract pipeline

## Introduction

This document explains the work flow of the abstract pipeline class. It covers
all the weird edge cases, weird hacks, and so forth.

## Start up

You create an Pipeline object and then you can open it with it's open method.
This is mainly for testing / the python way.

So the server is a bread and butter pynetdicom server which accepts C-ECHO and
C-STORE. In the early days I thought about adding a cache, which could be
accessed with C-FIND / C-MOVE / C-GET, but GDPR kinda killed this. A man can
dream.

Either way the this works that the server is the main thread, and new
connections are their own threads. However each threads operate on the same
object the `data_state` so locks are kinda needed.
Maaaaybe this should be fixed...

As C-ECHO doesn't alter the state they are irrelevant.

### Events

Okay Pynetdicom have their own list of events, however they don't play very nice
with the build-in type systems. So
`dicomnode.server.factories-association_events.py` is a wrapper class around
pynetdicom events. The AssociationEvents contains less information than the
pynetdicom event, but everything that should be needed.

## Accept Events

## C-STORES

## Release Events

Okay so release events are trigger after the C-STORE association is finished.

The call stack is rather fucked it goes:

* `handle_association_released`
* `_release_store_handler`
  * `_extract_input_containers`
* `_process_entry_point`
* `_pipeline_processing`
  * `process`
  * `dispatch`

okay so the top functions is just to point to the `_release_store_handler`
because it might be a C-echo or C-find or what ever. In those cases you don't
wanna run code.

The `_release_store_handler` then calls `_extract_input_containers` to extracts
all the data.